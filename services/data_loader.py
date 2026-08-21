"""Data loading service for Returns Analysis - SP-API integration."""

import csv
import gzip
import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import requests

from services.constants import (
    LWA_TOKEN_URL,
    RETURNS_REPORT_TYPE,
    FINANCES_PATH,
    REPORT_POLL_INTERVAL,
    REPORT_POLL_TIMEOUT,
    FINANCES_API_RATE_LIMIT_DELAY,
    REFUND_NO_RETURN,
    MAX_DATE_RANGE_DAYS,
)
from services.exceptions import (
    AuthenticationError,
    APIError,
    RateLimitError,
    ReportError,
    DataError,
    ValidationError,
)


def get_access_token(cfg: Dict[str, str]) -> str:
    """Get LWA access token using refresh token."""
    resp = requests.post(
        LWA_TOKEN_URL,
        data={
            "grant_type": "refresh_token",
            "refresh_token": cfg["SP_API_REFRESH_TOKEN"],
            "client_id": cfg["SP_API_LWA_CLIENT_ID"],
            "client_secret": cfg["SP_API_LWA_CLIENT_SECRET"],
        },
        timeout=30,
    )
    if resp.status_code == 400:
        raise AuthenticationError("Invalid refresh token or client credentials")
    if resp.status_code == 429:
        raise RateLimitError("Rate limited getting access token", status_code=429)
    resp.raise_for_status()
    return resp.json()["access_token"]


def create_report(cfg: Dict[str, str], access_token: str, start: str, end: str) -> str:
    """Create a returns report."""
    url = f"{cfg['SP_API_ENDPOINT']}/reports/2021-06-30/reports"
    body = {
        "reportType": RETURNS_REPORT_TYPE,
        "marketplaceIds": [cfg["SP_API_MARKETPLACE_ID"]],
        "dataStartTime": f"{start}T00:00:00Z",
        "dataEndTime": f"{end}T23:59:59Z",
    }
    resp = requests.post(
        url, 
        json=body, 
        headers={"x-amz-access-token": access_token}, 
        timeout=30
    )
    if resp.status_code == 429:
        raise RateLimitError("Rate limited creating report", status_code=429)
    resp.raise_for_status()
    return resp.json()["reportId"]


def poll_report(
    cfg: Dict[str, str], 
    access_token: str, 
    report_id: str,
    interval: int = REPORT_POLL_INTERVAL,
    timeout: int = REPORT_POLL_TIMEOUT,
    progress_callback: Optional[Callable[[str, int], None]] = None
) -> str:
    """Poll report until complete."""
    url = f"{cfg['SP_API_ENDPOINT']}/reports/2021-06-30/reports/{report_id}"
    waited = 0
    while waited <= timeout:
        resp = requests.get(
            url, 
            headers={"x-amz-access-token": access_token}, 
            timeout=30
        )
        if resp.status_code == 429:
            raise RateLimitError("Rate limited polling report", status_code=429)
        resp.raise_for_status()
        data = resp.json()
        status = data["processingStatus"]
        
        if progress_callback:
            progress_callback(status, waited)
        
        if status == "DONE":
            return data["reportDocumentId"]
        if status in ("FATAL", "CANCELLED"):
            raise ReportError(
                f"Report {report_id} failed", 
                report_id=report_id, 
                status=status
            )
        time.sleep(interval)
        waited += interval
    
    raise ReportError(
        f"Report {report_id} did not finish within {timeout}s",
        report_id=report_id,
        status="TIMEOUT"
    )


def download_report_document(
    cfg: Dict[str, str], 
    access_token: str, 
    document_id: str
) -> str:
    """Download and decompress report document."""
    url = f"{cfg['SP_API_ENDPOINT']}/reports/2021-06-30/documents/{document_id}"
    resp = requests.get(url, headers={"x-amz-access-token": access_token}, timeout=30)
    if resp.status_code == 429:
        raise RateLimitError("Rate limited getting document URL", status_code=429)
    resp.raise_for_status()
    doc = resp.json()

    file_resp = requests.get(doc["url"], timeout=120)
    file_resp.raise_for_status()
    raw = file_resp.content

    if doc.get("compressionAlgorithm") == "GZIP":
        raw = gzip.decompress(raw)

    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp1252")


def fetch_customer_returns(cfg: Dict[str, str], start: str, end: str) -> str:
    """Fetch customer returns report from SP-API."""
    access_token = get_access_token(cfg)
    report_id = create_report(cfg, access_token, start, end)
    document_id = poll_report(cfg, access_token, report_id)
    return download_report_document(cfg, access_token, document_id)


def parse_customer_returns(raw_tsv_text: str) -> pd.DataFrame:
    """Parse raw TSV returns report into DataFrame."""
    df = pd.read_csv(
        io.StringIO(raw_tsv_text), 
        sep="\t", 
        dtype=str, 
        quoting=csv.QUOTE_NONE
    )
    return normalize_returns_frame(df)


def normalize_returns_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize returns DataFrame columns and types."""
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()

    required = ["asin", "product-name", "reason", "customer-comments"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataError(f"Report missing expected columns: {missing}")

    if "return-date" in df.columns:
        df["return-date"] = pd.to_datetime(
            df["return-date"], errors="coerce", utc=True
        ).dt.tz_localize(None)
    else:
        df["return-date"] = pd.NaT

    if "quantity" in df.columns:
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    else:
        df["quantity"] = None

    return df


def fetch_refund_events(
    cfg: Dict[str, str],
    start: str,
    end: str,
    release_lag_days: int = 7,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    max_workers: int = 3
) -> pd.DataFrame:
    """
    Fetch refund events from Finances API with parallel page fetching.
    
    Args:
        cfg: SP-API configuration
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        release_lag_days: Extra days to catch late-released refunds
        progress_callback: Called with (page_num, total_rows)
        max_workers: Max parallel workers for page fetching
    
    Returns:
        DataFrame with refund events
    """
    access_token = get_access_token(cfg)
    url = f"{cfg['SP_API_ENDPOINT']}{FINANCES_PATH}"
    headers = {"x-amz-access-token": access_token}

    upper = pd.Timestamp(end) + pd.Timedelta(days=release_lag_days, hours=23, minutes=59)
    now_utc = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(minutes=5)
    upper = min(upper, now_utc)

    # First page to get total count and next tokens
    params = {
        "MaxResultsPerPage": 100,
        "PostedAfter": f"{start}T00:00:00Z",
        "PostedBefore": upper.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    
    all_rows: List[Dict] = []
    next_tokens: List[Optional[str]] = [None]
    page = 0
    
    # Collect all next tokens first (sequential, to get all tokens)
    while True:
        if next_tokens[-1]:
            params = {"NextToken": next_tokens[-1]}
        resp = _make_request_with_retry(url, params, headers)
        payload = resp.json()["payload"]
        
        for event in payload.get("FinancialEvents", {}).get("RefundEventList", []):
            for item in event.get("ShipmentItemAdjustmentList") or []:
                all_rows.append({
                    "order_id": event.get("AmazonOrderId"),
                    "sku": item.get("SellerSKU"),
                    "quantity": item.get("QuantityShipped"),
                    "posted": event.get("PostedDate"),
                })
        
        page += 1
        if progress_callback:
            progress_callback(page, len(all_rows))
        
        next_token = payload.get("NextToken")
        if not next_token:
            break
        next_tokens.append(next_token)
        time.sleep(FINANCES_API_RATE_LIMIT_DELAY)
    
    # If we have multiple pages, fetch remaining in parallel
    if len(next_tokens) > 1:
        _fetch_pages_parallel(
            url, headers, next_tokens[1:], all_rows, progress_callback, max_workers
        )
    
    return _normalize_refund_frame(pd.DataFrame(all_rows))


def _make_request_with_retry(
    url: str, 
    params: Dict, 
    headers: Dict,
    max_retries: int = 3
) -> requests.Response:
    """Make request with retry on rate limit."""
    for attempt in range(max_retries):
        resp = requests.get(url, params=params, headers=headers, timeout=60)
        if resp.status_code == 429:
            wait_time = 2 ** attempt * 5  # Exponential backoff: 5, 10, 20s
            time.sleep(wait_time)
            continue
        resp.raise_for_status()
        return resp
    # Final attempt
    resp = requests.get(url, params=params, headers=headers, timeout=60)
    if resp.status_code == 429:
        raise RateLimitError("Rate limit exceeded after retries", status_code=429)
    resp.raise_for_status()
    return resp


def _fetch_pages_parallel(
    url: str,
    headers: Dict,
    tokens: List[str],
    all_rows: List[Dict],
    progress_callback: Optional[Callable[[int, int], None]],
    max_workers: int
) -> None:
    """Fetch multiple pages in parallel."""
    def fetch_page(token: str) -> List[Dict]:
        params = {"NextToken": token}
        resp = _make_request_with_retry(url, params, headers)
        payload = resp.json()["payload"]
        rows = []
        for event in payload.get("FinancialEvents", {}).get("RefundEventList", []):
            for item in event.get("ShipmentItemAdjustmentList") or []:
                rows.append({
                    "order_id": event.get("AmazonOrderId"),
                    "sku": item.get("SellerSKU"),
                    "quantity": item.get("QuantityShipped"),
                    "posted": event.get("PostedDate"),
                })
        return rows
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_token = {executor.submit(fetch_page, token): token for token in tokens}
        for future in as_completed(future_to_token):
            rows = future.result()
            all_rows.extend(rows)
            if progress_callback:
                progress_callback(0, len(all_rows))  # Update count
            time.sleep(FINANCES_API_RATE_LIMIT_DELAY / max_workers)


def _normalize_refund_frame(ref: pd.DataFrame) -> pd.DataFrame:
    """Normalize refund DataFrame."""
    if ref.empty:
        return pd.DataFrame(columns=["order_id", "sku", "quantity", "posted"])
    
    ref["quantity"] = pd.to_numeric(ref["quantity"], errors="coerce").fillna(0)
    ref = ref[ref["quantity"] > 0]
    ref["posted"] = pd.to_datetime(ref["posted"], errors="coerce", utc=True).dt.tz_localize(None)
    return ref.reset_index(drop=True)


def parse_transaction_refunds(transactions_path: str) -> pd.DataFrame:
    """Parse refund rows from Unified Transaction CSV."""
    import re
    
    with open(transactions_path, "rb") as f:
        raw = f.read()
    
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw.decode("utf-8", errors="replace")

    m = re.search(r'^"?date/time"?\s*,', text, re.IGNORECASE | re.MULTILINE)
    if m is None:
        raise DataError(f"Could not find 'date/time' header row in {transactions_path}")

    tx = pd.read_csv(io.StringIO(text[m.start():]), dtype=str)
    tx.columns = [c.strip() for c in tx.columns]
    tx["quantity"] = pd.to_numeric(tx["quantity"], errors="coerce").fillna(0)

    ref = tx[tx["type"].astype(str).str.strip().str.lower() == "refund"].copy()
    ref = ref[ref["quantity"] > 0]
    
    posted = pd.to_datetime(
        ref["date/time"].str.replace(r"\s+P[DS]T$", "", regex=True),
        format="%b %d, %Y %I:%M:%S %p",
        errors="coerce",
    )
    
    return pd.DataFrame({
        "order_id": ref["order id"].values,
        "sku": ref["sku"].values,
        "quantity": ref["quantity"].values,
        "posted": posted.values,
    })


def refunds_to_return_rows(
    ref: pd.DataFrame, 
    returns_df: pd.DataFrame,
    verbose: bool = True
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Convert refund events with no matching return into return-like rows.
    
    Match key is (order_id, SKU). SKUs not in returns report can't be mapped to ASIN.
    """
    sku_asin = (
        returns_df.dropna(subset=["sku", "asin"])
        .groupby("sku")["asin"]
        .agg(lambda s: s.value_counts().index[0])
        .to_dict()
    )
    sku_name = (
        returns_df.dropna(subset=["sku", "product-name"])
        .groupby("sku")["product-name"]
        .first()
        .to_dict()
    )

    ret_keys = set(zip(returns_df["order-id"], returns_df["sku"]))
    ref = ref[[(o, s) not in ret_keys for o, s in zip(ref["order_id"], ref["sku"])]]

    skipped = sorted(ref[~ref["sku"].isin(sku_asin)]["sku"].dropna().unique().tolist())
    if skipped and verbose:
        print(f"  note: refund SKUs with no ASIN in returns report, skipped: {skipped}")
    ref = ref[ref["sku"].isin(sku_asin)]

    rows = pd.DataFrame({
        "return-date": ref["posted"].values,
        "order-id": ref["order_id"].values,
        "sku": ref["sku"].values,
        "asin": ref["sku"].map(sku_asin).values,
        "product-name": ref["sku"].map(sku_name).values,
        "quantity": ref["quantity"].values,
        "reason": REFUND_NO_RETURN,
        "customer-comments": pd.NA,
    })
    return rows, skipped


def load_refund_only_rows(transactions_path: str, returns_df: pd.DataFrame) -> pd.DataFrame:
    """CLI path: Unified Transaction CSV -> refund rows with no matching return."""
    ref = parse_transaction_refunds(transactions_path)
    rows, _ = refunds_to_return_rows(ref, returns_df)
    return rows


def validate_date_range(start: str, end: str) -> None:
    """Validate date range inputs."""
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    
    if start_dt > end_dt:
        raise ValidationError("Start date must be before end date")
    
    if (end_dt - start_dt).days > MAX_DATE_RANGE_DAYS:
        raise ValidationError(
            f"Date range exceeds {MAX_DATE_RANGE_DAYS} days maximum "
            f"(Finances API limitation)"
        )