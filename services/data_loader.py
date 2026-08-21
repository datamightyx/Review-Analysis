"""Data loading service for Returns Analysis - SP-API integration."""

import csv
import gzip
import io
import time
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
    MONEY_BACK_EVENT_LISTS,
    ASIN_UNMAPPED,
)

# Columns the returns report must carry for reconciliation to work at all
REQUIRED_RETURN_COLS = [
    "order-id", "sku", "asin", "product-name", "reason", "customer-comments",
]
# Columns used when present, blank-filled when the report omits them
OPTIONAL_RETURN_COLS = ["detailed-disposition", "status"]

# Full column set of a reconciled row, so returns and refund-only rows concat cleanly
RETURN_ROW_COLS = [
    "return-date", "order-id", "sku", "asin", "product-name", "quantity",
    "reason", "customer-comments", "detailed-disposition", "status",
    "refund-amount", "refund-source",
]
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

    missing = [c for c in REQUIRED_RETURN_COLS if c not in df.columns]
    if missing:
        raise DataError(f"Report missing expected columns: {missing}")

    if "return-date" in df.columns:
        df["return-date"] = pd.to_datetime(
            df["return-date"], errors="coerce", utc=True
        ).dt.tz_localize(None)
    else:
        df["return-date"] = pd.NaT

    # A row in the returns report is at least one physical unit, so a missing or
    # unparseable quantity falls back to 1 rather than silently contributing 0.
    if "quantity" in df.columns:
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(1)
    else:
        df["quantity"] = 1

    for col in OPTIONAL_RETURN_COLS:
        if col not in df.columns:
            df[col] = pd.NA

    # Present only on refund-only rows; kept here so the two frames concat cleanly
    df["refund-amount"] = 0.0
    df["refund-source"] = pd.NA

    return df


def _principal_amount(item: Dict) -> float:
    """Sum the Principal charge adjustment for one refunded item.

    Finances reports money leaving the seller as a negative amount. The sign is
    preserved so that reversals net against the original refund.
    """
    total = 0.0
    for charge in item.get("ItemChargeAdjustmentList") or []:
        if charge.get("ChargeType") == "Principal":
            amount = (charge.get("ChargeAmount") or {}).get("CurrencyAmount")
            if amount is not None:
                total += float(amount)
    return total


def _extract_money_back_rows(payload: Dict) -> List[Dict]:
    """Pull SKU-level money-back rows out of one financialEvents page.

    Covers plain refunds plus A-to-Z guarantee claims and chargebacks: all three
    put money back in the customer's hands and all three are ShipmentEvent-shaped,
    so one parser handles them.
    """
    events = payload.get("FinancialEvents") or {}
    rows: List[Dict] = []
    for list_name, source in MONEY_BACK_EVENT_LISTS.items():
        for event in events.get(list_name) or []:
            for item in event.get("ShipmentItemAdjustmentList") or []:
                rows.append({
                    "order_id": event.get("AmazonOrderId"),
                    "sku": item.get("SellerSKU"),
                    "quantity": item.get("QuantityShipped"),
                    "posted": event.get("PostedDate"),
                    "amount": _principal_amount(item),
                    "source": source,
                })
    return rows


def fetch_refund_events(
    cfg: Dict[str, str],
    start: str,
    end: str,
    release_lag_days: int = 7,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    """
    Fetch money-back events (refunds, A-to-Z claims, chargebacks) from Finances API.

    Pages are walked strictly sequentially. NextToken is a cursor: the token for
    page N+1 only exists once page N has been fetched, so there is nothing here
    that can be parallelised.

    Args:
        cfg: SP-API configuration
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        release_lag_days: Extra days to catch late-released refunds
        progress_callback: Called with (page_num, total_rows)

    Returns:
        DataFrame with columns order_id, sku, quantity, posted, amount, source
    """
    access_token = get_access_token(cfg)
    url = f"{cfg['SP_API_ENDPOINT']}{FINANCES_PATH}"
    headers = {"x-amz-access-token": access_token}

    upper = pd.Timestamp(end) + pd.Timedelta(days=release_lag_days, hours=23, minutes=59)
    now_utc = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(minutes=5)
    upper = min(upper, now_utc)

    params = {
        "MaxResultsPerPage": 100,
        "PostedAfter": f"{start}T00:00:00Z",
        "PostedBefore": upper.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    all_rows: List[Dict] = []
    page = 0
    next_token: Optional[str] = None

    while True:
        if next_token:
            params = {"NextToken": next_token}
        resp = _make_request_with_retry(url, params, headers)
        payload = resp.json()["payload"]

        all_rows.extend(_extract_money_back_rows(payload))

        page += 1
        if progress_callback:
            progress_callback(page, len(all_rows))

        next_token = payload.get("NextToken")
        if not next_token:
            break
        time.sleep(FINANCES_API_RATE_LIMIT_DELAY)

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


def _normalize_refund_frame(ref: pd.DataFrame) -> pd.DataFrame:
    """Normalize refund DataFrame.

    Negative quantities are refund reversals. They are kept so they can net
    against the positive rows during reconciliation; dropping them here would
    overcount refunds that Amazon later took back.
    """
    cols = ["order_id", "sku", "quantity", "posted", "amount", "source"]
    if ref.empty:
        return pd.DataFrame(columns=cols)

    ref = ref.copy()
    for col in ("amount", "source"):
        if col not in ref.columns:
            ref[col] = 0.0 if col == "amount" else pd.NA

    ref["quantity"] = pd.to_numeric(ref["quantity"], errors="coerce").fillna(0)
    ref["amount"] = pd.to_numeric(ref["amount"], errors="coerce").fillna(0.0)
    ref = ref[ref["quantity"] != 0]
    ref["posted"] = pd.to_datetime(ref["posted"], errors="coerce", utc=True).dt.tz_localize(None)
    return ref[cols].reset_index(drop=True)


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

    # "product sales" is the CSV's equivalent of the Principal charge adjustment
    if "product sales" in ref.columns:
        amount = pd.to_numeric(
            ref["product sales"].astype(str).str.replace(r"[^\d.\-]", "", regex=True),
            errors="coerce",
        ).fillna(0.0)
    else:
        amount = pd.Series(0.0, index=ref.index)

    return pd.DataFrame({
        "order_id": ref["order id"].values,
        "sku": ref["sku"].values,
        "quantity": ref["quantity"].values,
        "posted": posted.values,
        "amount": amount.values,
        "source": "Refund (CSV)",
    })


def refunds_to_return_rows(
    ref: pd.DataFrame,
    returns_df: pd.DataFrame,
    verbose: bool = True
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Convert money-back events into return-like rows for units that never came back.

    Reconciliation is by units, not by mere presence of a key. For each
    (order-id, SKU) the physically returned quantity is subtracted from the
    refunded quantity and only the positive residual becomes a row. An order with
    2 units physically returned but 3 refunded therefore contributes 1
    refund-only unit, where a presence-based match would have dropped it whole.

    SKUs that cannot be mapped to an ASIN are kept under the ASIN_UNMAPPED
    sentinel rather than discarded, so their units still reach the report. They
    are also returned in the second element for reporting.
    """
    if ref.empty:
        return pd.DataFrame(columns=RETURN_ROW_COLS), []

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

    # Units physically received back, per (order, SKU)
    ret_units = (
        returns_df.assign(
            _q=pd.to_numeric(returns_df["quantity"], errors="coerce").fillna(1)
        )
        .groupby(["order-id", "sku"])["_q"]
        .sum()
    )

    # Net the money-back events per key first, so reversals cancel originals
    agg = (
        ref.groupby(["order_id", "sku"], dropna=False)
        .agg(
            quantity=("quantity", "sum"),
            amount=("amount", "sum"),
            posted=("posted", "min"),
            source=("source", "first"),
        )
        .reset_index()
    )

    returned = [
        ret_units.get((o, s), 0) for o, s in zip(agg["order_id"], agg["sku"])
    ]
    agg["quantity"] = agg["quantity"] - returned
    agg = agg[agg["quantity"] > 0]

    if agg.empty:
        return pd.DataFrame(columns=RETURN_ROW_COLS), []

    unmapped_mask = ~agg["sku"].isin(sku_asin)
    skipped = sorted(agg[unmapped_mask]["sku"].dropna().unique().tolist())
    if skipped and verbose:
        print(
            "  note: refund SKUs with no ASIN in returns report, kept as "
            f"{ASIN_UNMAPPED}: {skipped}"
        )

    rows = pd.DataFrame({
        "return-date": agg["posted"].values,
        "order-id": agg["order_id"].values,
        "sku": agg["sku"].values,
        "asin": agg["sku"].map(sku_asin).fillna(ASIN_UNMAPPED).values,
        "product-name": agg["sku"].map(sku_name).fillna(agg["sku"]).values,
        "quantity": agg["quantity"].values,
        "reason": REFUND_NO_RETURN,
        "customer-comments": pd.NA,
        "detailed-disposition": pd.NA,
        "status": pd.NA,
        "refund-amount": agg["amount"].values,
        "refund-source": agg["source"].values,
    })
    return rows[RETURN_ROW_COLS], skipped


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