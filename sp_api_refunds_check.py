"""
Pull refund counts straight from SP-API and compare with locally exported reports.

Fetches GET_SALES_AND_TRAFFIC_REPORT (the same "Refunds" metric shown in the
Business Reports UI screenshot) for a date range, plus optionally
GET_FBA_REIMBURSEMENTS_DATA / GET_FBA_FULFILLMENT_CUSTOMER_RETURNS_DATA /
GET_DATE_RANGE_FINANCIAL_TRANSACTION_DATA, and prints refunded units per SKU/ASIN.

Auth: raw SP-API REST calls (LWA bearer token only, no AWS SigV4 — not
required for standard Reports API operations since 2023). Credentials read
from .env via python-dotenv.

Usage:
    python sp_api_refunds_check.py --start 2026-05-01 --end 2026-08-14
    python sp_api_refunds_check.py --start 2026-05-01 --end 2026-08-14 --report-type GET_FBA_REIMBURSEMENTS_DATA
"""

import argparse
import gzip
import io
import os
import sys
import time

import requests
from dotenv import load_dotenv

load_dotenv()

LWA_TOKEN_URL = "https://api.amazon.com/auth/o2/token"

REQUIRED_ENV = [
    "SP_API_LWA_CLIENT_ID",
    "SP_API_LWA_CLIENT_SECRET",
    "SP_API_REFRESH_TOKEN",
    "SP_API_MARKETPLACE_ID",
    "SP_API_ENDPOINT",
]


def get_config():
    missing = [k for k in REQUIRED_ENV if not os.environ.get(k)]
    if missing:
        sys.exit(f"Missing .env values: {', '.join(missing)}")
    return {k: os.environ[k] for k in REQUIRED_ENV}


def get_access_token(cfg):
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
    resp.raise_for_status()
    return resp.json()["access_token"]


def create_report(cfg, access_token, report_type, start, end):
    url = f"{cfg['SP_API_ENDPOINT']}/reports/2021-06-30/reports"
    body = {
        "reportType": report_type,
        "marketplaceIds": [cfg["SP_API_MARKETPLACE_ID"]],
        "dataStartTime": f"{start}T00:00:00Z",
        "dataEndTime": f"{end}T23:59:59Z",
    }
    if report_type == "GET_SALES_AND_TRAFFIC_REPORT":
        # without this, SP-API may only return date-level rows, no per-SKU breakdown
        body["reportOptions"] = {"asinGranularity": "SKU"}
    resp = requests.post(url, json=body, headers={"x-amz-access-token": access_token}, timeout=30)
    resp.raise_for_status()
    return resp.json()["reportId"]


def poll_report(cfg, access_token, report_id, interval=20, timeout=1800):
    url = f"{cfg['SP_API_ENDPOINT']}/reports/2021-06-30/reports/{report_id}"
    waited = 0
    while waited <= timeout:
        resp = requests.get(url, headers={"x-amz-access-token": access_token}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        status = data["processingStatus"]
        print(f"  report {report_id}: {status} ({waited}s elapsed)")
        if status == "DONE":
            return data["reportDocumentId"]
        if status in ("FATAL", "CANCELLED"):
            sys.exit(f"Report {report_id} failed: {status}")
        time.sleep(interval)
        waited += interval
    sys.exit(f"Report {report_id} did not finish within {timeout}s")


def download_report_document(cfg, access_token, document_id):
    url = f"{cfg['SP_API_ENDPOINT']}/reports/2021-06-30/documents/{document_id}"
    resp = requests.get(url, headers={"x-amz-access-token": access_token}, timeout=30)
    resp.raise_for_status()
    doc = resp.json()

    file_resp = requests.get(doc["url"], timeout=120)
    file_resp.raise_for_status()
    raw = file_resp.content

    if doc.get("compressionAlgorithm") == "GZIP":
        raw = gzip.decompress(raw)

    return raw.decode("utf-8")


def summarize_sales_and_traffic(raw_json_text):
    import json

    data = json.loads(raw_json_text)
    by_asin = data.get("salesAndTrafficByAsin", [])
    if not by_asin:
        print("No salesAndTrafficByAsin rows in report.")
        print("Raw report keys:", list(data.keys()))
        return

    print("Sample raw row (to verify field names):")
    print(json.dumps(by_asin[0], indent=2))

    # SP-API can return >1 row per SKU (e.g. mid-period parent-ASIN reparenting
    # splits the period into separate aggregates) — sum them, don't take one row.
    from collections import defaultdict

    totals = defaultdict(lambda: {"ordered": 0, "refunded": 0, "row_count": 0})
    for row in by_asin:
        sales = row.get("salesByAsin", {})
        sku = row.get("sku", "")
        asin = row.get("childAsin", row.get("parentAsin", ""))
        key = (sku, asin)
        totals[key]["ordered"] += sales.get("unitsOrdered", row.get("unitsOrdered", 0))
        totals[key]["refunded"] += sales.get("unitsRefunded", row.get("unitsRefunded", 0))
        totals[key]["row_count"] += 1

    print(f"\n{'SKU':25}{'ASIN':15}{'unitsOrdered':>14}{'unitsRefunded':>15}{'rows summed':>13}")
    for (sku, asin), v in sorted(totals.items(), key=lambda kv: -kv[1]["refunded"]):
        note = f"{v['row_count']}" + (" *" if v["row_count"] > 1 else "")
        print(f"{sku:25}{asin:15}{v['ordered']:>14}{v['refunded']:>15}{note:>13}")
    if any(v["row_count"] > 1 for v in totals.values()):
        print("\n* SKU appeared in multiple rows — summed. Likely mid-period ASIN reparenting.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument(
        "--report-type",
        default="GET_SALES_AND_TRAFFIC_REPORT",
        help="SP-API report type (default: GET_SALES_AND_TRAFFIC_REPORT)",
    )
    parser.add_argument("--save-raw", help="optional path to save the raw report document")
    args = parser.parse_args()

    cfg = get_config()
    print("Requesting LWA access token...")
    access_token = get_access_token(cfg)

    print(f"Creating report {args.report_type} for {args.start}..{args.end}...")
    report_id = create_report(cfg, access_token, args.report_type, args.start, args.end)

    print("Polling report status...")
    document_id = poll_report(cfg, access_token, report_id)

    print("Downloading report document...")
    content = download_report_document(cfg, access_token, document_id)

    if args.save_raw:
        with open(args.save_raw, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Saved raw report to {args.save_raw}")

    if args.report_type == "GET_SALES_AND_TRAFFIC_REPORT":
        summarize_sales_and_traffic(content)
    else:
        print(content[:2000])
        print("... (use --save-raw to capture full output for non-JSON/CSV report types)")


if __name__ == "__main__":
    main()
