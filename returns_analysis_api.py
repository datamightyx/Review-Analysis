"""
Pull FBA customer returns straight from SP-API and build the same
mismatch-analysis Excel report as pages/1_Returns_Analysis.py.

Fetches GET_FBA_FULFILLMENT_CUSTOMER_RETURNS_DATA (tab-delimited: return-date,
order-id, sku, asin, fnsku, product-name, quantity, fulfillment-center-id,
detailed-disposition, reason, status, license-plate-number, customer-comments),
then runs the comment-vs-stated-reason mismatch analysis and writes an .xlsx
with a SUMMARY sheet plus one sheet per ASIN.

Auth: raw SP-API REST calls (LWA bearer token only). Credentials read from
.env via python-dotenv (same vars as sp_api_refunds_check.py).

Usage:
    python returns_analysis_api.py --start 2026-05-01 --end 2026-08-14
    python returns_analysis_api.py --start 2026-05-01 --end 2026-08-14 --output MyReport.xlsx
"""

import argparse
import os
import sys

import pandas as pd
from dotenv import load_dotenv

from services import (
    REQUIRED_ENV,
    get_access_token,
    fetch_customer_returns,
    parse_customer_returns,
    normalize_returns_frame,
    fetch_refund_events,
    parse_transaction_refunds,
    refunds_to_return_rows,
    load_refund_only_rows,
    build_workbook,
    save_workbook,
    ValidationError,
    DataError,
    ReportError,
)

load_dotenv()


def get_config():
    missing = [k for k in REQUIRED_ENV if not os.environ.get(k)]
    if missing:
        sys.exit(f"Missing .env values: {', '.join(missing)}")
    return {k: os.environ[k] for k in REQUIRED_ENV}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--start", help="YYYY-MM-DD (required unless --local-returns)")
    parser.add_argument("--end", help="YYYY-MM-DD (required unless --local-returns)")
    parser.add_argument("--output", default="Returns_Analysis.xlsx", help="output .xlsx path")
    parser.add_argument("--save-raw", help="optional path to save the raw TSV report")
    parser.add_argument(
        "--transactions",
        help="Unified Transaction report CSV; adds refunds that have no physical "
             "return so totals match the Business Reports refund count",
    )
    parser.add_argument(
        "--local-returns",
        help="use this returns report file instead of calling SP-API (tab-separated TSV or XLSX)",
    )
    parser.add_argument(
        "--release-lag",
        type=int,
        default=7,
        help="Days to pad end date for Finances API release lag (default: 7)",
    )
    args = parser.parse_args()

    # Load returns data
    if args.local_returns:
        ext = os.path.splitext(args.local_returns)[1].lower()
        if ext in (".xlsx", ".xls"):
            raw = None
            df_local = pd.read_excel(args.local_returns, dtype=str)
        else:
            with open(args.local_returns, "rb") as f:
                data = f.read()
            try:
                raw = data.decode("utf-8")
            except UnicodeDecodeError:
                raw = data.decode("cp1252")
            df_local = None
    else:
        if not (args.start and args.end):
            parser.error("--start and --end are required unless --local-returns is used")
        
        # Validate date range
        try:
            from services import validate_date_range
            validate_date_range(args.start, args.end)
        except ValidationError as e:
            sys.exit(str(e))
        
        cfg = get_config()
        raw = fetch_customer_returns(cfg, args.start, args.end)
        df_local = None

    if args.save_raw and raw is not None:
        with open(args.save_raw, "w", encoding="utf-8", newline="") as f:
            f.write(raw)
        print(f"Saved raw report to {args.save_raw}")

    if df_local is not None:
        df = normalize_returns_frame(df_local)
    else:
        df = parse_customer_returns(raw)
    
    if df.empty:
        sys.exit("No customer returns rows in report for this date range.")

    print(f"Parsed {len(df)} physical return rows across {df['asin'].nunique()} ASINs.")

    # Add refunds if requested
    if args.transactions:
        extra = load_refund_only_rows(args.transactions, df)
        print(f"Adding {len(extra)} refunds with no matching physical return "
              f"(returnless refunds / never shipped back / not yet scanned).")
        df = pd.concat([df, extra], ignore_index=True)
    elif not args.local_returns:
        # Fetch from Finances API
        cfg = get_config()
        print("Fetching refund events from Finances API...")
        ref = fetch_refund_events(
            cfg, args.start, args.end,
            release_lag_days=args.release_lag,
        )
        print(f"Retrieved {len(ref)} refund events, {int(ref['quantity'].sum())} units")
        extra, skipped = refunds_to_return_rows(ref, df, verbose=True)
        print(f"Adding {len(extra)} refunds with no matching physical return, "
              f"{int(extra['quantity'].sum()) if len(extra) else 0} units")
        if skipped:
            print(f"SKUs with no ASIN mapping (kept under a placeholder ASIN): {skipped}")
        df = pd.concat([df, extra], ignore_index=True)

    print("Building analysis workbook...")
    wb = build_workbook(df)
    save_workbook(wb, args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()