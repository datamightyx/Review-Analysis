"""
Reconcile FBA customer returns (physical returns) against refund events
(money returned) and explain the count gap per ASIN.

Why this exists
---------------
GET_FBA_FULFILLMENT_CUSTOMER_RETURNS_DATA only contains units that physically
came back and were scanned at an FC. The "Refunds" number shown in Business
Reports (and the Refund rows in the Unified Transaction report) counts money
refunded, which also includes:
  * returnless refunds (Amazon refunds, customer keeps the item),
  * refunds where the customer never shipped the item back,
  * refunds where the return is still in transit / not yet scanned,
  * FC-lost returns that get reimbursed instead of received.
So refunds >= returns, always. Comment-based mismatch analysis can only run on
the returns subset, because only physical returns carry customer-comments.

Inputs
------
--returns       raw TSV saved by returns_analysis_api.py --save-raw, or the
                Seller Central .xlsx / .csv export (same columns)
--transactions  Unified Transaction report CSV (Date Range Reports)
--reimbursements optional FBA reimbursements CSV (adds SKU->ASIN mapping and
                CustomerReturn reimbursement counts)
--output        optional .xlsx with the reconciliation table

Usage
-----
    python returns_refunds_reconcile.py \
        --returns raw_returns.tsv \
        --transactions 2026May1-2026Aug14CustomUnifiedTransaction.csv \
        --reimbursements 111938020683.csv \
        --output Returns_vs_Refunds.xlsx
"""

import argparse
import csv
import io
import os
import re
import sys

import pandas as pd

TX_HEADER_MARKER = "date/time"


# ── loading ────────────────────────────────────────────────────────────────

def read_text(path):
    with open(path, "rb") as f:
        raw = f.read()
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def load_returns(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path, dtype=str)
    else:
        text = read_text(path)
        sep = "\t" if "\t" in text.split("\n", 1)[0] else ","
        # customer-comments are unquoted free text that can start with a quote
        # char; QUOTE_NONE keeps one report line == one dataframe row
        df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str, quoting=csv.QUOTE_NONE)
    df.columns = df.columns.str.strip().str.lower().str.strip('"')
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].str.strip('"')
    df["return-date"] = pd.to_datetime(df["return-date"], errors="coerce", utc=True).dt.tz_localize(None)
    df["quantity"] = pd.to_numeric(df.get("quantity"), errors="coerce").fillna(1)
    return df


def load_transactions(path):
    """Unified Transaction CSV: skip the preamble, find the real header row."""
    text = read_text(path)
    # slice on the raw text, not splitlines(): data rows contain quoted fields
    # with embedded newlines, so line-splitting would corrupt them
    m = re.search(rf'^"?{re.escape(TX_HEADER_MARKER)}"?\s*,', text, re.IGNORECASE | re.MULTILINE)
    if m is None:
        sys.exit(f"{path}: could not find a '{TX_HEADER_MARKER}' header row")
    df = pd.read_csv(io.StringIO(text[m.start():]), dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df["dt"] = pd.to_datetime(
        df["date/time"].str.replace(r"\s+P[DS]T$", "", regex=True),
        format="%b %d, %Y %I:%M:%S %p",
        errors="coerce",
    )
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0)
    return df


def refund_events(tx):
    r = tx[tx["type"].astype(str).str.strip().str.lower() == "refund"].copy()
    # a refund row with quantity 0 is a fee-only / tax-only adjustment line
    r = r[r["quantity"] > 0]
    return r


# ── reconciliation ─────────────────────────────────────────────────────────

def build_sku_asin_map(returns, reimb=None):
    m = returns.dropna(subset=["sku", "asin"]).groupby("sku")["asin"].agg(
        lambda s: s.value_counts().index[0]
    ).to_dict()
    if reimb is not None and {"sku", "asin"} <= set(reimb.columns):
        for sku, asin in reimb.dropna(subset=["sku", "asin"])[["sku", "asin"]].values:
            m.setdefault(sku, asin)
    return m


def reconcile(returns, refunds, sku_asin):
    refunds = refunds.copy()
    refunds["asin"] = refunds["sku"].map(sku_asin)
    unmapped = sorted(refunds[refunds["asin"].isna()]["sku"].dropna().unique().tolist())

    ret_keys = set(zip(returns["order-id"], returns["sku"]))
    refunds["has_return"] = [
        (o, s) in ret_keys for o, s in zip(refunds["order id"], refunds["sku"])
    ]

    ref_keys = set(zip(refunds["order id"], refunds["sku"]))
    returns = returns.copy()
    returns["has_refund"] = [
        (o, s) in ref_keys for o, s in zip(returns["order-id"], returns["sku"])
    ]

    rows = []
    for asin in sorted(set(returns["asin"].dropna()) | set(refunds["asin"].dropna())):
        ret = returns[returns["asin"] == asin]
        ref = refunds[refunds["asin"] == asin]
        rows.append({
            "ASIN": asin,
            "SKU": (ret["sku"].iloc[0] if len(ret) else ref["sku"].iloc[0]),
            "Returned units": int(ret["quantity"].sum()),
            "Return lines": len(ret),
            "Refund events": len(ref),
            "Refunded units": int(ref["quantity"].sum()),
            "Refunds matched to return": int(ref["has_return"].sum()),
            "Refunds with NO return": int((~ref["has_return"]).sum()),
            "Returns with NO refund": int((~ret["has_refund"]).sum()),
            "Gap (refunded - returned units)": int(ref["quantity"].sum() - ret["quantity"].sum()),
        })
    table = pd.DataFrame(rows).sort_values("Refunded units", ascending=False)
    return table, unmapped, refunds, returns


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--returns", required=True)
    ap.add_argument("--transactions", required=True)
    ap.add_argument("--reimbursements")
    ap.add_argument("--output", help="optional .xlsx output path")
    args = ap.parse_args()

    returns = load_returns(args.returns)
    tx = load_transactions(args.transactions)
    refunds = refund_events(tx)

    reimb = None
    if args.reimbursements:
        reimb = pd.read_csv(io.StringIO(read_text(args.reimbursements)), dtype=str)
        reimb.columns = reimb.columns.str.strip().str.lower()

    sku_asin = build_sku_asin_map(returns, reimb)
    table, unmapped, refunds, returns = reconcile(returns, refunds, sku_asin)

    print(f"Returns window (return-date, UTC):  {returns['return-date'].min()}  ..  "
          f"{returns['return-date'].max()}   ({len(returns)} lines)")
    print(f"Refunds window (posted date, PT):   {refunds['dt'].min()}  ..  "
          f"{refunds['dt'].max()}   ({len(refunds)} events)")
    if unmapped:
        print(f"WARNING: refund SKUs with no ASIN mapping (excluded from per-ASIN rows): {unmapped}")
    print()
    print(table.to_string(index=False))
    print()
    print(f"TOTAL returned units: {table['Returned units'].sum()}   "
          f"refunded units: {table['Refunded units'].sum()}   "
          f"gap: {table['Gap (refunded - returned units)'].sum()}")

    if reimb is not None and "reason" in reimb.columns:
        cr = reimb[reimb["reason"] == "CustomerReturn"]
        print(f"\nCustomerReturn reimbursements (return never made it back): {len(cr)} rows, "
              f"{cr['asin'].nunique()} ASINs")

    if args.output:
        no_return = refunds[~refunds["has_return"]][
            ["dt", "order id", "sku", "asin", "quantity", "description"]
        ].rename(columns={"dt": "refund date"})
        with pd.ExcelWriter(args.output, engine="openpyxl") as xl:
            table.to_excel(xl, sheet_name="RECONCILIATION", index=False)
            no_return.to_excel(xl, sheet_name="REFUND_NO_RETURN", index=False)
        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
