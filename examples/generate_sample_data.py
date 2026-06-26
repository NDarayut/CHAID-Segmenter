"""Generate a synthetic loan book for testing ``chaid_segmenter``.

Fields produced (column names kept verbatim from the request):

    ACCOUNT_ID, CUSOMER_ID, MEMBER_ID, BANK_ID, BANK_NAME, PRODUCT_TYPE, VINTAGE,
    LOAN_CATEGORY, TENURE_QUANTITY, DISBURSEMENT_AMOUNT, 90PLSDPD_MOB6

``BANK_ID`` and ``BANK_NAME`` are a 1:1 pair (each bank has one id and one name).

``MEMBER_ID`` mimics a real credit-bureau institution code: a *high-cardinality*
categorical ("001".."200") shown as an id, not a name, where some members carry
more risk than others. Plain nominal CHAID can't handle hundreds of categories, so
feed it with ``{"method": "target"}`` (or just list it) to group members into a few
target-rate tiers.

Note: ``MEMBER_ID`` is text in-memory, but ``pd.read_csv`` will parse "001".."200"
back as integers. Read it as text (``pd.read_csv(path, dtype={"MEMBER_ID": str})``)
or force grouping with ``{"method": "target", "categorical": True}`` so it is treated
as an id, not a number.

``90PLSDPD_MOB6`` is the binary KPI/target (1 = went 90+ days past due by
month-on-book 6). It is driven by several of the other fields so the CHAID tree
discovers meaningful, non-random segments.

Usage
-----
    python examples/generate_sample_data.py                 # writes sample_loan_book.csv
    python examples/generate_sample_data.py --rows 20000 --out book.csv
    python examples/generate_sample_data.py --demo          # also run a segmentation
"""
import argparse

import numpy as np
import pandas as pd

PRODUCT_TYPES = ["Term Loan", "Credit Card", "Auto Loan", "Mortgage", "Overdraft", "Personal Loan"]
LOAN_CATEGORIES = ["Secured", "Unsecured", "Micro", "Salary"]
# (BANK_ID, BANK_NAME) pairs. AMK and Hattha are micro-finance institutions.
BANKS = [
    ("BNK01", "ABA"),
    ("BNK02", "ACLEDA"),
    ("BNK03", "Wing"),
    ("BNK04", "Prince Bank"),
    ("BNK05", "AMK"),
    ("BNK06", "Hattha"),
]
VINTAGES = [f"{y}Q{q}" for y in range(2021, 2025) for q in range(1, 5)]  # 2021Q1 .. 2024Q4
TENURES = [6, 12, 18, 24, 36, 48, 60, 72]                                 # months
N_MEMBERS = 200                                                           # institution codes


def generate_loan_book(rows=10000, seed=42, missing_frac=0.015):
    """Return a synthetic loan book as a ``pandas.DataFrame``."""
    rng = np.random.default_rng(seed)
    n = rows

    account_id = [f"ACC{i:08d}" for i in range(1, n + 1)]
    # A customer may hold several accounts -> fewer customers than accounts.
    n_customers = max(1, int(n * 0.75))
    customer_id = [f"CUST{c:07d}" for c in rng.integers(1, n_customers + 1, n)]

    # High-cardinality institution code "001".."200"; ~15% of members run hot.
    member_risk = rng.normal(0.0, 0.03, N_MEMBERS + 1)
    hot = rng.random(N_MEMBERS + 1) < 0.15
    member_risk[hot] += rng.uniform(0.08, 0.18, hot.sum())
    member_num = rng.integers(1, N_MEMBERS + 1, n)
    member_id = np.array([f"{m:03d}" for m in member_num])

    bank_idx = rng.choice(len(BANKS), n, p=[0.28, 0.24, 0.16, 0.12, 0.10, 0.10])
    bank_id = [BANKS[i][0] for i in bank_idx]
    bank_name = np.array([BANKS[i][1] for i in bank_idx])

    product_type = rng.choice(PRODUCT_TYPES, n, p=[0.30, 0.12, 0.18, 0.10, 0.10, 0.20])
    loan_category = rng.choice(LOAN_CATEGORIES, n, p=[0.45, 0.25, 0.15, 0.15])
    vintage = rng.choice(VINTAGES, n)
    tenure = rng.choice(TENURES, n, p=[0.10, 0.18, 0.14, 0.20, 0.18, 0.10, 0.06, 0.04])
    disbursement = np.round(rng.lognormal(8.6, 0.7, n) / 50.0) * 50.0      # ~K riel/USD, to nearest 50
    disbursement = np.clip(disbursement, 300, 80000)

    # --- risk model for 90+ DPD by MOB6 ---------------------------------
    p = np.full(n, 0.04)
    p += (loan_category == "Unsecured") * 0.12
    p += (loan_category == "Micro") * 0.08
    p += np.isin(product_type, ["Credit Card", "Personal Loan"]) * 0.07
    p += (disbursement > 20000) * 0.06
    p += (tenure <= 12) * 0.05
    p += np.isin(vintage, ["2022Q1", "2022Q2"]) * 0.05          # stressed cohort
    p += np.isin(bank_name, ["AMK", "Hattha"]) * 0.07           # MFI: higher early DPD
    p += member_risk[member_num]                                # per-institution risk
    p += ((loan_category == "Unsecured") & (disbursement > 20000)) * 0.08   # interaction
    p += rng.normal(0, 0.02, n)                                  # idiosyncratic noise
    p = np.clip(p, 0.01, 0.92)
    dpd = (rng.random(n) < p).astype(int)

    df = pd.DataFrame({
        "ACCOUNT_ID": account_id,
        "CUSOMER_ID": customer_id,
        "MEMBER_ID": member_id,
        "BANK_ID": bank_id,
        "BANK_NAME": bank_name,
        "PRODUCT_TYPE": product_type,
        "VINTAGE": vintage,
        "LOAN_CATEGORY": loan_category,
        "TENURE_QUANTITY": tenure,
        "DISBURSEMENT_AMOUNT": disbursement,
        "90PLSDPD_MOB6": dpd,
    })

    # Sprinkle a few missing disbursement amounts to exercise the "or missing" branch.
    if missing_frac:
        miss = rng.random(n) < missing_frac
        df.loc[miss, "DISBURSEMENT_AMOUNT"] = np.nan

    return df


def run_demo(df):
    """Show how to segment the generated data with the package."""
    from chaid_segmenter import ChaidSegmenter

    # List the predictors to use. MEMBER_ID is a high-cardinality institution code:
    # because it is listed (not full-auto) the segmenter groups it by the target rate
    # via optbinning instead of dropping it. ACCOUNT_ID / CUSOMER_ID are left out.
    seg = ChaidSegmenter(
        target="90PLSDPD_MOB6",
        positive_class=1,
        predictors=[
            "MEMBER_ID",            # high-cardinality id -> supervised risk tiers
            "BANK_NAME", "PRODUCT_TYPE", "VINTAGE", "LOAN_CATEGORY",
            "TENURE_QUANTITY", "DISBURSEMENT_AMOUNT",
        ],
        max_depth=3,
        min_child_node_size=0.02,
    ).fit(df)

    print("\nPredictor methods:",
          {k: (v if isinstance(v, str) else v["method"]) for k, v in seg.resolved_predictors.items()})
    print("\n=== Segments (highest 90+DPD rate first) ===")
    print(seg.summary().to_string(index=False, formatters={
        "population": lambda v: f"{int(v):,}",
        "population_pct": "{:.1%}".format, "rate": "{:.1%}".format, "lift": "{:.2f}x".format}))
    seg.plot("loan_segments.png")
    print("\nSaved tree visualisation to loan_segments.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, default=10000, help="number of accounts (default 10000)")
    parser.add_argument("--seed", type=int, default=42, help="random seed (default 42)")
    parser.add_argument("--out", default="sample_loan_book.csv", help="output CSV path")
    parser.add_argument("--demo", action="store_true", help="also run a ChaidSegmenter demo")
    args = parser.parse_args()

    df = generate_loan_book(args.rows, args.seed)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} rows x {df.shape[1]} columns to {args.out}")
    print(f"Overall 90PLSDPD_MOB6 rate: {df['90PLSDPD_MOB6'].mean():.1%}\n")
    print(df.head().to_string(index=False))

    if args.demo:
        run_demo(df)


if __name__ == "__main__":
    main()
