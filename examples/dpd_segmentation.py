"""End-to-end example: segment a synthetic loan book on 90+ DPD risk.

Run from the repo root:

    python examples/dpd_segmentation.py

Produces a printed segment summary and writes ``dpd_tree.png``.
"""
import numpy as np
import pandas as pd

from chaid_segmenter import ChaidSegmenter


def make_loan_book(n=8000, seed=42):
    """Synthesize a loan book where risk concentrates in a clear segment."""
    rng = np.random.RandomState(seed)
    age = rng.uniform(18, 70, n)
    income = rng.lognormal(mean=6.4, sigma=0.5, size=n)        # monthly income
    tenure = rng.uniform(0, 72, n)                             # months on book
    score = rng.uniform(300, 850, n)                           # bureau score
    region = rng.choice(["Phnom Penh", "Siem Reap", "Battambang"], n,
                         p=[0.55, 0.25, 0.20])
    bank = rng.choice(["ABA", "ACLEDA", "Wing"], n)

    # Flagship risk pocket: young, Phnom Penh, banks with ABA -> high 90+DPD.
    risk = np.full(n, 0.06)
    risk += np.where(age < 25, 0.20, 0.0)
    risk += np.where(region == "Phnom Penh", 0.10, 0.0)
    risk += np.where(bank == "ABA", 0.12, 0.0)
    risk += np.where(score < 550, 0.15, 0.0)
    risk = np.clip(risk, 0, 0.95)
    dpd90 = (rng.uniform(0, 1, n) < risk).astype(int)

    return pd.DataFrame({
        "age": age, "income": income, "tenure": tenure, "score": score,
        "region": region, "bank": bank, "dpd90": dpd90,
    })


def main():
    df = make_loan_book()
    print("Loan book: {} accounts, overall 90+DPD = {:.1%}\n".format(
        len(df), df["dpd90"].mean()))

    seg = ChaidSegmenter(
        target="dpd90",
        positive_class=1,
        predictors={
            "age":    {"method": "target", "max_bins": 4},        # supervised
            "income": {"method": "equal_width", "bins": 4},       # fixed interval
            "tenure": {"method": "equal_frequency", "bins": 4},   # quantile
            "score":  {"method": "manual", "edges": [550, 650, 750]},
            "region": {"method": "nominal"},
            "bank":   {"method": "nominal"},
        },
        max_depth=3,
        min_child_node_size=0.02,
        alpha_merge=0.05,
    )
    seg.fit(df)

    summary = seg.summary()
    pd.set_option("display.width", 140)
    pd.set_option("display.max_colwidth", 80)
    print("=== Segments (highest 90+DPD rate first) ===")
    print(summary.to_string(index=False, formatters={
        "population": lambda v: "{:,}".format(int(v)),
        "population_pct": "{:.1%}".format,
        "rate": "{:.1%}".format,
        "lift": "{:.2f}x".format,
    }))

    print("\n=== Flagship high-risk segment ===")
    top = seg.segments()[0]
    print("{}\n  -> {:.0%} 90+DPD rate, {:.0%} of population (n={:,}), {:.2f}x lift".format(
        top.description, top.rate, top.population_pct, int(top.population), top.lift))

    out = "dpd_tree.png"
    seg.plot(out)
    print("\nSaved tree visualisation to {}".format(out))


if __name__ == "__main__":
    main()
