# src/data/download.py

import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def load_config(config_path: str = "configs/config.yaml"):
    return OmegaConf.load(config_path)


def download_esci(cfg) -> None:
    raw_dir = ROOT / cfg.paths.data_raw
    raw_dir.mkdir(parents=True, exist_ok=True)

    pairs_out    = raw_dir / "pairs_raw.csv"
    products_out = raw_dir / "products_raw.csv"

    if pairs_out.exists() and products_out.exists():
        print("Raw data already exists. Skipping download.")
        return

    print("\n[1/2] Downloading train split...")
    train_df = load_dataset(
        "tasksource/esci", split="train", trust_remote_code=True
    ).to_pandas()

    print("[2/2] Downloading test split...")
    test_df = load_dataset(
        "tasksource/esci", split="test", trust_remote_code=True
    ).to_pandas()

    full_df = pd.concat([train_df, test_df], ignore_index=True)
    full_df = full_df[
        full_df["product_locale"] == cfg.dataset.locale
    ].reset_index(drop=True)

    print(f"\n  Total rows after locale filter: {len(full_df):,}")

    # save pairs
    pairs_df = full_df[["query_id", "query", "product_id", "esci_label"]].copy()
    pairs_df.to_csv(pairs_out, index=False)
    print(f"  Saved {len(pairs_df):,} pairs → {pairs_out}")

    # save products (deduplicated)
    product_cols = [
        "product_id", "product_title", "product_bullet_point",
        "product_brand", "product_color", "product_description",
    ]
    products_df = (
        full_df[product_cols]
        .drop_duplicates(subset="product_id")
        .reset_index(drop=True)
    )
    products_df.to_csv(products_out, index=False)
    print(f"  Saved {len(products_df):,} unique products → {products_out}")


def explore(cfg) -> None:
    raw_dir     = ROOT / cfg.paths.data_raw
    pairs_df    = pd.read_csv(raw_dir / "pairs_raw.csv")
    products_df = pd.read_csv(raw_dir / "products_raw.csv")

    print("\n" + "="*55)
    print("ESCI DATASET — EXPLORATION SUMMARY")
    print("="*55)

    print(f"\n  Total pairs     : {len(pairs_df):,}")
    print(f"  Unique queries  : {pairs_df['query_id'].nunique():,}")
    print(f"  Unique products : {pairs_df['product_id'].nunique():,}")

    print("\n  Label distribution:")
    vc = pairs_df["esci_label"].value_counts()
    for label, count in vc.items():
        pct = count / len(pairs_df) * 100
        bar = "█" * int(pct / 2)
        print(f"    {label:<12} {count:>8,}  ({pct:5.1f}%)  {bar}")

    print("\n  Missing values in products:")
    for col in ["product_title", "product_bullet_point",
                "product_brand", "product_color", "product_description"]:
        n = products_df[col].isna().sum()
        print(f"    {col:<25}: {n:>6,} missing ({n/len(products_df)*100:.1f}%)")

    print("\n  Sample product titles:")
    for i, title in enumerate(
        products_df["product_title"].dropna().sample(3, random_state=42), 1
    ):
        print(f"    {i}. {title[:75]}...")

    pair_pids      = set(pairs_df["product_id"].astype(str))
    catalogue_pids = set(products_df["product_id"].astype(str))
    orphans        = pair_pids - catalogue_pids
    print(f"\n  Products in pairs but not catalogue: {len(orphans):,}")
    if not orphans:
        print("  ✓ All pair products found in catalogue.")

    print("="*55)


if __name__ == "__main__":
    cfg = load_config()
    download_esci(cfg)
    explore(cfg)