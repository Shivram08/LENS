# src/data/download.py

import json
import os
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from omegaconf import OmegaConf
from tqdm import tqdm

# ── make src/ importable when run as a script ──────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def load_config(config_path: str = "configs/config.yaml"):
    return OmegaConf.load(config_path)


def download_esci(cfg) -> None:
    """
    Downloads the ESCI dataset from HuggingFace and saves raw CSVs.

    Saves:
        data/raw/train_raw.csv        ← query-product pairs with labels
        data/raw/product_catalogue.csv ← all product metadata
    """
    raw_dir = ROOT / cfg.paths.data_raw
    raw_dir.mkdir(parents=True, exist_ok=True)

    train_out = raw_dir / "train_raw.csv"
    catalogue_out = raw_dir / "product_catalogue.csv"

    # ── skip if already downloaded ─────────────────────────────────────────
    if train_out.exists() and catalogue_out.exists():
        print("Raw data already exists. Skipping download.")
        print(f"  train_raw.csv      : {train_out}")
        print(f"  product_catalogue  : {catalogue_out}")
        return

    # ── download pairs (train split) ───────────────────────────────────────
    print("\n[1/2] Downloading query-product pairs...")
    pairs_ds = load_dataset(
        "tasksource/esci",
        split="train",          # HuggingFace 'train' split = our train+val+test pool
        trust_remote_code=True,
    )
    pairs_df = pairs_ds.to_pandas()

    # keep only US locale — consistent language & product style
    pairs_df = pairs_df[pairs_df["product_locale"] == cfg.dataset.locale].reset_index(drop=True)

    # keep only the columns we need
    pairs_df = pairs_df[[
        "query_id",
        "query",
        "product_id",
        "esci_label",
    ]]

    pairs_df.to_csv(train_out, index=False)
    print(f"  Saved {len(pairs_df):,} pairs → {train_out}")
    print(f"  Label distribution:\n{pairs_df['esci_label'].value_counts().to_string()}")

    # ── download product catalogue ──────────────────────────────────────────
    print("\n[2/2] Downloading product catalogue...")
    catalogue_ds = load_dataset(
        "tasksource/esci",
        split="test",           # HuggingFace 'test' split = product catalogue
        trust_remote_code=True,
    )
    catalogue_df = catalogue_ds.to_pandas()

    # keep only US locale
    catalogue_df = catalogue_df[
        catalogue_df["product_locale"] == cfg.dataset.locale
    ].reset_index(drop=True)

    # keep only the columns we need
    catalogue_df = catalogue_df[[
        "product_id",
        "product_title",
        "product_bullet_point",
        "product_brand",
        "product_color_name",
        "product_image_url",    # we'll download images using these URLs
    ]].drop_duplicates(subset="product_id").reset_index(drop=True)

    catalogue_df.to_csv(catalogue_out, index=False)
    print(f"  Saved {len(catalogue_df):,} unique products → {catalogue_out}")


def explore(cfg) -> None:
    """
    Prints a structured summary of the raw data so we understand
    what we're working with before building splits.
    """
    raw_dir = ROOT / cfg.paths.data_raw
    pairs_df = pd.read_csv(raw_dir / "train_raw.csv")
    catalogue_df = pd.read_csv(raw_dir / "product_catalogue.csv")

    print("\n" + "="*60)
    print("ESCI DATASET — EXPLORATION SUMMARY")
    print("="*60)

    print("\n── PAIRS ──────────────────────────────────────────────────")
    print(f"  Total pairs          : {len(pairs_df):,}")
    print(f"  Unique queries       : {pairs_df['query_id'].nunique():,}")
    print(f"  Unique products      : {pairs_df['product_id'].nunique():,}")

    print("\n  Label distribution:")
    vc = pairs_df["esci_label"].value_counts()
    for label, count in vc.items():
        pct = count / len(pairs_df) * 100
        bar = "█" * int(pct / 2)
        print(f"    {label}  {count:>7,}  ({pct:5.1f}%)  {bar}")

    print("\n── PRODUCT CATALOGUE ──────────────────────────────────────")
    print(f"  Total unique products: {len(catalogue_df):,}")

    # check for missing values in key fields
    print("\n  Missing values:")
    for col in ["product_title", "product_bullet_point",
                "product_brand", "product_image_url"]:
        n_missing = catalogue_df[col].isna().sum()
        pct = n_missing / len(catalogue_df) * 100
        print(f"    {col:<30}: {n_missing:>6,} missing ({pct:.1f}%)")

    # sample a few rows so we see what the text looks like
    print("\n  Sample product titles:")
    samples = catalogue_df["product_title"].dropna().sample(5, random_state=42)
    for i, title in enumerate(samples, 1):
        print(f"    {i}. {title[:80]}...")

    print("\n── JOIN HEALTH CHECK ───────────────────────────────────────")
    pairs_product_ids = set(pairs_df["product_id"].unique())
    catalogue_product_ids = set(catalogue_df["product_id"].unique())
    in_pairs_not_catalogue = pairs_product_ids - catalogue_product_ids
    print(f"  Products in pairs but not catalogue: {len(in_pairs_not_catalogue):,}")
    if in_pairs_not_catalogue:
        print("  ⚠ These pairs will be dropped during processing.")
    else:
        print("  ✓ All pair products found in catalogue.")

    print("\n" + "="*60)


if __name__ == "__main__":
    cfg = load_config()
    download_esci(cfg)
    explore(cfg)