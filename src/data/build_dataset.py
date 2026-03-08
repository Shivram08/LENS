# src/data/build_dataset.py
#
# Builds stratified train/val/test splits from raw ESCI pairs.
#
# Design decisions:
#   - Split at QUERY level (not pair level) → zero query leakage
#   - Stratified by esci_label → balanced label distribution in all splits
#   - Only keeps products that have a valid downloaded image
#   - Produces queries.csv and products.csv as lookup tables
#   - Logs all dataset statistics to MLflow

import sys
import mlflow
import pandas as pd
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def load_config(config_path: str = "configs/config.yaml"):
    return OmegaConf.load(config_path)


def load_raw_data(cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads raw pairs and builds products table.
    Joins image manifest to filter products without valid images.

    Returns:
        pairs_df    : (query_id, query, product_id, esci_label)
        products_df : (product_id, product_title, product_bullet_point,
                       product_brand, product_color, product_description,
                       has_image)
    """
    raw_dir = ROOT / cfg.paths.data_raw

    # ── load pairs ─────────────────────────────────────────────────────────
    pairs_df = pd.read_csv(raw_dir / "pairs_raw.csv")
    print(f"Raw pairs loaded       : {len(pairs_df):,}")

    # ── load image manifest ────────────────────────────────────────────────
    manifest_df = pd.read_csv(raw_dir / "image_manifest.csv")
    valid_image_ids = set(
        manifest_df.loc[manifest_df["status"] == "ok", "product_id"].astype(str)
    )
    print(f"Products with images   : {len(valid_image_ids):,}")

    # ── build products table from pairs data ───────────────────────────────
    # Re-download products from HuggingFace if not already saved
    products_path = raw_dir / "products_raw.csv"
    if not products_path.exists():
        print("products_raw.csv not found — re-downloading from HuggingFace...")
        from datasets import load_dataset
        train_ds = load_dataset("tasksource/esci", split="train", trust_remote_code=True)
        test_ds  = load_dataset("tasksource/esci", split="test",  trust_remote_code=True)
        full_df  = pd.concat([
            train_ds.to_pandas(),
            test_ds.to_pandas()
        ], ignore_index=True)
        full_df  = full_df[full_df["product_locale"] == cfg.dataset.locale]
        product_cols = [
            "product_id", "product_title", "product_bullet_point",
            "product_brand", "product_color", "product_description",
        ]
        products_df = (
            full_df[product_cols]
            .drop_duplicates(subset="product_id")
            .reset_index(drop=True)
        )
        products_df.to_csv(products_path, index=False)
    else:
        products_df = pd.read_csv(products_path)

    # ── mark which products have valid images ──────────────────────────────
    products_df["product_id"] = products_df["product_id"].astype(str)
    products_df["has_image"]  = products_df["product_id"].isin(valid_image_ids)
    print(f"Total unique products  : {len(products_df):,}")
    print(f"  with valid image     : {products_df['has_image'].sum():,}")
    print(f"  without image        : {(~products_df['has_image']).sum():,}")

    return pairs_df, products_df


def filter_pairs(
    pairs_df: pd.DataFrame,
    products_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Filters pairs to only those whose product has a valid image.
    This is critical — we cannot train the image encoder on missing images.

    Args:
        pairs_df    : raw query-product pairs
        products_df : products with has_image flag

    Returns:
        filtered pairs_df
    """
    valid_ids  = set(products_df.loc[products_df["has_image"], "product_id"].astype(str))
    pairs_df   = pairs_df.copy()
    pairs_df["product_id"] = pairs_df["product_id"].astype(str)

    before = len(pairs_df)
    pairs_df = pairs_df[pairs_df["product_id"].isin(valid_ids)].reset_index(drop=True)
    after  = len(pairs_df)

    print(f"\nPairs after image filter : {after:,}  (dropped {before-after:,})")
    return pairs_df


def split_at_query_level(
    pairs_df: pd.DataFrame,
    cfg,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits data at the QUERY level to prevent leakage.

    Strategy:
      1. Get all unique query_ids
      2. For each query_id, compute its dominant esci_label
         (the label that appears most for that query's products)
      3. Use that dominant label to stratify the query-level split
      4. Assign all pairs belonging to a query_id to exactly one split

    This guarantees:
      - Zero query overlap between train / val / test
      - Approximately balanced label distributions across splits
      - Every query's full set of products stays together

    Args:
        pairs_df : filtered pairs (query_id, query, product_id, esci_label)
        cfg      : OmegaConf config
        seed     : random seed for reproducibility

    Returns:
        train_df, val_df, test_df
    """
    np.random.seed(seed)

    # ── step 1: get unique queries with dominant label ─────────────────────
    # dominant label = most frequent esci_label for that query's products
    # this gives us a single stratification key per query
    dominant = (
        pairs_df.groupby("query_id")["esci_label"]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
        .rename(columns={"esci_label": "dominant_label"})
    )

    print(f"\nUnique queries         : {len(dominant):,}")
    print("Dominant label distribution:")
    vc = dominant["dominant_label"].value_counts()
    for label, count in vc.items():
        pct = count / len(dominant) * 100
        print(f"  {label:<12} {count:>7,}  ({pct:.1f}%)")

    # ── step 2: split query_ids (80 / 10 / 10) ────────────────────────────
    # first split: 80% train, 20% temp
    train_qids, temp_qids = train_test_split(
        dominant["query_id"].values,
        test_size=0.20,
        stratify=dominant["dominant_label"].values,
        random_state=seed,
    )

    # second split: split temp 50/50 → 10% val, 10% test
    temp_dominant = dominant[dominant["query_id"].isin(temp_qids)]
    val_qids, test_qids = train_test_split(
        temp_dominant["query_id"].values,
        test_size=0.50,
        stratify=temp_dominant["dominant_label"].values,
        random_state=seed,
    )

    print(f"\nQuery split:")
    print(f"  train queries : {len(train_qids):,}")
    print(f"  val queries   : {len(val_qids):,}")
    print(f"  test queries  : {len(test_qids):,}")

    # ── step 3: assign pairs to splits ────────────────────────────────────
    train_set = set(train_qids)
    val_set   = set(val_qids)
    test_set  = set(test_qids)

    train_df = pairs_df[pairs_df["query_id"].isin(train_set)].reset_index(drop=True)
    val_df   = pairs_df[pairs_df["query_id"].isin(val_set)].reset_index(drop=True)
    test_df  = pairs_df[pairs_df["query_id"].isin(test_set)].reset_index(drop=True)

    return train_df, val_df, test_df


def sample_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Downsamples splits to target sizes defined in config.
    Uses stratified sampling to preserve label distribution.

    Args:
        train_df, val_df, test_df : full splits
        cfg                        : config with dataset.n_train/val/test

    Returns:
        sampled train_df, val_df, test_df
    """
    def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
        if len(df) <= n:
            return df
        # sample n rows, preserving esci_label proportions
        return (
            df.groupby("esci_label", group_keys=False)
            .apply(lambda x: x.sample(
                frac=n / len(df),
                random_state=seed,
            ))
            .reset_index(drop=True)
            .sample(frac=1, random_state=seed)   # shuffle
            .reset_index(drop=True)
        )

    train_sampled = stratified_sample(train_df, cfg.dataset.n_train, seed)
    val_sampled   = stratified_sample(val_df,   cfg.dataset.n_val,   seed)
    test_sampled  = stratified_sample(test_df,  cfg.dataset.n_test,  seed)

    return train_sampled, val_sampled, test_sampled


def verify_no_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """
    Asserts zero query overlap between splits.
    Raises AssertionError immediately if leakage is detected.
    """
    train_qids = set(train_df["query_id"])
    val_qids   = set(val_df["query_id"])
    test_qids  = set(test_df["query_id"])

    tv_overlap  = train_qids & val_qids
    tt_overlap  = train_qids & test_qids
    vt_overlap  = val_qids   & test_qids

    assert len(tv_overlap) == 0,  f"LEAKAGE: {len(tv_overlap)} queries in train ∩ val"
    assert len(tt_overlap) == 0,  f"LEAKAGE: {len(tt_overlap)} queries in train ∩ test"
    assert len(vt_overlap) == 0,  f"LEAKAGE: {len(vt_overlap)} queries in val ∩ test"

    print("\n✓ Zero query leakage confirmed across all splits.")


def print_split_summary(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Prints a detailed summary table of all three splits."""
    print("\n" + "="*65)
    print("SPLIT SUMMARY")
    print("="*65)

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"\n  {name.upper()}")
        print(f"    pairs          : {len(df):,}")
        print(f"    unique queries : {df['query_id'].nunique():,}")
        print(f"    unique products: {df['product_id'].nunique():,}")
        print(f"    label distribution:")
        vc = df["esci_label"].value_counts()
        for label, count in vc.items():
            pct = count / len(df) * 100
            bar = "█" * int(pct / 2)
            print(f"      {label:<12} {count:>7,}  ({pct:5.1f}%)  {bar}")

    print("\n" + "="*65)


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    products_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
    cfg,
) -> None:
    """
    Saves all processed files to data/processed/.

    Files produced:
        train_pairs.csv   ← (query_id, query, product_id, esci_label)
        val_pairs.csv
        test_pairs.csv
        queries.csv       ← (query_id, query) — all unique queries in train
        products.csv      ← (product_id, product_text, has_image)
                             product_text = title + " . " + bullets
    """
    proc_dir = ROOT / cfg.paths.data_processed
    proc_dir.mkdir(parents=True, exist_ok=True)

    # ── save pair splits ───────────────────────────────────────────────────
    train_df.to_csv(proc_dir / "train_pairs.csv", index=False)
    val_df.to_csv(  proc_dir / "val_pairs.csv",   index=False)
    test_df.to_csv( proc_dir / "test_pairs.csv",  index=False)

    # ── build queries lookup (train queries only) ──────────────────────────
    queries_df = (
        train_df[["query_id", "query"]]
        .drop_duplicates(subset="query_id")
        .reset_index(drop=True)
    )
    queries_df.to_csv(proc_dir / "queries.csv", index=False)

    # ── build products lookup with concatenated text field ─────────────────
    # product_text = title + " . " + bullet_points
    # This is the text field MiniLM will encode.
    # Fallback: title only if bullet_point is missing.
    products_out = products_df.copy()
    products_out["product_title"]        = products_out["product_title"].fillna("")
    products_out["product_bullet_point"] = products_out["product_bullet_point"].fillna("")

    products_out["product_text"] = products_out.apply(
        lambda r: (
            r["product_title"].strip() + " . " + r["product_bullet_point"].strip()
            if r["product_bullet_point"].strip()
            else r["product_title"].strip()
        ),
        axis=1,
    )

    # keep only products referenced in any split
    all_product_ids = set(
        pd.concat([train_df, val_df, test_df])["product_id"].astype(str)
    )
    products_out = products_out[
        products_out["product_id"].astype(str).isin(all_product_ids)
    ].reset_index(drop=True)

    products_out[[
        "product_id", "product_text", "product_title",
        "product_bullet_point", "has_image",
    ]].to_csv(proc_dir / "products.csv", index=False)

    print(f"\nSaved to {proc_dir}/")
    print(f"  train_pairs.csv : {len(train_df):,} rows")
    print(f"  val_pairs.csv   : {len(val_df):,} rows")
    print(f"  test_pairs.csv  : {len(test_df):,} rows")
    print(f"  queries.csv     : {len(queries_df):,} rows")
    print(f"  products.csv    : {len(products_out):,} rows")


def log_to_mlflow(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg,
) -> None:
    """
    Logs dataset statistics to MLflow so every experiment
    has a traceable record of exactly what data was used.
    """
    mlflow.set_tracking_uri((ROOT / cfg.paths.logs).as_uri())
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run(run_name="build_dataset"):
        # log config params
        mlflow.log_param("locale",  cfg.dataset.locale)
        mlflow.log_param("n_train", len(train_df))
        mlflow.log_param("n_val",   len(val_df))
        mlflow.log_param("n_test",  len(test_df))
        mlflow.log_param("seed",    42)

        # log label distributions as metrics
        for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            vc = df["esci_label"].value_counts(normalize=True)
            for label, frac in vc.items():
                mlflow.log_metric(f"{split_name}_{label}_frac", round(frac, 4))

        mlflow.log_param("train_unique_queries",  train_df["query_id"].nunique())
        mlflow.log_param("val_unique_queries",    val_df["query_id"].nunique())
        mlflow.log_param("test_unique_queries",   test_df["query_id"].nunique())

        print("\n✓ Dataset statistics logged to MLflow.")


if __name__ == "__main__":
    cfg = load_config()

    print("="*55)
    print("LENS — BUILD DATASET")
    print("="*55)

    # 1. load raw data
    pairs_df, products_df = load_raw_data(cfg)

    # 2. filter to image-available products only
    pairs_df = filter_pairs(pairs_df, products_df)

    # 3. split at query level
    train_df, val_df, test_df = split_at_query_level(pairs_df, cfg)

    # 4. downsample to target sizes
    train_df, val_df, test_df = sample_splits(train_df, val_df, test_df, cfg)

    # 5. verify zero leakage
    verify_no_leakage(train_df, val_df, test_df)

    # 6. print summary
    print_split_summary(train_df, val_df, test_df)

    # 7. save all files
    save_splits(train_df, val_df, test_df, products_df, pairs_df, cfg)

    # 8. log to MLflow
    log_to_mlflow(train_df, val_df, test_df, cfg)

    print("\n✓ Phase 1 complete. Ready for Phase 2 (Encoders & Embeddings).")