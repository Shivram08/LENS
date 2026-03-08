# src/data/download_images.py
#
# Parallel image downloader for SQID product images.
# Features:
#   - Concurrent downloads via ThreadPoolExecutor
#   - Resume capability (skips already-downloaded files)
#   - Retry logic for transient failures
#   - Manifest tracking (product_id → status)
#   - Progress bar via tqdm

import csv
import sys
import time
import random
import hashlib
import requests
import pandas as pd
from io import BytesIO
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ── constants ──────────────────────────────────────────────────────────────────
MAX_WORKERS   = 32      # concurrent download threads
TIMEOUT       = 10      # seconds per request
MAX_RETRIES   = 3       # attempts before marking as failed
RETRY_DELAY   = 2.0     # base seconds between retries (exponential backoff)
TARGET_SIZE   = 224     # resize images to 224×224 for DINOv2

# Amazon default "no image" URL — skip these
DEFAULT_VIDEO_URL = (
    "https://m.media-amazon.com/images/G/01/digital/video/"
    "web/Default_Background_Art_LTR._SX1080_FMjpg_.jpg"
)


def load_config(config_path: str = "configs/config.yaml"):
    return OmegaConf.load(config_path)


def download_and_save(
    product_id: str,
    url: str,
    out_dir: Path,
) -> dict:
    """
    Downloads a single product image, validates it, resizes to 224×224,
    and saves as JPEG. Returns a status dict for the manifest.

    Returns:
        {
            "product_id": str,
            "status": "ok" | "skipped" | "no_url" | "default_url" | "failed",
            "path": str | None,
        }
    """
    out_path = out_dir / f"{product_id}.jpg"

    # ── already downloaded ─────────────────────────────────────────────────
    if out_path.exists():
        return {"product_id": product_id, "status": "skipped", "path": str(out_path)}

    # ── no URL ────────────────────────────────────────────────────────────
    if not isinstance(url, str) or pd.isna(url) or url.strip() == "":
        return {"product_id": product_id, "status": "no_url", "path": None}

    # ── default video placeholder ──────────────────────────────────────────
    if url.strip() == DEFAULT_VIDEO_URL:
        return {"product_id": product_id, "status": "default_url", "path": None}

    # ── attempt download with retries ──────────────────────────────────────
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            headers = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}
            resp = requests.get(url.strip(), timeout=TIMEOUT, headers=headers)
            resp.raise_for_status()

            # validate it's a real image
            img = Image.open(BytesIO(resp.content)).convert("RGB")

            # resize to 224×224 using high-quality Lanczos resampling
            img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)

            # save as JPEG (quality=90 balances size vs fidelity)
            img.save(out_path, format="JPEG", quality=90)

            return {
                "product_id": product_id,
                "status": "ok",
                "path": str(out_path),
            }

        except Exception as e:
            if attempt < MAX_RETRIES:
                # exponential backoff with jitter
                sleep_time = RETRY_DELAY * (2 ** (attempt - 1)) + random.uniform(0, 1)
                time.sleep(sleep_time)
            else:
                return {
                    "product_id": product_id,
                    "status": "failed",
                    "path": None,
                }

    # should never reach here
    return {"product_id": product_id, "status": "failed", "path": None}


def download_all(cfg) -> None:
    """
    Downloads all product images in parallel and writes a manifest CSV.

    Manifest saved to: data/raw/image_manifest.csv
    Images saved to:   data/images/{product_id}.jpg
    """
    raw_dir = ROOT / cfg.paths.data_raw
    img_dir = ROOT / cfg.paths.images
    img_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = raw_dir / "image_manifest.csv"

    # ── load URL file ──────────────────────────────────────────────────────
    urls_df = pd.read_csv(raw_dir / "product_image_urls.csv")
    print(f"\nTotal products with URL entries : {len(urls_df):,}")
    print(f"  - Has URL                      : {urls_df['image_url'].notna().sum():,}")
    print(f"  - Missing URL (NaN)            : {urls_df['image_url'].isna().sum():,}")

    # ── load existing manifest to enable resume ────────────────────────────
    already_done = set()
    if manifest_path.exists():
        manifest_df = pd.read_csv(manifest_path)
        already_done = set(manifest_df["product_id"].astype(str).tolist())
        print(f"\nResume mode: {len(already_done):,} products already processed.")

    # filter to only unprocessed rows
    todo_df = urls_df[~urls_df["product_id"].astype(str).isin(already_done)]
    print(f"Remaining to process           : {len(todo_df):,}")

    if len(todo_df) == 0:
        print("\nAll images already downloaded.")
        summarize_manifest(manifest_path)
        return

    # ── parallel download ──────────────────────────────────────────────────
    print(f"\nStarting download with {MAX_WORKERS} parallel workers...")
    print("Images saved to:", img_dir)

    results = []
    manifest_file = open(manifest_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        manifest_file, fieldnames=["product_id", "status", "path"]
    )

    # write header only if file is new
    if len(already_done) == 0:
        writer.writeheader()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                download_and_save,
                str(row["product_id"]),
                row["image_url"],
                img_dir,
            ): str(row["product_id"])
            for _, row in todo_df.iterrows()
        }

        with tqdm(total=len(futures), unit="img", dynamic_ncols=True) as pbar:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)

                # write to manifest immediately (crash-safe)
                writer.writerow(result)
                manifest_file.flush()

                # update progress bar with live stats
                ok      = sum(1 for r in results if r["status"] == "ok")
                skipped = sum(1 for r in results if r["status"] == "skipped")
                failed  = sum(1 for r in results if r["status"] == "failed")
                pbar.set_postfix(ok=ok, skip=skipped, fail=failed, refresh=False)
                pbar.update(1)

    manifest_file.close()
    summarize_manifest(manifest_path)


def summarize_manifest(manifest_path: Path) -> None:
    """Prints a clean summary of the download manifest."""
    df = pd.read_csv(manifest_path)

    print("\n" + "="*55)
    print("IMAGE DOWNLOAD SUMMARY")
    print("="*55)
    vc = df["status"].value_counts()
    total = len(df)
    for status, count in vc.items():
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {status:<12} {count:>7,}  ({pct:5.1f}%)  {bar}")
    print(f"  {'TOTAL':<12} {total:>7,}")

    usable = vc.get("ok", 0) + vc.get("skipped", 0)
    print(f"\n  Usable images (ok + skipped): {usable:,}")
    print(f"  Coverage: {usable/total*100:.1f}%")
    print("="*55)


if __name__ == "__main__":
    cfg = load_config()
    download_all(cfg)