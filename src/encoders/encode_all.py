# src/encoders/encode_all.py
#
# Offline feature extraction pipeline.
# Encodes all products (image + text) and all queries (text).
# Builds FAISS indices over product embeddings.
# Logs all metadata to MLflow.
#
# Outputs:
#   embeddings/product_image_embs.pt   (N_products, 384)
#   embeddings/product_text_embs.pt    (N_products, 384)
#   embeddings/query_embs.pt           (N_queries,  384)
#   embeddings/product_ids.json        index → product_id
#   embeddings/query_ids.json          index → query_id
#   embeddings/faiss_image.index       FAISS IndexFlatIP
#   embeddings/faiss_text.index        FAISS IndexFlatIP

import sys
import json
import mlflow
import torch
import faiss
import numpy as np
import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.encoders.image_encoder import ImageEncoder
from src.encoders.text_encoder  import TextEncoder


def load_config(config_path: str = "configs/config.yaml"):
    return OmegaConf.load(config_path)


def get_device() -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    if device == "cuda":
        print(f"  GPU   : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM  : {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    return device


def encode_product_images(
    products_df: pd.DataFrame,
    img_dir: Path,
    emb_dir: Path,
    cfg,
    device: str,
) -> torch.Tensor:
    """
    Encodes all product images with DINOv2.
    Skips if embeddings already exist on disk.

    Returns:
        image_embs: tensor of shape (N, 384), L2-normalized
    """
    out_path = emb_dir / "product_image_embs.pt"
    if out_path.exists():
        print("  product_image_embs.pt already exists — loading from disk.")
        return torch.load(out_path, weights_only=True)

    encoder = ImageEncoder(
        model_name    = cfg.encoders.image_model,
        embedding_dim = cfg.encoders.embedding_dim,
        device        = device,
    )

    # build ordered list of image paths
    # products without images get a zero tensor (handled inside encode_paths)
    image_paths = [
        img_dir / f"{pid}.jpg"
        for pid in products_df["product_id"].astype(str)
    ]

    print(f"  Encoding {len(image_paths):,} product images...")
    embs = encoder.encode_paths(
        image_paths,
        batch_size    = cfg.encoders.image_batch_size,
        show_progress = True,
    )                                                    # (N, 384)

    torch.save(embs, out_path)
    print(f"  Saved → {out_path}  shape={tuple(embs.shape)}")

    # free GPU memory before text encoding
    del encoder
    torch.cuda.empty_cache()

    return embs


def encode_product_texts(
    products_df: pd.DataFrame,
    emb_dir: Path,
    cfg,
    device: str,
) -> torch.Tensor:
    """
    Encodes all product text fields with MiniLM.
    Skips if embeddings already exist on disk.

    Returns:
        text_embs: tensor of shape (N, 384), L2-normalized
    """
    out_path = emb_dir / "product_text_embs.pt"
    if out_path.exists():
        print("  product_text_embs.pt already exists — loading from disk.")
        return torch.load(out_path, weights_only=True)

    encoder = TextEncoder(
        model_name    = cfg.encoders.text_model,
        embedding_dim = cfg.encoders.embedding_dim,
        device        = device,
    )

    texts = products_df["product_text"].fillna("").tolist()

    print(f"  Encoding {len(texts):,} product texts...")
    embs = encoder.encode_texts(
        texts,
        batch_size    = cfg.encoders.text_batch_size,
        show_progress = True,
    )                                                    # (N, 384)

    torch.save(embs, out_path)
    print(f"  Saved → {out_path}  shape={tuple(embs.shape)}")

    del encoder
    torch.cuda.empty_cache()

    return embs


def encode_queries(
    queries_df: pd.DataFrame,
    emb_dir: Path,
    cfg,
    device: str,
) -> torch.Tensor:
    """
    Encodes all queries with MiniLM.
    Skips if embeddings already exist on disk.

    Returns:
        query_embs: tensor of shape (Q, 384), L2-normalized
    """
    out_path = emb_dir / "query_embs.pt"
    if out_path.exists():
        print("  query_embs.pt already exists — loading from disk.")
        return torch.load(out_path, weights_only=True)

    encoder = TextEncoder(
        model_name    = cfg.encoders.text_model,
        embedding_dim = cfg.encoders.embedding_dim,
        device        = device,
    )

    texts = queries_df["query"].fillna("").tolist()

    print(f"  Encoding {len(texts):,} queries...")
    embs = encoder.encode_texts(
        texts,
        batch_size    = cfg.encoders.text_batch_size,
        show_progress = True,
    )                                                    # (Q, 384)

    torch.save(embs, out_path)
    print(f"  Saved → {out_path}  shape={tuple(embs.shape)}")

    del encoder
    torch.cuda.empty_cache()

    return embs


def save_id_mappings(
    products_df: pd.DataFrame,
    queries_df: pd.DataFrame,
    emb_dir: Path,
) -> None:
    """
    Saves index → ID mappings as JSON.

    These are the critical registry files that map FAISS integer
    indices back to product_ids and query_ids.
    Must never be modified after FAISS indices are built.
    """
    product_ids = products_df["product_id"].astype(str).tolist()
    query_ids   = queries_df["query_id"].astype(str).tolist()

    with open(emb_dir / "product_ids.json", "w") as f:
        json.dump(product_ids, f)

    with open(emb_dir / "query_ids.json", "w") as f:
        json.dump(query_ids, f)

    print(f"  Saved product_ids.json ({len(product_ids):,} entries)")
    print(f"  Saved query_ids.json   ({len(query_ids):,} entries)")


def build_faiss_indices(
    image_embs: torch.Tensor,
    text_embs: torch.Tensor,
    emb_dir: Path,
    cfg,
) -> None:
    """
    Builds and saves FAISS IndexFlatIP indices over product embeddings.

    IndexFlatIP performs exact inner product search.
    Since embeddings are L2-normalized, inner product = cosine similarity.

    Args:
        image_embs : (N, 384) normalized image embeddings
        text_embs  : (N, 384) normalized text embeddings
        emb_dir    : output directory
        cfg        : config
    """
    d = cfg.encoders.embedding_dim                       # 384

    for name, embs in [("image", image_embs), ("text", text_embs)]:
        out_path = emb_dir / f"faiss_{name}.index"
        if out_path.exists():
            print(f"  faiss_{name}.index already exists — skipping.")
            continue

        # convert to float32 numpy (FAISS requirement)
        vectors = embs.numpy().astype(np.float32)        # (N, 384)

        # verify normalization — all norms should be ~1.0
        norms = np.linalg.norm(vectors, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5), \
            f"Embeddings not normalized! Mean norm={norms.mean():.4f}"

        # build index
        index = faiss.IndexFlatIP(d)                     # exact inner product
        index.add(vectors)                               # add all N vectors

        faiss.write_index(index, str(out_path))
        print(f"  Built faiss_{name}.index — {index.ntotal:,} vectors, d={d}")


def verify_embeddings(
    image_embs: torch.Tensor,
    text_embs: torch.Tensor,
    query_embs: torch.Tensor,
    products_df: pd.DataFrame,
    queries_df: pd.DataFrame,
) -> None:
    """
    Runs critical checks on all embeddings before saving.
    Raises AssertionError immediately if anything is wrong.
    """
    n_products = len(products_df)
    n_queries  = len(queries_df)

    # shape checks
    assert image_embs.shape == (n_products, 384), \
        f"image_embs shape mismatch: {image_embs.shape} vs ({n_products}, 384)"
    assert text_embs.shape  == (n_products, 384), \
        f"text_embs shape mismatch: {text_embs.shape} vs ({n_products}, 384)"
    assert query_embs.shape == (n_queries,  384), \
        f"query_embs shape mismatch: {query_embs.shape} vs ({n_queries}, 384)"

    # normalization checks — all norms should be 1.0 ± 1e-5
    for name, embs in [
        ("image_embs", image_embs),
        ("text_embs",  text_embs),
        ("query_embs", query_embs),
    ]:
        norms = embs.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
            f"{name} not normalized — mean norm={norms.mean():.4f}"

    # NaN / Inf check
    for name, embs in [
        ("image_embs", image_embs),
        ("text_embs",  text_embs),
        ("query_embs", query_embs),
    ]:
        assert not torch.isnan(embs).any(), f"{name} contains NaN"
        assert not torch.isinf(embs).any(), f"{name} contains Inf"

    print("  ✓ All embedding checks passed.")


def log_to_mlflow(
    image_embs: torch.Tensor,
    text_embs: torch.Tensor,
    query_embs: torch.Tensor,
    cfg,
) -> None:
    """Logs embedding metadata to MLflow."""
    mlflow.set_tracking_uri((ROOT / cfg.paths.logs).as_uri())
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run(run_name="encode_all"):
        mlflow.log_param("image_model",     cfg.encoders.image_model)
        mlflow.log_param("text_model",      cfg.encoders.text_model)
        mlflow.log_param("embedding_dim",   cfg.encoders.embedding_dim)
        mlflow.log_param("n_products",      image_embs.shape[0])
        mlflow.log_param("n_queries",       query_embs.shape[0])
        mlflow.log_param("image_batch_size",cfg.encoders.image_batch_size)
        mlflow.log_param("text_batch_size", cfg.encoders.text_batch_size)

        # log mean norm as sanity metric
        mlflow.log_metric("image_emb_mean_norm",
                          image_embs.norm(dim=1).mean().item())
        mlflow.log_metric("text_emb_mean_norm",
                          text_embs.norm(dim=1).mean().item())
        mlflow.log_metric("query_emb_mean_norm",
                          query_embs.norm(dim=1).mean().item())

    print("  ✓ Encoding metadata logged to MLflow.")


if __name__ == "__main__":
    cfg = load_config()

    print()
    print("="*55)
    print("LENS — PHASE 2: ENCODE ALL")
    print("="*55)

    device  = get_device()
    emb_dir = ROOT / cfg.paths.embeddings
    emb_dir.mkdir(parents=True, exist_ok=True)

    # ── load processed data ────────────────────────────────────────────────
    proc_dir    = ROOT / cfg.paths.data_processed
    products_df = pd.read_csv(proc_dir / "products.csv")
    queries_df  = pd.read_csv(proc_dir / "queries.csv")

    print(f"\n  Products to encode : {len(products_df):,}")
    print(f"  Queries to encode  : {len(queries_df):,}")

    # ── save ID mappings first (frozen registry) ───────────────────────────
    print("\n[ 1 ] ID MAPPINGS")
    save_id_mappings(products_df, queries_df, emb_dir)

    # ── encode product images ──────────────────────────────────────────────
    print("\n[ 2 ] PRODUCT IMAGE EMBEDDINGS  (DINOv2)")
    img_dir     = ROOT / cfg.paths.images
    image_embs  = encode_product_images(
        products_df, img_dir, emb_dir, cfg, device
    )

    # ── encode product texts ───────────────────────────────────────────────
    print("\n[ 3 ] PRODUCT TEXT EMBEDDINGS  (MiniLM)")
    text_embs   = encode_product_texts(
        products_df, emb_dir, cfg, device
    )

    # ── encode queries ─────────────────────────────────────────────────────
    print("\n[ 4 ] QUERY EMBEDDINGS  (MiniLM)")
    query_embs  = encode_queries(
        queries_df, emb_dir, cfg, device
    )

    # ── verify all embeddings ──────────────────────────────────────────────
    print("\n[ 5 ] EMBEDDING VERIFICATION")
    verify_embeddings(image_embs, text_embs, query_embs, products_df, queries_df)

    # ── build FAISS indices ────────────────────────────────────────────────
    print("\n[ 6 ] FAISS INDICES")
    build_faiss_indices(image_embs, text_embs, emb_dir, cfg)

    # ── log to MLflow ──────────────────────────────────────────────────────
    print("\n[ 7 ] MLFLOW LOGGING")
    log_to_mlflow(image_embs, text_embs, query_embs, cfg)

    print()
    print("="*55)
    print("  Phase 2 complete. Ready for Phase 3 (Fusion Bridge).")
    print("="*55)