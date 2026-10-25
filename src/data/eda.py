"""Exploratory data analysis (EDA) for CUB-200-2011 with captions (Task 1.1).

Each analysis is one "step". A step writes figures to <out>/figures, tables to
<out>/tables, and its key numbers to <out>/summary.json.

Usage
-----
    python -m src.data.eda --out outputs/eda                          # all steps
    python -m src.data.eda --out outputs/eda --steps inspect classes  # some steps
    python -m src.data.eda --out outputs/eda_test --sample 300        # quick test run

The per-image pass (used by the "images" and "quality" steps) is slow, so its
result is cached in <out>/tables/image_stats*.csv. Use --recompute to redo it.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures without needing a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

TRAIN_REPO = "Multimodal-Fatima/CUB_train"
TEST_REPO = "Multimodal-Fatima/CUB_test"


# --------------------------------------------------------------------------
# Section 0 - helpers and shared state
# --------------------------------------------------------------------------
@dataclass
class EDAContext:
    """Everything the steps share: the two splits, output folders, and the summary."""

    train: object
    test: object
    out: Path
    seed: int = 42
    sample: int | None = None
    recompute: bool = False
    near_dup_threshold: int = 4
    summary: dict = field(default_factory=dict)
    _metadata: pd.DataFrame | None = None
    _image_df: pd.DataFrame | None = None
    _captions: tuple | None = None

    def __post_init__(self):
        self.fig_dir = self.out / "figures"
        self.table_dir = self.out / "tables"
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.table_dir.mkdir(parents=True, exist_ok=True)
        summary_path = self.out / "summary.json"
        if summary_path.exists():  # keep results of steps run earlier
            self.summary = json.loads(summary_path.read_text())

    def split(self, name: str):
        return self.train if name == "train" else self.test

    def save_summary(self):
        (self.out / "summary.json").write_text(json.dumps(self.summary, indent=2, default=_to_builtin))

    @property
    def metadata(self) -> pd.DataFrame:
        if self._metadata is None:
            self._metadata = build_metadata(self.train, self.test)
        return self._metadata

    @property
    def image_df(self) -> pd.DataFrame:
        if self._image_df is None:
            self._image_df = load_or_compute_image_stats(self)
        return self._image_df

    @property
    def captions(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self._captions is None:
            self._captions = build_caption_frames(self)
        return self._captions


def _to_builtin(obj):
    """Make numpy numbers JSON-serialisable."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Cannot serialise {type(obj)}")


def save_fig(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def describe(series: pd.Series) -> dict:
    """Min / percentiles / max of a numeric column, rounded for the summary."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {}
    q = s.quantile([0.05, 0.5, 0.95])
    return {
        "min": round(float(s.min()), 3), "p5": round(float(q[0.05]), 3),
        "median": round(float(q[0.5]), 3), "mean": round(float(s.mean()), 3),
        "p95": round(float(q[0.95]), 3), "max": round(float(s.max()), 3),
    }


def normalize_name(text: str) -> str:
    """'001.Black_footed_Albatross' / 'Black footed Albatross' -> 'black footed albatross'."""
    text = re.sub(r"^\d+\.", "", str(text)).lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def species_from_file_name(file_name: str) -> str:
    """'Black_Footed_Albatross_0007_796138.jpg' -> 'black footed albatross'."""
    stem = Path(str(file_name)).stem
    return normalize_name(stem.rsplit("_", 2)[0])


def show_image_grid(ctx: EDAContext, items, path: Path, title: str, ncols: int = 4):
    """items: list of (split, idx, caption). Draws the images in a grid with captions."""
    items = list(items)[:16]
    if not items:
        return
    nrows = int(np.ceil(len(items) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.4 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (split, idx, caption) in zip(axes.flat, items):
        try:
            img = ctx.split(split)[int(idx)]["image"].convert("RGB")
            ax.imshow(img)
        except Exception as exc:  # corrupted file: show the error instead of crashing
            ax.text(0.5, 0.5, f"cannot decode\n{type(exc).__name__}", ha="center", va="center")
        ax.set_title(textwrap.fill(str(caption), 34), fontsize=8)
    fig.suptitle(title)
    save_fig(fig, path)


# --------------------------------------------------------------------------
# Section 1 - loading and inspection
# --------------------------------------------------------------------------
def load_splits(sample: int | None = None, seed: int = 42, cache_dir: str | None = None):
    from datasets import load_dataset

    train = load_dataset(TRAIN_REPO, split="train", cache_dir=cache_dir)
    test = load_dataset(TEST_REPO, split="test", cache_dir=cache_dir)
    if sample:  # small random subset for quick test runs
        train = train.shuffle(seed=seed).select(range(min(sample, len(train))))
        test = test.shuffle(seed=seed).select(range(min(sample, len(test))))
    return train, test


def label_names(ds) -> list[str] | None:
    feature = ds.features.get("label")
    return list(feature.names) if hasattr(feature, "names") else None


def build_metadata(train, test) -> pd.DataFrame:
    """One row per image with everything except the pixels (fast: no image decoding)."""
    frames = []
    for split, ds in (("train", train), ("test", test)):
        cols = [c for c in ("id", "label", "file_name", "description") if c in ds.column_names]
        df = ds.select_columns(cols).to_pandas()
        df.insert(0, "idx", range(len(df)))
        df.insert(0, "split", split)
        names = label_names(ds)
        if names is not None:
            df["label_name"] = [names[int(v)] if pd.notna(v) else None for v in df["label"]]
        else:
            df["label_name"] = df["label"].astype(str)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def step_inspect(ctx: EDAContext):
    """Schema, row counts, id uniqueness, label / file-name consistency, sample grid."""
    meta = ctx.metadata
    info = {}
    for split in ("train", "test"):
        ds = ctx.split(split)
        info[split] = {
            "num_rows": len(ds),
            "columns": {name: str(feat) for name, feat in ds.features.items()},
        }
        print(f"  {split}: {len(ds)} rows, columns = {ds.column_names}")

    if "id" in meta:
        info["duplicate_ids_within_split"] = int(meta.duplicated(["split", "id"]).sum())
        info["ids_shared_by_train_and_test"] = int(
            len(set(meta.loc[meta.split == "train", "id"]) & set(meta.loc[meta.split == "test", "id"]))
        )
        # If id is just the row number, shared ids say nothing about leakage; file_name is the real key
        info["id_is_row_index"] = {split: bool((g["id"] == g["idx"]).all()) for split, g in meta.groupby("split")}

    # Does the species in the file name agree with the label?
    if "file_name" in meta:
        meta = meta.assign(
            file_species=meta["file_name"].map(species_from_file_name),
            label_norm=meta["label_name"].map(normalize_name),
        )
        mismatch = meta[meta.file_species != meta.label_norm]
        mismatch[["split", "idx", "file_name", "label_name"]].to_csv(
            ctx.table_dir / "label_filename_mismatch.csv", index=False
        )
        info["label_filename_mismatches"] = len(mismatch)
        info["file_names_shared_by_train_and_test"] = int(
            len(set(meta.loc[meta.split == "train", "file_name"])
                & set(meta.loc[meta.split == "test", "file_name"]))
        )

    ctx.summary["inspect"] = info

    rng = random.Random(ctx.seed)
    train_meta = ctx.metadata[ctx.metadata.split == "train"]
    picks = rng.sample(range(len(train_meta)), k=min(12, len(train_meta)))
    items = []
    for i in picks:
        row = train_meta.iloc[i]
        first_caption = str(row.get("description", "")).strip().split("\n")[0]
        items.append(("train", row.idx, f"{row.label_name}\n\"{first_caption}\""))
    show_image_grid(ctx, items, ctx.fig_dir / "01_sample_grid.png", "Random training samples")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
STEPS = {
    "inspect": step_inspect,
}


def main(argv=None):
    parser = argparse.ArgumentParser(description="EDA for CUB-200-2011 with captions")
    parser.add_argument("--out", default="outputs/eda", help="output folder")
    parser.add_argument("--steps", nargs="+", choices=list(STEPS), default=list(STEPS))
    parser.add_argument("--sample", type=int, default=None, help="use N random images per split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--near-dup-threshold", type=int, default=4, help="max differing dHash bits")
    parser.add_argument("--recompute", action="store_true", help="ignore the cached image stats")
    parser.add_argument("--cache-dir", default=None, help="Hugging Face cache folder")
    args = parser.parse_args(argv)

    train, test = load_splits(args.sample, args.seed, args.cache_dir)
    ctx = EDAContext(train=train, test=test, out=Path(args.out), seed=args.seed, sample=args.sample,
                     recompute=args.recompute, near_dup_threshold=args.near_dup_threshold)
    for name in args.steps:
        print(f"[eda] step: {name}")
        STEPS[name](ctx)
        ctx.save_summary()
    print(f"[eda] done. Summary: {ctx.out / 'summary.json'}")


if __name__ == "__main__":
    main()
