"""
MarkupLM — SWDE Finetuning Experiment (4 Verticals)
=====================================================
Reproduces the SWDE benchmark from:
  Li et al. (2022) "MarkupLM: Pre-Training of Text and Markup Language
  for Visually-Rich Document Understanding", ACL 2022.

Protocol (from paper):
  - 10-fold leave-n-sites-out cross-validation
  - n_seed=1 training website per fold
  - 2000 pages per website
  - 4 previous context nodes
  - Metric: page-level precision / recall / F1

Usage
-----
# Step 1 (once): pack + prepare SWDE data
python markuplm_finetune_swde.py \\
    --swde_path      "MarkupLM Finetuning/SWDE" \\
    --processed_path "MarkupLM Finetuning/SWDE/processed" \\
    --output_dir     "MarkupLM Finetuning/results_markuplm" \\
    --cuda           0

# Step 2: run only training (data already prepared)
python markuplm_finetune_swde.py \\
    --swde_path      "MarkupLM Finetuning/SWDE" \\
    --processed_path "MarkupLM Finetuning/SWDE/processed" \\
    --output_dir     "MarkupLM Finetuning/results_markuplm" \\
    --skip_data_prep \\
    --verticals    auto book camera job movie nbaplayer restaurant university \\
"""

import argparse
import csv
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("swde_experiment.log"),
    ],
)
logger = logging.getLogger(__name__)

# ── Experiment configuration ─────────────────────────────────────────────────

SELECTED_VERTICALS = ["auto", "book", "camera", "job", "movie", "nbaplayer", "restaurant", "university"]

# Exact hyperparameters from the MarkupLM paper (Table 9 / Appendix)
HPARAMS = {
    "n_seed":                    1,  # paper uses 5 training website per fold, so effectively no seed variation, but we keep 1 for consistency with the run.py interface
    "n_pages":                   2000,
    "prev_nodes_into_account":   4,
    "per_gpu_train_batch_size":  32,
    "per_gpu_eval_batch_size":   32,
    "num_train_epochs":          10,
    "learning_rate":             "2e-5",
    "warmup_ratio":              0.1,
    "max_seq_length":            384,
    "doc_stride":                128,
    "save_steps":                1000000,   # effectively no intermediate save
}

# Path to the official MarkupLM run_swde scripts (relative to this file)
SCRIPTS_DIR = (
    Path(__file__).parent
    / "unilm_repo" / "markuplm" / "examples" / "fine_tuning" / "run_swde"
)
MARKUPLM_ROOT = SCRIPTS_DIR.parent.parent.parent.parent  # unilm_repo/markuplm/

# ─────────────────────────────────────────────────────────────────────────────


def _run(cmd: list, cwd: Path = None, env: dict = None):
    logger.info("CMD: %s", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True, cwd=str(cwd or SCRIPTS_DIR), env=env)


# ── Pipeline steps ────────────────────────────────────────────────────────────

def pack_data(swde_path: Path, pickle_path: Path, verticals: list, n_pages: int):
    """Pack first n_pages HTML pages per website into a single pickle."""
    if pickle_path.exists():
        logger.info("Pickle already exists, skipping: %s", pickle_path)
        return
    logger.info("=== STEP 1/3 — pack_data (verticals: %s, n_pages: %d) ===", verticals, n_pages)
    import pickle as pkl

    swde_data = []
    for v in verticals:
        v_path = swde_path / v
        if not v_path.exists():
            logger.warning("Vertical folder not found, skipping: %s", v_path)
            continue
        for w in sorted(os.listdir(v_path)):
            for filename in sorted(os.listdir(v_path / w))[:n_pages]:
                swde_data.append({
                    "vertical": v,
                    "website":  w,
                    "path":     str(Path(v) / w / filename),
                })

    logger.info("Reading %d HTML pages...", len(swde_data))
    for page in swde_data:
        with open(swde_path / page["path"]) as f:
            page["html_str"] = f.read()

    with open(pickle_path, "wb") as f:
        pkl.dump(swde_data, f)
    logger.info("Pickle saved → %s", pickle_path)


def prepare_data(groundtruth_path: Path, pickle_path: Path, processed_path: Path,
                 verticals: list, n_pages: int):
    """Extract XPaths and build per-website feature pickles (prepare_data.py)."""
    processed_path.mkdir(parents=True, exist_ok=True)
    existing = {f.stem.split("-")[0] for f in processed_path.glob(f"*-{n_pages}.pickle")}
    if all(v in existing for v in verticals):
        logger.info("Processed features found for all verticals, skipping prepare_data.")
        return
    logger.info("=== STEP 2/3 — prepare_data ===")
    _run([
        sys.executable, SCRIPTS_DIR / "prepare_data.py",
        f"--input_groundtruth_path={groundtruth_path}",
        f"--input_pickle_path={pickle_path}",
        f"--output_data_path={processed_path}",
        f"--n_pages={n_pages}",
        f"--verticals={','.join(verticals)}",
        f"--num_workers=1",
    ])


def train_eval(
    vertical: str,
    processed_path: Path,
    output_dir: Path,
    model_name_or_path: str,
    env: dict,
    n_seed: int = None,
):
    """Run 10-fold train+eval for one vertical (run.py)."""
    n_seed = n_seed if n_seed is not None else HPARAMS["n_seed"]
    score_file = output_dir / f"{vertical}-all-10-runs-score.txt"
    if score_file.exists():
        logger.info("SKIP vertical=%s n_seed=%d — score file already exists: %s", vertical, n_seed, score_file)
        return
    logger.info("=== STEP 3/3 — train+eval: vertical=%s, n_seed=%d ===", vertical, n_seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    _run([
        sys.executable, SCRIPTS_DIR / "run.py",
        "--root_dir",                    processed_path,
        "--vertical",                    vertical,
        "--n_seed",                      n_seed,
        "--n_pages",                     HPARAMS["n_pages"],
        "--prev_nodes_into_account",     HPARAMS["prev_nodes_into_account"],
        "--model_name_or_path",          model_name_or_path,
        "--output_dir",                  output_dir,
        "--do_train",
        "--do_eval",
        "--per_gpu_train_batch_size",    HPARAMS["per_gpu_train_batch_size"],
        "--per_gpu_eval_batch_size",     HPARAMS["per_gpu_eval_batch_size"],
        "--num_train_epochs",            HPARAMS["num_train_epochs"],
        "--learning_rate",               HPARAMS["learning_rate"],
        "--warmup_ratio",                HPARAMS["warmup_ratio"],
        "--max_seq_length",              HPARAMS["max_seq_length"],
        "--doc_stride",                  HPARAMS["doc_stride"],
        "--save_steps",                  HPARAMS["save_steps"],
        "--overwrite_output_dir",
    ], env=env)


# ── Results ───────────────────────────────────────────────────────────────────

def _parse_score_file(path: Path) -> dict:
    result = {}
    with open(path) as f:
        for line in f:
            if " : " in line:
                key, val = line.strip().split(" : ", 1)
                result[key.strip()] = float(val.strip())
    return result


def aggregate_and_report(output_dir: Path, verticals: list, k: int = None):
    rows = []
    for v in verticals:
        score_file = output_dir / f"{v}-all-10-runs-score.txt"
        if not score_file.exists():
            logger.warning("Missing score file for '%s': %s", v, score_file)
            continue
        r = _parse_score_file(score_file)
        rows.append({"vertical": v, **r})

    if not rows:
        logger.error("No result files found — check output_dir.")
        return

    n = len(rows)
    rows.append({
        "vertical":  "AVERAGE",
        "Precision": sum(r["Precision"] for r in rows) / n,
        "Recall":    sum(r["Recall"]    for r in rows) / n,
        "F1":        sum(r["F1"]        for r in rows) / n,
    })

    summary_csv = output_dir / "swde_results_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["vertical", "Precision", "Recall", "F1"])
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Summary CSV → %s", summary_csv)

    label = f"k={k}" if k is not None else "paper protocol"
    print("\n" + "=" * 54)
    print(f"  SWDE — MarkupLM-base  ({label})")
    print("=" * 54)
    print(f"  {'Vertical':<14} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("  " + "─" * 48)
    for row in rows:
        marker = "  " if row["vertical"] != "AVERAGE" else "► "
        print(
            f"{marker}{row['vertical']:<14}"
            f" {row['Precision']:>10.4f}"
            f" {row['Recall']:>10.4f}"
            f" {row['F1']:>10.4f}"
        )
    print("=" * 54 + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MarkupLM SWDE experiment — 4 verticals"
    )
    parser.add_argument(
        "--swde_path", required=True,
        help="Root of the SWDE dataset (parent of auto/, book/, groundtruth/, ...)",
    )
    parser.add_argument(
        "--processed_path", required=True,
        help="Directory for storing processed per-website feature pickles",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Directory for model checkpoints and result files",
    )
    parser.add_argument(
        "--model_name_or_path", default="microsoft/markuplm-base",
        help="HuggingFace model identifier or local path (default: markuplm-base)",
    )
    parser.add_argument(
        "--verticals", nargs="+", default=SELECTED_VERTICALS,
        metavar="V",
        help=f"Verticals to run (default: {' '.join(SELECTED_VERTICALS)})",
    )
    parser.add_argument(
        "--cuda", default="0", metavar="IDS",
        help="CUDA_VISIBLE_DEVICES string (default: 0)",
    )
    parser.add_argument(
        "--n_pages", type=int, default=HPARAMS["n_pages"],
        help=f"Pages per website (default: {HPARAMS['n_pages']})",
    )
    parser.add_argument(
        "--skip_data_prep", action="store_true",
        help="Skip pack_data + prepare_data (use when features are already generated)",
    )
    parser.add_argument(
        "--k_values", nargs="+", type=int, default=[1],
        metavar="K",
        help="n_seed values (training sites per fold) to test, e.g. --k_values 1 2 3 4 5 (default: 1)",
    )
    args = parser.parse_args()

    swde_path      = Path(args.swde_path).resolve()
    processed_path = Path(args.processed_path).resolve()
    output_dir     = Path(args.output_dir).resolve()
    pickle_path    = swde_path / f"swde-{args.n_pages}p.pickle"
    groundtruth    = swde_path / "groundtruth"

    # Environment: GPU selection + PYTHONPATH for local imports inside run_swde/
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda
    env["PYTHONPATH"] = ":".join(filter(None, [
        str(SCRIPTS_DIR),
        str(MARKUPLM_ROOT),
        env.get("PYTHONPATH", ""),
    ]))

    HPARAMS["n_pages"] = args.n_pages

    logger.info("Verticals : %s", args.verticals)
    logger.info("Model     : %s", args.model_name_or_path)
    logger.info("HPARAMS   : %s", HPARAMS)

    # ── Data preparation (one-time) ──────────────────────────────────────────
    if not args.skip_data_prep:
        pack_data(swde_path, pickle_path, args.verticals, args.n_pages)
        prepare_data(groundtruth, pickle_path, processed_path, args.verticals, args.n_pages)

    # ── Training + evaluation per vertical and per k ─────────────────────────
    for k in args.k_values:
        k_output_dir = output_dir / f"k{k}"
        logger.info("=== k=%d ===", k)
        for vertical in args.verticals:
            train_eval(vertical, processed_path, k_output_dir, args.model_name_or_path, env, n_seed=k)
        aggregate_and_report(k_output_dir, args.verticals, k=k)


if __name__ == "__main__":
    main()
