"""Collision-free sparse causal sketches for full Unicorn-Wget graphs."""

from __future__ import annotations

import argparse
import json
import re
import tarfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from detection.conformal import conformal_predictions  # noqa: E402
from build_unicorn_wget_temporal import ARCHIVES, natural_key  # noqa: E402
from evaluate_cached_full_sketch_repeats import split_ids  # noqa: E402


def scan(raw_dir: Path):
    records, counters = [], []
    raw_edges = 0
    started = time.perf_counter()
    for archive_name, label in ARCHIVES:
        with tarfile.open(raw_dir / archive_name, "r:gz") as archive:
            members = sorted([
                member for member in archive.getmembers()
                if member.isfile() and "/stream/" in member.name and member.name.endswith(".txt")
                and "/._" not in member.name and not Path(member.name).name.startswith("._")
            ], key=lambda member: natural_key(member.name))
            for member in members:
                features = Counter()
                edges = 0
                stream = archive.extractfile(member)
                if stream is None:
                    continue
                for raw in stream:
                    fields = raw.decode("utf-8", errors="ignore").strip().split()
                    if len(fields) < 3:
                        continue
                    payload = fields[2].split(":")
                    if len(payload) < 3:
                        continue
                    src, dst, relation = payload[:3]
                    features[f"relation={relation}"] += 1
                    features[f"src={src}"] += 1
                    features[f"dst={dst}"] += 1
                    features[f"triple={src}|{relation}|{dst}"] += 1
                    edges += 1
                graph_id = len(records)
                records.append({"graph_id": graph_id, "label": label, "archive": archive_name,
                                "member": member.name, "raw_edges": edges})
                counters.append(features)
                raw_edges += edges
                if len(records) % 25 == 0:
                    print(f"graphs={len(records)}/175 edges={raw_edges:,}", flush=True)
    return records, counters, raw_edges, time.perf_counter() - started


def transformed_variants(matrix):
    probability = normalize(matrix, norm="l1", axis=1)
    hellinger = probability.copy()
    hellinger.data = np.sqrt(hellinger.data)
    tfidf = TfidfTransformer(sublinear_tf=True, norm="l2").fit_transform(matrix)
    return {"exact_hellinger": hellinger.tocsr(), "exact_tfidf_cosine": tfidf.tocsr()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=r"D:\download\Unicorn Wget")
    parser.add_argument("--out-dir", default="artifacts/tifs_results/Unicorn-Wget/full_exact_sparse_sketch")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--split-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    records, counters, raw_edges, scan_seconds = scan(Path(args.raw_dir))
    vectorizer = DictVectorizer(sparse=True, dtype=np.float32)
    matrix = vectorizer.fit_transform(counters).tocsr()
    variants = transformed_variants(matrix)
    rows = []
    for seed in args.split_seeds:
        train_ids, val_ids, test_ids, labels = split_ids("Unicorn-Wget", seed)
        for variant, features in variants.items():
            metric = "euclidean" if variant == "exact_hellinger" else "cosine"
            model = NearestNeighbors(n_neighbors=5, metric=metric, algorithm="brute").fit(features[train_ids])
            val_scores = model.kneighbors(features[val_ids], return_distance=True)[0].mean(axis=1)
            scores = model.kneighbors(features[test_ids], return_distance=True)[0].mean(axis=1)
            predicted = conformal_predictions(val_scores, scores, args.alpha)
            rows.append({
                "dataset": "Unicorn-Wget", "variant": variant, "split_seed": seed,
                "raw_edges": raw_edges, "vocabulary_size": matrix.shape[1],
                "train_graphs": len(train_ids), "calibration_graphs": len(val_ids),
                "test_graphs": len(test_ids), "reported_graphs": int(predicted.sum()),
                "precision": float(precision_score(labels, predicted, zero_division=0)),
                "recall": float(recall_score(labels, predicted, zero_division=0)),
                "f1": float(f1_score(labels, predicted, zero_division=0)),
                "mcc": float(matthews_corrcoef(labels, predicted)),
                "auroc": float(roc_auc_score(labels, scores)),
                "average_precision": float(average_precision_score(labels, scores)),
                "alpha": args.alpha, "scan_seconds": scan_seconds,
                "label_leakage_control": "exact vocabulary is unlabeled; fit benign train; calibrate benign validation",
            })
    frame = pd.DataFrame(rows)
    summary = frame.groupby(["dataset", "variant"], as_index=False).agg(
        repetitions=("split_seed", "count"), precision_mean=("precision", "mean"),
        recall_mean=("recall", "mean"), f1_mean=("f1", "mean"), f1_min=("f1", "min"),
        mcc_mean=("mcc", "mean"), auroc_mean=("auroc", "mean"),
        average_precision_mean=("average_precision", "mean"),
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(out / "unlabelled_exact_counts.npz", matrix)
    pd.DataFrame(records).to_csv(out / "graph_index.csv", index=False)
    frame.to_csv(out / "split_repetitions.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    (out / "metadata.json").write_text(json.dumps({
        "raw_edges": raw_edges, "graphs": len(records), "vocabulary_size": matrix.shape[1],
        "scan_seconds": scan_seconds, "feature_names_sha_only": True,
    }, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
