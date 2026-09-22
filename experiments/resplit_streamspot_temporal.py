"""Create the corrected scenario-stratified StreamSpot split without rescanning raw data."""

from __future__ import annotations

import csv
import json
import os
import shutil
from collections import Counter
from pathlib import Path

from build_streamspot_temporal import split_for_graph


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts" / "raw_temporal_pilot" / "StreamSpot" / "edge_embeds"
TARGET = ROOT / "artifacts" / "raw_temporal_pilot_stratified" / "StreamSpot" / "edge_embeds"


def main():
    if TARGET.exists() and any(TARGET.rglob("*.TemporalData")):
        raise FileExistsError(f"Refusing to overwrite {TARGET}")
    for split in ["train", "val", "test"]:
        (TARGET / split).mkdir(parents=True, exist_ok=True)
    source_files = {}
    for path in SOURCE.glob("*/*.TemporalData"):
        graph_id = int(path.stem.split("_")[-1])
        source_files[graph_id] = path
    if set(source_files) != set(range(600)):
        raise ValueError(f"Expected graph IDs 0-599, found {len(source_files)}")

    counts = Counter()
    truth = []
    test_window = 0
    for graph_id in range(600):
        split = split_for_graph(graph_id)
        source = source_files[graph_id]
        destination = TARGET / split / source.name
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
        counts[split] += 1
        if split == "test":
            truth.append({
                "time_window": test_window, "graph_id": graph_id,
                "label": int(300 <= graph_id <= 399),
            })
            test_window += 1

    shutil.copy2(SOURCE / "nodeid2msg.json", TARGET / "nodeid2msg.json")
    with (TARGET / "ground_truth_graphs.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["time_window", "graph_id", "label"])
        writer.writeheader()
        writer.writerows(truth)
    old_metadata = json.loads((SOURCE / "metadata.json").read_text(encoding="utf-8"))
    metadata = {
        **old_metadata,
        "protocol_version": "streamspot-scenario-stratified-v2",
        "split_rule": (
            "within each benign scenario: repetitions 0-59 train, 60-79 calibration, "
            "80-99 test; attack IDs 300-399 test only"
        ),
        "files_by_split": dict(counts),
        "sampled_edges_by_split": {
            split: count * int(old_metadata["edges_per_graph"])
            for split, count in counts.items()
        },
        "ground_truth": str(TARGET / "ground_truth_graphs.csv"),
        "resplit_from": str(SOURCE),
        "label_leakage_control": (
            "public scenario IDs define benign coverage; attack graphs are test-only; "
            "no test metric selects the split"
        ),
    }
    (TARGET / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
