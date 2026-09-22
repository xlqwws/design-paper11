"""Build a leakage-free StreamSpot TemporalData pilot from the official TSV.

The full source has about 89.8M edges.  Pilot mode uses deterministic reservoir
sampling independently inside every graph, then restores original edge order.
This is suitable for pipeline and graph-level evaluation, not a full-stream
latency or path-reconstruction claim.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import tarfile
from collections import Counter
from pathlib import Path

import torch
from torch_geometric.data import TemporalData


NODE_TYPES = list("abcdefgh")
EDGE_TYPES = list("ABCDEFGHijklmnopqrstuvwxyz")


def split_for_graph(graph_id: int) -> str:
    if 300 <= graph_id <= 399:
        return "test"
    scenario_index = graph_id % 100
    if scenario_index < 60:
        return "train"
    if scenario_index < 80:
        return "val"
    return "test"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default=r"D:\download\all.tar\all.tar")
    parser.add_argument("--out-dir", default="artifacts/raw_temporal_pilot/StreamSpot/edge_embeds")
    parser.add_argument("--edges-per-graph", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists() and any(out_dir.rglob("*.TemporalData")):
        raise FileExistsError(f"Refusing to overwrite existing TemporalData in {out_dir}")
    for split in ["train", "val", "test"]:
        (out_dir / split).mkdir(parents=True, exist_ok=True)

    reservoirs = {graph_id: [] for graph_id in range(600)}
    seen = Counter()
    rng = {graph_id: random.Random(args.seed + graph_id) for graph_id in range(600)}
    raw_lines = 0
    with tarfile.open(args.raw, "r") as archive:
        stream = archive.extractfile("all.tsv")
        if stream is None:
            raise FileNotFoundError("all.tsv is missing from StreamSpot archive")
        for raw in stream:
            fields = raw.decode("utf-8", errors="ignore").rstrip().split("\t")
            if len(fields) < 6:
                continue
            src, src_type, dst, dst_type, edge_type, graph_text = fields[:6]
            graph_id = int(graph_text)
            if src_type not in NODE_TYPES or dst_type not in NODE_TYPES or edge_type not in EDGE_TYPES:
                raise ValueError(f"Unknown StreamSpot type in row: {fields[:6]}")
            raw_lines += 1
            seen[graph_id] += 1
            event = (raw_lines, src, src_type, dst, dst_type, edge_type)
            reservoir = reservoirs[graph_id]
            if len(reservoir) < args.edges_per_graph:
                reservoir.append(event)
            else:
                selected = rng[graph_id].randrange(seen[graph_id])
                if selected < args.edges_per_graph:
                    reservoir[selected] = event
            if raw_lines % 10_000_000 == 0:
                print(f"Scanned {raw_lines:,} edges", flush=True)

    node_ids = {}
    node_labels = {}
    files_by_split = Counter()
    edges_by_split = Counter()
    ground_truth_rows = []
    test_window = 0

    def node_id(graph_id: int, node_type: str, local_id: str) -> int:
        key = (graph_id, node_type, local_id)
        if key not in node_ids:
            index = len(node_ids)
            node_ids[key] = index
            node_labels[index] = f"{graph_id}:{node_type}:{local_id}"
        return node_ids[key]

    for graph_id in range(600):
        events = sorted(reservoirs[graph_id], key=lambda item: item[0])
        if not events:
            continue
        split = split_for_graph(graph_id)
        src = torch.tensor(
            [node_id(graph_id, event[2], event[1]) for event in events], dtype=torch.long
        )
        dst = torch.tensor(
            [node_id(graph_id, event[4], event[3]) for event in events], dtype=torch.long
        )
        timestamps = torch.tensor([event[0] for event in events], dtype=torch.long)
        src_types = torch.tensor([NODE_TYPES.index(event[2]) for event in events], dtype=torch.long)
        dst_types = torch.tensor([NODE_TYPES.index(event[4]) for event in events], dtype=torch.long)
        edge_types = torch.tensor([EDGE_TYPES.index(event[5]) for event in events], dtype=torch.long)
        message = torch.cat(
            [torch.nn.functional.one_hot(src_types, len(NODE_TYPES)),
             torch.nn.functional.one_hot(edge_types, len(EDGE_TYPES)),
             torch.nn.functional.one_hot(dst_types, len(NODE_TYPES))], dim=1,
        ).float()
        data = TemporalData(src=src, dst=dst, t=timestamps, msg=message, edge_type=edge_types)
        torch.save(data, out_dir / split / f"graph_{graph_id:05d}.TemporalData")
        files_by_split[split] += 1
        edges_by_split[split] += len(events)
        if split == "test":
            ground_truth_rows.append({
                "time_window": test_window,
                "graph_id": graph_id,
                "label": int(300 <= graph_id <= 399),
            })
            test_window += 1

    with (out_dir / "nodeid2msg.json").open("w", encoding="utf-8") as file:
        json.dump({str(key): value for key, value in node_labels.items()}, file)
    gt_path = out_dir / "ground_truth_graphs.csv"
    with gt_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["time_window", "graph_id", "label"])
        writer.writeheader()
        writer.writerows(ground_truth_rows)
    metadata = {
        "dataset": "StreamSpot",
        "status": "pilot",
        "sampling": "deterministic_per_graph_reservoir",
        "seed": args.seed,
        "edges_per_graph": args.edges_per_graph,
        "raw_source": str(Path(args.raw).resolve()),
        "raw_edges": raw_lines,
        "graphs": len([value for value in seen.values() if value]),
        "node_types": NODE_TYPES,
        "edge_types": EDGE_TYPES,
        "split_rule": "within each benign scenario: repetitions 0-59 train, 60-79 calibration, 80-99 test; attack IDs 300-399 test only",
        "attack_label_rule": "official graph IDs 300-399",
        "label_leakage_control": "graph label is written only after model input construction",
        "files_by_split": dict(files_by_split),
        "sampled_edges_by_split": dict(edges_by_split),
        "nodes": len(node_ids),
        "ground_truth": str(gt_path),
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
