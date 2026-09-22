"""Build a leakage-free temporal Unicorn-Wget graph pilot from official archives."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import tarfile
from collections import Counter
from pathlib import Path

import torch
from torch_geometric.data import TemporalData


ARCHIVES = (
    ("benign.tar.gz", 0),
    ("attack_baseline.tar.gz", 1),
    ("attack_interval.tar.gz", 1),
)


def natural_key(text):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


def split_for(label, benign_index):
    if label == 1:
        return "test"
    if benign_index < 75:
        return "train"
    if benign_index < 100:
        return "val"
    return "test"


def stable_bucket(value, buckets):
    digest = hashlib.sha256(value.encode("utf-8", errors="ignore")).digest()
    return int.from_bytes(digest[:8], "big") % buckets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=r"D:\download\Unicorn Wget")
    parser.add_argument("--out-dir", default="artifacts/raw_temporal_pilot/Unicorn-Wget/edge_embeds")
    parser.add_argument("--edges-per-graph", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main():
    args = parse_args()
    output = Path(args.out_dir).resolve()
    if output.exists() and any(output.rglob("*.TemporalData")):
        raise FileExistsError(f"Refusing to overwrite existing TemporalData in {output}")
    for split in ("train", "val", "test"):
        (output / split).mkdir(parents=True, exist_ok=True)

    graph_records = []
    node_types = set()
    edge_types = set()
    raw_edges = 0
    benign_index = 0
    for archive_name, label in ARCHIVES:
        archive_path = Path(args.raw_dir) / archive_name
        with tarfile.open(archive_path, "r:gz") as archive:
            members = sorted(
                [member for member in archive.getmembers()
                 if member.isfile() and "/stream/" in member.name
                 and member.name.endswith(".txt") and "/._" not in member.name
                 and not Path(member.name).name.startswith("._")],
                key=lambda member: natural_key(member.name),
            )
            for member in members:
                graph_id = len(graph_records)
                rng = random.Random(args.seed + graph_id)
                reservoir = []
                seen = 0
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
                    seen += 1
                    raw_edges += 1
                    src, dst = fields[0], fields[1]
                    src_type, dst_type, edge_type = payload[:3]
                    node_types.update((src_type, dst_type))
                    edge_types.add(edge_type)
                    event = (seen, src, src_type, dst, dst_type, edge_type)
                    if len(reservoir) < args.edges_per_graph:
                        reservoir.append(event)
                    else:
                        selected = rng.randrange(seen)
                        if selected < args.edges_per_graph:
                            reservoir[selected] = event
                split = split_for(label, benign_index)
                if label == 0:
                    benign_index += 1
                graph_records.append({
                    "graph_id": graph_id, "archive": archive_name, "member": member.name,
                    "label": label, "split": split, "events": sorted(reservoir),
                    "raw_edges": seen,
                })
                print(f"Scanned {len(graph_records)}/175 graphs ({raw_edges:,} edges)", flush=True)

    node_types = sorted(node_types)
    edge_types = sorted(edge_types)
    node_type_to_id = {name: stable_bucket(name, 8) for name in node_types}
    edge_type_to_id = {name: stable_bucket(name, 32) for name in edge_types}
    node_ids = {}
    node_messages = {}
    files_by_split = Counter()
    edges_by_split = Counter()
    test_truth = []
    test_window = 0

    def node_id(graph_id, local_id, node_type):
        key = (graph_id, local_id)
        if key not in node_ids:
            index = len(node_ids)
            node_ids[key] = index
            node_messages[index] = f"graph={graph_id};type={node_type};local={local_id}"
        return node_ids[key]

    for record in graph_records:
        events = record["events"]
        if not events:
            continue
        graph_id = record["graph_id"]
        src = torch.tensor([node_id(graph_id, row[1], row[2]) for row in events], dtype=torch.long)
        dst = torch.tensor([node_id(graph_id, row[3], row[4]) for row in events], dtype=torch.long)
        src_type = torch.tensor([node_type_to_id[row[2]] for row in events], dtype=torch.long)
        dst_type = torch.tensor([node_type_to_id[row[4]] for row in events], dtype=torch.long)
        edge_type = torch.tensor([edge_type_to_id[row[5]] for row in events], dtype=torch.long)
        message = torch.cat([
            torch.nn.functional.one_hot(src_type, 8),
            torch.nn.functional.one_hot(edge_type, 32),
            torch.nn.functional.one_hot(dst_type, 8),
        ], dim=1).float()
        timestamp = torch.tensor(
            [graph_id * 1_000_000_000 + row[0] for row in events], dtype=torch.long
        )
        data = TemporalData(src=src, dst=dst, t=timestamp, msg=message, edge_type=edge_type)
        split = record["split"]
        torch.save(data, output / split / f"graph_{graph_id:05d}.TemporalData")
        files_by_split[split] += 1
        edges_by_split[split] += len(events)
        if split == "test":
            test_truth.append({
                "time_window": test_window, "graph_id": graph_id,
                "label": int(record["label"]), "source_member": record["member"],
            })
            test_window += 1

    with (output / "nodeid2msg.json").open("w", encoding="utf-8") as file:
        json.dump({str(key): value for key, value in node_messages.items()}, file)
    with (output / "ground_truth_graphs.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["time_window", "graph_id", "label", "source_member"])
        writer.writeheader()
        writer.writerows(test_truth)
    metadata = {
        "dataset": "Unicorn-Wget", "status": "pilot",
        "sampling": "deterministic_per_graph_reservoir", "seed": args.seed,
        "edges_per_graph": args.edges_per_graph,
        "raw_sources": [str(Path(args.raw_dir) / name) for name, _ in ARCHIVES],
        "raw_edges": raw_edges, "graphs": len(graph_records),
        "node_type_cardinality": len(node_types), "edge_type_cardinality": len(edge_types),
        "node_type_buckets": 8, "edge_type_buckets": 32,
        "node_types": [f"node_bucket_{index}" for index in range(8)],
        "edge_types": [f"relation_bucket_{index}" for index in range(32)],
        "type_encoding": "fixed_sha256_bucket_without_labels",
        "split_rule": "benign 0-74 train, 75-99 calibration, 100-124 test; all 50 attacks test",
        "label_leakage_control": "archive labels are joined only after feature tensors are built",
        "files_by_split": dict(files_by_split), "sampled_edges_by_split": dict(edges_by_split),
        "nodes": len(node_ids), "test_attack_graphs": sum(row["label"] for row in test_truth),
        "ground_truth": str(output / "ground_truth_graphs.csv"),
    }
    with (output / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
