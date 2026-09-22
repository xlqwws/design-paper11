"""Join immutable full-E3 node decisions to campaign labels for detection-rate reporting."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisions", required=True)
    parser.add_argument("--node-map", required=True)
    parser.add_argument("--ground-truth-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--variant", default="normalized_structure_only")
    args = parser.parse_args()
    decisions = pd.read_csv(args.decisions)
    predicted = set(
        decisions.loc[decisions[f"{args.variant}_predicted"].astype(int) == 1, "node_id"].astype(int)
    )
    node_map = json.loads(Path(args.node_map).read_text(encoding="utf-8"))
    uuid_to_node = {uuid.upper(): int(node) for node, uuid in node_map.items()}
    rows = []
    for path in sorted(Path(args.ground_truth_dir).glob("*.csv")):
        uuids = set()
        with path.open("r", encoding="utf-8", errors="replace") as file:
            for row in csv.reader(file):
                if row:
                    uuids.add(row[0].strip().upper())
        campaign_nodes = {uuid_to_node[uuid] for uuid in uuids if uuid in uuid_to_node}
        test_nodes = campaign_nodes & set(decisions["node_id"].astype(int))
        detected = test_nodes & predicted
        rows.append({
            "dataset": "E3-CADETS-Causal", "campaign": path.stem,
            "variant": args.variant, "campaign_nodes_in_corpus": len(campaign_nodes),
            "campaign_nodes_in_test": len(test_nodes), "detected_nodes": len(detected),
            "campaign_detected": int(bool(detected)),
            "node_recall_within_campaign": len(detected) / len(test_nodes) if test_nodes else None,
        })
    frame = pd.DataFrame(rows)
    summary = {
        "dataset": "E3-CADETS-Causal", "variant": args.variant,
        "campaigns": len(frame), "detected_campaigns": int(frame["campaign_detected"].sum()),
        "campaign_detection_rate": float(frame["campaign_detected"].mean()),
        "join_time": "after immutable online node decisions",
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "per_campaign.csv", index=False)
    pd.DataFrame([summary]).to_csv(out / "summary.csv", index=False)
    (out / "summary.json").write_text(json.dumps({"summary": summary, "campaigns": rows}, indent=2), encoding="utf-8")
    print(frame.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
