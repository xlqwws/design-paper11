"""Build label-free GPU pooled node semantics for E3 campaign days."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytz
import torch


EASTERN = pytz.timezone("America/New_York")


def partition_paths(test_dir: Path, days: set[int]) -> dict[int, list[Path]]:
    grouped = {day: [] for day in days}
    for path in sorted(test_dir.glob("*.TemporalData")):
        data = torch.load(path)
        day = datetime.fromtimestamp(int(data.t[0]) / 1e9, EASTERN).day
        if day in grouped:
            grouped[day].append(path)
    return grouped


def aggregate_day(
    paths: list[Path], nodes: np.ndarray, embedding_dim: int, device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(nodes)
    semantic_sum = torch.zeros((count, embedding_dim), dtype=torch.float32, device=device)
    semantic_absmax = torch.zeros_like(semantic_sum)
    observations = torch.zeros(count, dtype=torch.float32, device=device)

    for index, path in enumerate(paths):
        data = torch.load(path)
        src_local = torch.from_numpy(
            np.searchsorted(nodes, data.src.cpu().numpy()).astype(np.int64)
        ).to(device)
        dst_local = torch.from_numpy(
            np.searchsorted(nodes, data.dst.cpu().numpy()).astype(np.int64)
        ).to(device)
        msg = data.msg.to(device)
        dst_type_start = msg.shape[1] - embedding_dim - 3
        src_semantic = msg[:, 3:3 + embedding_dim]
        dst_semantic = msg[:, dst_type_start + 3:]

        semantic_sum.index_add_(0, src_local, src_semantic)
        semantic_sum.index_add_(0, dst_local, dst_semantic)
        ones = torch.ones(len(src_local), dtype=torch.float32, device=device)
        observations.index_add_(0, src_local, ones)
        observations.index_add_(0, dst_local, ones)

        src_index = src_local[:, None].expand(-1, embedding_dim)
        dst_index = dst_local[:, None].expand(-1, embedding_dim)
        semantic_absmax.scatter_reduce_(
            0, src_index, src_semantic.abs(), reduce="amax", include_self=True
        )
        semantic_absmax.scatter_reduce_(
            0, dst_index, dst_semantic.abs(), reduce="amax", include_self=True
        )
        if (index + 1) % 25 == 0:
            print(f"windows={index + 1}/{len(paths)}", flush=True)

    mean = semantic_sum / observations.clamp_min(1)[:, None]
    mean = torch.nn.functional.normalize(mean, p=2, dim=1)
    return mean.cpu().numpy(), semantic_absmax.cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", default="artifacts/raw_temporal_full/E3-CADETS/edge_embeds"
    )
    parser.add_argument(
        "--source-cache", default="artifacts/feature_cache/E3-CADETS-Causal/cross_campaign_transfer"
    )
    parser.add_argument(
        "--out-dir", default="artifacts/feature_cache/E3-CADETS-Causal/aggregated_semantics"
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    source_cache = Path(args.source_cache)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    embedding_dim = int(metadata["embedding_dim"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    grouped = partition_paths(data_dir / "test", {6, 12, 13})

    for day in (6, 12, 13):
        source = np.load(source_cache / f"day_{day}.npz")
        nodes = source["nodes"]
        original = source["features"]
        structure_temporal = original[:, :38]
        latest = original[:, 38:38 + embedding_dim]
        mean, absmax = aggregate_day(grouped[day], nodes, embedding_dim, device)
        pooled = np.concatenate([structure_temporal, mean, absmax], axis=1).astype(np.float32)
        all_views = np.concatenate(
            [structure_temporal, latest, mean, absmax], axis=1
        ).astype(np.float32)
        np.savez_compressed(
            out / f"day_{day}.npz", nodes=nodes,
            structure_temporal=structure_temporal.astype(np.float32),
            latest=latest.astype(np.float32), mean=mean.astype(np.float32),
            absmax=absmax.astype(np.float32), pooled=pooled, all_views=all_views,
        )
        print(
            f"day={day} nodes={len(nodes):,} pooled={pooled.shape} all={all_views.shape}",
            flush=True,
        )


if __name__ == "__main__":
    main()
