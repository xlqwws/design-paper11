"""Causal structural sketch head for graph/window anomaly detection."""

from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np
import torch


def _stable_bucket(parts, buckets):
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") % buckets


def _entropy(counts):
    total = float(np.sum(counts))
    if total <= 0:
        return 0.0
    probability = np.asarray(counts, dtype=float) / total
    probability = probability[probability > 0]
    return float(-np.sum(probability * np.log(probability)))


def extract_causal_graph_sketch(
    data,
    num_node_types,
    num_edge_types,
    wl_buckets=256,
    transition_buckets=128,
    max_events=None,
):
    """Encode a graph using only its first ``max_events`` events.

    Calling this at window close is equivalent to incrementally observing the
    same event prefix; no future edge or label participates in the sketch.
    """
    event_count = len(data.src) if max_events is None else min(int(max_events), len(data.src))
    src = data.src[:event_count].detach().cpu().numpy().astype(np.int64)
    dst = data.dst[:event_count].detach().cpu().numpy().astype(np.int64)
    msg = data.msg[:event_count].detach().cpu()
    edge_type = data.edge_type[:event_count].detach().cpu()
    relations = (
        edge_type.numpy().astype(np.int64)
        if edge_type.ndim == 1 else torch.argmax(edge_type, dim=1).numpy().astype(np.int64)
    )
    src_types = torch.argmax(msg[:, :num_node_types], dim=1).numpy().astype(np.int64)
    dst_offset = num_node_types + num_edge_types
    dst_types = torch.argmax(
        msg[:, dst_offset:dst_offset + num_node_types], dim=1
    ).numpy().astype(np.int64)

    relation_hist = np.bincount(relations, minlength=num_edge_types).astype(float)
    src_type_hist = np.bincount(src_types, minlength=num_node_types).astype(float)
    dst_type_hist = np.bincount(dst_types, minlength=num_node_types).astype(float)
    triple_ids = (
        (src_types * num_edge_types + relations) * num_node_types + dst_types
    )
    triple_hist = np.bincount(
        triple_ids, minlength=num_node_types * num_edge_types * num_node_types
    ).astype(float)
    transition_hist = np.zeros(transition_buckets, dtype=float)
    for left, right in zip(relations[:-1], relations[1:]):
        transition_hist[_stable_bucket((int(left), int(right)), transition_buckets)] += 1.0

    node_types = {}
    adjacency = defaultdict(list)
    in_degree = defaultdict(int)
    out_degree = defaultdict(int)
    unique_edges = set()
    for source, target, relation, source_type, target_type in zip(
        src, dst, relations, src_types, dst_types
    ):
        source, target, relation = int(source), int(target), int(relation)
        node_types[source] = int(source_type)
        node_types[target] = int(target_type)
        adjacency[source].append(("out", relation, target))
        adjacency[target].append(("in", relation, source))
        out_degree[source] += 1
        in_degree[target] += 1
        unique_edges.add((source, target, relation))

    labels = {node: node_types[node] for node in node_types}
    wl_features = []
    for iteration in range(2):
        histogram = np.zeros(wl_buckets, dtype=float)
        updated = {}
        for node in sorted(labels):
            neighborhood = sorted(
                (direction, relation, labels.get(neighbor, -1))
                for direction, relation, neighbor in adjacency.get(node, [])
            )
            bucket = _stable_bucket((iteration, labels[node], tuple(neighborhood)), wl_buckets)
            histogram[bucket] += 1.0
            updated[node] = bucket
        wl_features.append(histogram)
        labels = updated

    nodes = sorted(node_types)
    indegrees = np.asarray([in_degree[node] for node in nodes], dtype=float)
    outdegrees = np.asarray([out_degree[node] for node in nodes], dtype=float)
    degrees = indegrees + outdegrees
    denominator = max(event_count, 1)
    node_denominator = max(len(nodes), 1)
    structural = np.asarray([
        np.log1p(event_count), np.log1p(len(nodes)), len(nodes) / denominator,
        len(unique_edges) / denominator, float(np.mean(src == dst)) if event_count else 0.0,
        float(np.max(degrees)) / denominator if len(degrees) else 0.0,
        float(np.std(degrees)) / max(float(np.mean(degrees)), 1e-12) if len(degrees) else 0.0,
        float(np.percentile(degrees, 95)) / denominator if len(degrees) else 0.0,
        float(np.max(indegrees)) / denominator if len(indegrees) else 0.0,
        float(np.max(outdegrees)) / denominator if len(outdegrees) else 0.0,
        _entropy(relation_hist) / max(np.log(max(num_edge_types, 2)), 1e-12),
        _entropy(src_type_hist + dst_type_hist) / max(np.log(max(num_node_types, 2)), 1e-12),
        len(unique_edges) / node_denominator,
    ], dtype=float)

    histograms = [relation_hist, src_type_hist, dst_type_hist, triple_hist, transition_hist] + wl_features
    normalized = [values / max(float(np.sum(values)), 1.0) for values in histograms]
    return np.concatenate(normalized + [structural]).astype(np.float32)
