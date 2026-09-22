from config import *
from provnet_utils import *
from collections import defaultdict
import wandb
import os
import torch

from .tracing_methods import (
    depimpact,
)
from . import online_dynamic_tracing

def get_new_stats(tw_to_info,
                  evaluation_results,
                  cfg):
    method = cfg.detection.evaluation.used_method.strip()
    if method == "node_tw_evaluation":
        flat_y_truth = []
        flat_y_hat = []
        scores = []
        for tw, nid_to_result in evaluation_results.items():
            for nid, result in nid_to_result.items():
                score, y_hat, y_true = result["score"], result["y_hat"], result["y_true"]
                scores.append(score)
                flat_y_truth.append(y_true)
                if int(tw) in tw_to_info:
                    flat_y_hat.append(int(str(nid) in tw_to_info[int(tw)]['subgraph_nodes']))
                else:
                    flat_y_hat.append(0)

        new_stats = classifier_evaluation(flat_y_truth, flat_y_hat, scores)
    elif method == "node_evaluation":
        node_results = {}
        for tw, nid_to_result in evaluation_results.items():
            for nid, result in nid_to_result.items():
                score, y_hat, y_true = result["score"], result["y_hat"], result["y_true"]

                if nid not in node_results:
                    node_results[nid] = {}
                node_results[nid]['score'] = max(score, node_results[nid].get('score', 0))
                node_results[nid]['y_true'] = y_true
                if int(tw) in tw_to_info:
                    node_results[nid]['y_hat'] = int(int(str(nid) in tw_to_info[int(tw)]['subgraph_nodes']) or node_results[nid].get('y_hat', 0))
                else:
                    node_results[nid]['y_hat'] = node_results[nid].get('y_hat', 0)

        flat_y_truth = []
        flat_y_hat = []
        scores = []

        for nid, data in node_results.items():
            flat_y_truth.append(data["y_true"])
            flat_y_hat.append(data["y_hat"])
            scores.append(data["score"])
        new_stats = classifier_evaluation(flat_y_truth, flat_y_hat, scores)

    return new_stats

def transfer_results_of_node_evaluation(results_without_tw, tw_to_timestr, cfg):
    base_dir = cfg.graph_construction.build_graphs._graphs_dir
    results = defaultdict(lambda: defaultdict(dict))

    for tw, timestr in tw_to_timestr.items():
        day = timestr[8:10].lstrip('0')
        graph_dir = os.path.join(base_dir, f"graph_{day}/{timestr}")
        graph = torch.load(graph_dir)

        for node_id in graph.nodes():
            node_id = int(node_id)

            if node_id in results_without_tw:
                results[tw][node_id]['score'] = results_without_tw[node_id]['score']
                results[tw][node_id]['y_hat'] = results_without_tw[node_id]['y_hat']
                results[tw][node_id]['y_true'] = results_without_tw[node_id]['y_true']

    return results

def evaluate_online_dynamic_outputs(cfg):
    in_dir = cfg.detection.evaluation.node_evaluation._precision_recall_dir
    test_losses_dir = os.path.join(cfg.detection.gnn_testing._edge_losses_dir, "test")
    if not os.path.isdir(in_dir) or not os.path.isdir(test_losses_dir):
        log("[OnlineDynamicTracing] Skip metric evaluation because detection outputs are missing.")
        return None

    best_mcc, best_stats = -1e6, {}
    best_model_epoch = listdir_sorted(test_losses_dir)[-1]
    for model_epoch_dir in listdir_sorted(test_losses_dir):
        stats_file = os.path.join(in_dir, f"stats_{model_epoch_dir}.pth")
        if not os.path.exists(stats_file):
            continue
        stats = torch.load(stats_file)
        if stats["mcc"] > best_mcc:
            best_mcc = stats["mcc"]
            best_stats = stats
            best_model_epoch = model_epoch_dir

    sorted_tw_paths = sorted(os.listdir(os.path.join(cfg.edge_featurization.embed_edges._edge_embeds_dir, 'test')))
    tw_to_time = {}
    for tw, tw_file in enumerate(sorted_tw_paths):
        tw_to_time[tw] = tw_file[:-20]

    method = cfg.detection.evaluation.used_method.strip()
    results_file = os.path.join(in_dir, f"result_{best_model_epoch}.pth")
    if not os.path.exists(results_file):
        log(f"[OnlineDynamicTracing] Missing evaluation results: {results_file}")
        return None

    if method == "node_tw_evaluation":
        results = torch.load(results_file)
    elif method == "node_evaluation":
        if str(getattr(cfg.dataset, "graph_source", "darpa_tc")).strip() == "external_temporal":
            log("[OnlineDynamicTracing] Skip node_evaluation metric remapping for external temporal datasets.")
            return None
        results_without_tw = torch.load(results_file)
        results = transfer_results_of_node_evaluation(results_without_tw, tw_to_time, cfg)
    else:
        log(f"[OnlineDynamicTracing] Unsupported evaluation method for metric summary: {method}")
        return None

    online_results = os.path.join(
        cfg.attack_reconstruction.tracing._dynamic_tracing_dir,
        best_model_epoch,
        "test",
        "results.pth",
    )
    if not os.path.exists(online_results):
        log(f"[OnlineDynamicTracing] Online tracing result not found: {online_results}")
        return None

    tw_to_info = torch.load(online_results)
    new_stats = get_new_stats(
        tw_to_info=tw_to_info,
        evaluation_results=results,
        cfg=cfg,
    )

    log(f"Best model epoch is {best_model_epoch}")
    log("==" * 20)
    log("Before online_dynamic_tracing:")
    for k, v in best_stats.items():
        log(f"{k}: {v}")
    log("==" * 20)

    stats_traced = {}
    log("After online_dynamic_tracing:")
    for k, v in new_stats.items():
        log(f"{k}: {v}")
        stats_traced["online_dynamic_" + k] = v
    log("==" * 20)
    wandb.log(stats_traced)
    return new_stats

def main(cfg):
    if online_dynamic_tracing.is_online_dynamic_enabled(cfg):
        online_dynamic_tracing.summarize(cfg)
        evaluate_online_dynamic_outputs(cfg)
        return

    in_dir = cfg.detection.evaluation.node_evaluation._precision_recall_dir
    test_losses_dir = os.path.join(cfg.detection.gnn_testing._edge_losses_dir, "test")

    best_mcc, best_stats = -1e6, {}
    best_model_epoch = listdir_sorted(test_losses_dir)[-1]
    for model_epoch_dir in listdir_sorted(test_losses_dir):

        stats_file = os.path.join(in_dir, f"stats_{model_epoch_dir}.pth")
        stats = torch.load(stats_file)
        if stats["mcc"] > best_mcc:
            best_mcc = stats["mcc"]
            best_stats = stats
            best_model_epoch = model_epoch_dir

    sorted_tw_paths = sorted(os.listdir(os.path.join(cfg.edge_featurization.embed_edges._edge_embeds_dir, 'test')))
    tw_to_time = {}
    for tw, tw_file in enumerate(sorted_tw_paths):
        tw_to_time[tw] = tw_file[:-20]

    method = cfg.detection.evaluation.used_method.strip()

    results_file = os.path.join(in_dir, f"result_{best_model_epoch}.pth")
    if method == "node_tw_evaluation":
        results = torch.load(results_file)
    elif method == "node_evaluation":
        results_without_tw = torch.load(results_file)
        results = transfer_results_of_node_evaluation(results_without_tw, tw_to_time, cfg)

    if cfg.attack_reconstruction.tracing.used_method == 'depimpact':
        tw_to_info, all_traced_nodes = depimpact.main(results, tw_to_time, cfg)
        new_stats = get_new_stats(
            tw_to_info=tw_to_info,
            evaluation_results=results,
            cfg=cfg
        )

        log(f"Best model epoch is {best_model_epoch}")
        log("==" * 20)
        log(f"Before attack_reconstruction:")
        for k, v in best_stats.items():
            log(f"{k}: {v}")
        log("==" * 20)

        stats_traced = {}
        log(f"After attack_reconstruction:")
        for k, v in new_stats.items():
            log(f"{k}: {v}")
            stats_traced["tracing_" + k] = v
        log("==" * 20)

        # wandb.log(best_stats)
        wandb.log(stats_traced)


if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)
