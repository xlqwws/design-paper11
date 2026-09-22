import argparse
import random

import torch
import wandb
import numpy as np
from provnet_utils import remove_underscore_keys, log

from graph_construction import (
    build_dynamic_trace_graphs,
)
from edge_featurization import (
    build_feature_word2vec,
    embed_edges_feature_word2vec,
)
from detection import (
    dynamic_trace_gnn_training,
    dynamic_trace_gnn_testing,
    evaluation,
)

from config import (
    get_yml_cfg,
    get_runtime_required_args,
)

from attack_reconstruction import (
    tracing,
)

import time

def main(cfg, args, **kwargs):
    if cfg.detection.gnn_training.use_seed:
        seed = cfg._seed
        random.seed(seed)
        np.random.seed(seed)

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    t0 = time.time()
    t1 = t2 = t3 = t0

    # By default, graph_construction + training/detection is run
    if not args.run_from_training:
        # graph_construction
        build_dynamic_trace_graphs.main(cfg)
        t1 = time.time()
        
        # Featurization
        build_feature_word2vec.main(cfg)
        t2 = time.time()
        embed_edges_feature_word2vec.main(cfg)
        t3 = time.time()

    # Detection
    dynamic_trace_gnn_training.main(cfg)
    torch.cuda.empty_cache()
    t4 = time.time()
    dynamic_trace_gnn_testing.main(cfg)
    t5 = time.time()
    if args.skip_offline_evaluation:
        log("Skipping database-backed offline evaluation; online outputs remain available for causal evaluation.")
    else:
        evaluation.main(cfg)
    t6 = time.time()

    # Triage
    tracing.main(cfg)
    t7 = time.time()

    time_consumption = {
        "time_total": round(t7 - t0, 2),
        "time_build_graphs": round(t1 - t0, 2),
        "time_embed_nodes": round(t2 - t1, 2),
        "time_embed_edges": round(t3 - t2, 2),
        "time_gnn_training": round(t4 - t3, 2),
        "time_gnn_testing": round(t5 - t4, 2),
        "time_evaluation": round(t6 - t5, 2),
        "time_tracing": round(t7 - t6, 2),
    }

    log("==" * 30)
    log("Run finished. Time consumed in each step:")
    for k, v in time_consumption.items():
        log(f"{k}: {v} s")

    log("==" * 30)
    wandb.log(time_consumption)


if __name__ == '__main__':
    args, unknown_args = get_runtime_required_args(return_unknown_args=True)
    
    exp_name = args.exp if args.exp != "" else \
        args.__dict__["dataset"]
        # "|".join([f"{k.split('.')[-1]}={v}" for k, v in args.__dict__.items() if "." in k and v is not None])
    tags = args.tags.split(",") if args.tags != "" else [args.model]
    
    wandb.init(
        mode="online" if args.wandb else "disabled",
        project="dynamic_trace", # Can be changed
        name=exp_name,
        tags=tags,
    )
    
    if len(unknown_args) > 0:
        raise argparse.ArgumentTypeError(f"Unknown args {unknown_args}")

    cfg = get_yml_cfg(args)
    wandb.config.update(remove_underscore_keys(dict(cfg), keys_to_keep=["_task_path"]))

    main(cfg, args)
    
    wandb.finish()
