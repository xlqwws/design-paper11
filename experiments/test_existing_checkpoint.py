"""Run causal validation/test replay without retraining an existing checkpoint."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import get_runtime_required_args, get_yml_cfg  # noqa: E402
from detection import dynamic_trace_gnn_testing  # noqa: E402


def main():
    args, unknown = get_runtime_required_args(return_unknown_args=True)
    if unknown:
        raise ValueError(f"Unknown arguments: {unknown}")
    cfg = get_yml_cfg(args)
    seed = cfg._seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model_dir = Path(cfg.detection.gnn_training._trained_models_dir)
    if not model_dir.is_dir() or not any(model_dir.glob("model_epoch_*")):
        raise FileNotFoundError(f"No existing checkpoints in {model_dir}")
    dynamic_trace_gnn_testing.main(cfg)
    print(f"edge_losses={cfg.detection.gnn_testing._edge_losses_dir}")
    print(f"online_tracing={cfg.attack_reconstruction.tracing._dynamic_tracing_dir}")


if __name__ == "__main__":
    main()
