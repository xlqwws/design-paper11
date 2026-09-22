import json

from tqdm import tqdm

from encoders import DynamicTraceEncoder
from provnet_utils import *
from data_utils import *
from config import *
from model import *
from factory import *
import torch
from detection.evaluation_utils import calculate_threshold
from attack_reconstruction.online_dynamic_tracing import (
    OnlineDynamicAttackTracer,
    is_online_dynamic_enabled,
    load_calibration_scores,
    load_calibration_node_scores,
    resolve_online_threshold,
)


def _is_external_temporal_dataset(cfg):
    graph_source = str(getattr(cfg.dataset, "graph_source", "darpa_tc")).strip()
    prebuilt_dir = str(getattr(cfg.edge_featurization.embed_edges, "prebuilt_dir", "")).strip()
    return graph_source == "external_temporal" or bool(prebuilt_dir)


def _gen_external_nodeid2msg(*datasets):
    nodeid2msg = {}
    for dataset in datasets:
        for data in dataset:
            if hasattr(data, "nodeid2msg") and isinstance(data.nodeid2msg, dict):
                nodeid2msg.update(data.nodeid2msg)
                continue
            node_ids = torch.cat([data.src.to("cpu"), data.dst.to("cpu")]).unique().tolist()
            for node_id in node_ids:
                nodeid2msg[int(node_id)] = str(node_id)
    return nodeid2msg


def _load_external_nodeid2msg(cfg, *datasets):
    mapping_path = os.path.join(
        cfg.edge_featurization.embed_edges._edge_embeds_dir, "nodeid2msg.json"
    )
    if os.path.isfile(mapping_path):
        with open(mapping_path, "r", encoding="utf-8") as file:
            return {int(key): str(value) for key, value in json.load(file).items()}
    return _gen_external_nodeid2msg(*datasets)


def _load_external_relation_names(cfg):
    metadata_path = os.path.join(
        cfg.edge_featurization.embed_edges._edge_embeds_dir, "metadata.json"
    )
    if not os.path.isfile(metadata_path):
        return None
    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)
    relation_names = metadata.get("relations") or metadata.get("edge_types")
    return [str(value) for value in relation_names] if relation_names else None


@torch.no_grad()
def test(
        data,
        full_data,
        model,
        nodeid2msg,
        split,
        model_epoch_file,
        cfg,
        device,
        online_tracer=None,
        time_window_idx=None,
        relation_names=None,
):
    model.eval()

    time_with_loss = {}  # key: time，  value： the losses
    edge_list = []
    unique_nodes = torch.tensor([]).to(device=device)
    start_time = data.t[0]
    event_count = 0
    tot_loss = 0
    start = time.perf_counter()

    # NOTE: warning, this may reindexes the graph
    inference_batch_size = None
    if online_tracer is not None:
        inference_batch_size = cfg.detection.gnn_testing.causal_batch_size
    batch_loader = batch_loader_factory(
        cfg, data, model.graph_reindexer, batch_size=inference_batch_size
    )

    for batch in batch_loader:
        unique_nodes = torch.cat([unique_nodes, batch.edge_index.flatten()]).unique()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        batch_started_ns = time.perf_counter_ns()
        each_edge_loss = model(batch, full_data, inference=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        batch_latency_us = (time.perf_counter_ns() - batch_started_ns) / 1000.0
        tot_loss += each_edge_loss.sum().item()

        # If the graph has been reindexed in the loader, we retrieve original node IDs
        # to later find the labels
        if hasattr(batch, "original_edge_index"):
            edge_index = batch.original_edge_index
        else:
            edge_index = batch.edge_index
        
        num_events = each_edge_loss.shape[0]
        model_latency_per_event_us = batch_latency_us / max(num_events, 1)
        edge_types = torch.argmax(batch.edge_type, dim=1) + 1
        for i in range(num_events):
            srcnode = int(edge_index[0, i])
            dstnode = int(edge_index[1, i])

            srcmsg = nodeid2msg.get(srcnode, str(srcnode))
            dstmsg = nodeid2msg.get(dstnode, str(dstnode))
            t_var = int(batch.t[i])
            edge_type_idx = edge_types[i].item()
            if relation_names and 0 < edge_type_idx <= len(relation_names):
                edge_type = relation_names[edge_type_idx - 1]
            else:
                edge_type = rel2id.get(edge_type_idx, f"relation_{edge_type_idx}")
            loss = each_edge_loss[i]

            temp_dic = {
                'loss': float(loss),
                'srcnode': srcnode,
                'dstnode': dstnode,
                'srcmsg': srcmsg,
                'dstmsg': dstmsg,
                'edge_type': edge_type,
                'time': t_var,
                'model_latency_us': model_latency_per_event_us,
            }
            edge_list.append(temp_dic)
            if online_tracer is not None:
                online_tracer.observe_edge(temp_dic, time_window_idx)

        event_count += num_events
    tot_loss /= event_count

    # Here is a checkpoint, which records all edge losses in the current time window
    time_interval = ns_time_to_datetime_US(start_time) + "~" + ns_time_to_datetime_US(edge_list[-1]["time"])

    end = time.perf_counter()
    logs_dir = os.path.join(cfg.detection.gnn_testing._edge_losses_dir, split, model_epoch_file)
    os.makedirs(logs_dir, exist_ok=True)
    safe_time_interval = time_interval.replace(":", "-")
    csv_file = os.path.join(logs_dir, safe_time_interval + ".csv")

    df = pd.DataFrame(edge_list)
    df.to_csv(csv_file, sep=',', header=True, index=False, encoding='utf-8')

    if online_tracer is not None:
        online_tracer.close_time_window(time_window_idx, time_interval)

    # log(
    #     f'Time: {time_interval}, Loss: {tot_loss:.4f}, Nodes_count: {len(unique_nodes)}, Edges_count: {event_count}, Cost Time: {(end - start):.2f}s')


def main(cfg):
    train_data, val_data, test_data, _, max_node_num = load_all_datasets(cfg)
    causal_full_data = {
        "val": build_causal_full_data(train_data, val_data),
        "test": build_causal_full_data(train_data, test_data),
    }
    if _is_external_temporal_dataset(cfg):
        nodeid2msg = _load_external_nodeid2msg(cfg, val_data, test_data)
        relation_names = _load_external_relation_names(cfg)
    else:
        # load the map between nodeID and node labels
        cur, _ = init_database_connection(cfg)
        nodeid2msg = gen_nodeid2msg(cur=cur)
        nodeid2msg = {k: str(v) for k, v in nodeid2msg.items()}  # pre-compute because it's too slow in main loop
        relation_names = None

    # For each model trained at a given epoch, we test
    gnn_models_dir = cfg.detection.gnn_training._trained_models_dir
    all_trained_models = ["model_epoch_1"] if cfg._from_weights else listdir_sorted(gnn_models_dir)
    checkpoint_policy = str(cfg.detection.gnn_testing.checkpoint_policy).strip().lower()
    if checkpoint_policy == "latest" and all_trained_models:
        all_trained_models = [all_trained_models[-1]]
    elif checkpoint_policy != "all":
        raise ValueError("gnn_testing.checkpoint_policy must be `latest` or `all`")

    device = get_device(cfg)

    for trained_model in all_trained_models:
        log(f"Evaluation with model {trained_model}...")
        torch.cuda.empty_cache()
        model = build_model(data_sample=test_data[0], device=device, cfg=cfg, max_node_num=max_node_num)
        if cfg._from_weights:
            weights_name = getattr(cfg.dataset, "weights_name", cfg.dataset.name)
            model.load_state_dict(torch.load(
                os.path.join(cfg._from_weights_path, f"{weights_name}.pkl"),
                map_location=device,
            ))
        else:
            model = load_model(model, os.path.join(gnn_models_dir, trained_model))

        # TODO: we may want to move the validation set into the training for early stopping
        for graphs, split in [
            (val_data, "val"),
            (test_data, "test"),
        ]:
            log(f"    Testing {split} set...")
            # Validation mutates the temporal neighbor state. Restore the exact
            # training checkpoint before test so validation events cannot leak
            # into the online test stream.
            if split == "test":
                if cfg._from_weights:
                    if isinstance(model.encoder, DynamicTraceEncoder):
                        model.encoder.reset_state()
                else:
                    model = build_model(
                        data_sample=test_data[0], device=device, cfg=cfg, max_node_num=max_node_num
                    )
                    model = load_model(model, os.path.join(gnn_models_dir, trained_model))
            online_tracer = None
            if split == "test" and is_online_dynamic_enabled(cfg):
                val_tw_path = os.path.join(cfg.detection.gnn_testing._edge_losses_dir, "val", trained_model)
                val_thresholds = calculate_threshold(val_tw_path)
                calibration_scores = load_calibration_scores(val_tw_path)
                calibration_node_scores = load_calibration_node_scores(
                    val_tw_path, calibration_scores,
                    cfg.attack_reconstruction.tracing.online_dynamic.node_gate_top_k,
                )
                online_threshold = resolve_online_threshold(
                    val_thresholds, cfg, calibration_scores=calibration_scores
                )
                online_tracer = OnlineDynamicAttackTracer(
                    cfg=cfg,
                    model_epoch_file=trained_model,
                    split=split,
                    anomaly_threshold=online_threshold,
                    nodeid2msg=nodeid2msg,
                    calibration_scores=calibration_scores,
                    calibration_node_scores=calibration_node_scores,
                )
                log(
                    "[OnlineDynamicTracing] Enabled during causal GNN testing "
                    f"with threshold={online_threshold:.6f}, "
                    f"calibration_n={len(calibration_scores)}, "
                    f"batch_size={cfg.detection.gnn_testing.causal_batch_size}"
                )

            for tw_idx, g in enumerate(tqdm(graphs, desc=f"{split} set with {trained_model}")):
                g.to(device=device)
                test(
                    data=g,
                    full_data=causal_full_data[split],
                    model=model,
                    nodeid2msg=nodeid2msg,
                    split=split,
                    model_epoch_file=trained_model,
                    cfg=cfg,
                    device=device,
                    online_tracer=online_tracer,
                    time_window_idx=tw_idx,
                    relation_names=relation_names,
                )
                g.to("cpu")

            if online_tracer is not None:
                online_tracer.flush()

        del model


if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)
