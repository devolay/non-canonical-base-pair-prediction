import gc
import pandas as pd
import argparse
import numpy as np
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch as PyGBatch

from pair_prediction.config import ModelConfig
from pair_prediction.model.lit_wrapper import LitWrapper
from pair_prediction.data.dataset import LinkPredictionDataset
from pair_prediction.constants import BASE_DIR


def get_amp_ctx(precision: str):
    """Return autocast context manager and dtype for the given precision."""
    precision = precision.lower()
    if precision == "fp32":
        return nullcontext(), torch.float32
    elif precision == "fp16":
        return torch.cuda.amp.autocast(dtype=torch.float16), torch.float16
    elif precision == "bf16":
        return torch.cuda.amp.autocast(dtype=torch.bfloat16), torch.bfloat16
    else:
        raise ValueError(f"Unknown precision: {precision}")

def unwrap_optimizers(opt_cfg):
    """
    Lit.configure_optimizers() can return optimizer, list, dict, or tuple.
    This tries to robustly extract the optimizer(s). We use the first one.
    """
    if isinstance(opt_cfg, (list, tuple)):
        first = opt_cfg[0]
        if isinstance(first, (list, tuple)):
            return first[0]
        return first
    if isinstance(opt_cfg, dict):
        return opt_cfg.get("optimizer", None)
    return opt_cfg

@torch.no_grad()
def measure_inference(model: nn.Module, batch, precision: str, warmup_steps=3, measure_steps=10):
    model.eval()

    amp_ctx, _ = get_amp_ctx(precision)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # Warmup
    for _ in range(warmup_steps):
        with amp_ctx:
            _ = model._common_step(batch)

    # Timed/Measured
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    total_ms = 0.0
    torch.cuda.reset_peak_memory_stats()

    for _ in range(measure_steps):
        starter.record()
        with amp_ctx:
            _ = model._common_step(batch)
        ender.record()
        torch.cuda.synchronize()
        total_ms += starter.elapsed_time(ender)

    peak_mem_bytes = torch.cuda.max_memory_allocated()
    return {
        "inference_mem_MB": peak_mem_bytes / (1024**2),
        "inference_time_ms_per_batch": total_ms / measure_steps,
    }

def measure_train_step(model: nn.Module, batch, precision: str):
    model.train()
    amp_ctx, _ = get_amp_ctx(precision)

    opt_cfg = model.configure_optimizers()
    optimizer = unwrap_optimizers(opt_cfg)
    assert optimizer is not None, "Could not unwrap optimizer from LitWrapper.configure_optimizers()"

    scaler = torch.cuda.amp.GradScaler(enabled=(precision in ["fp16", "bf16"]))

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    optimizer.zero_grad(set_to_none=True)
    with amp_ctx:
        loss = model.training_step(batch, batch_idx=0)
    if isinstance(loss, dict):
        loss = loss["loss"]
    # backward
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    peak_mem_bytes = torch.cuda.max_memory_allocated()
    return {"train_mem_MB": peak_mem_bytes / (1024**2)}

def build_single_batch(dset, batch_size, device):
    loader = DataLoader(dset, batch_size=batch_size, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    return batch.to(device)

def tile_graph(data: Data, k: int) -> Data:
    out = data.clone()

    edge_index = data.edge_index
    num_nodes = data.num_nodes

    edge_blocks = []
    edge_type_blocks = []
    features_blocks = []
    for i in range(k):
        offset = i * num_nodes
        edge_blocks.append(edge_index + offset)
        edge_type_blocks.extend(data.edge_type)
        features_blocks.append(data.features)

    out.edge_index = torch.cat(edge_blocks, dim=1)
    out.edge_type = np.array([edge_type_blocks])
    out.features = torch.cat(features_blocks, dim=1)
    out.seq = out.seq * k
    out.id = f"{out.id}_x{k}"
    
    return out


def build_length_scaled_batch(dset, base_index: int, k: int, device):
    """Take one sample from dataset and tile it k times into a single longer graph."""
    base = dset[base_index]
    big = tile_graph(base, k)
    return DataLoader([big], batch_size=1, shuffle=False, num_workers=0).dataset[0].to(device)


def benchmark(
    config_path: str,
    batch_sizes=(1, 2, 4, 8, 16),
    precisions=("fp32", "fp16", "bf16"),
    device="cuda",
    warmup_steps=3,
    measure_steps=10,
):
    cudnn.benchmark = True
    device = torch.device(device)

    cfg = ModelConfig.from_yaml(config_path)
    data_root = Path("data/")
    train_ds = LinkPredictionDataset(root=data_root, mode="train")

    results = []

    for prec in precisions:
        for bs in batch_sizes:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            model = LitWrapper(cfg).to(device)
            batch = build_single_batch(train_ds, bs, device)

            inf = measure_inference(model, batch, precision=prec,
                                    warmup_steps=warmup_steps, measure_steps=measure_steps)
            trn = measure_train_step(model, batch, precision=prec)

            try:
                L_est = int((batch.features.shape[0] if hasattr(batch, "features") else batch.x.shape[0]) / bs)
            except Exception:
                L_est = None

            results.append({
                "mode": "batch",
                "precision": prec,
                "batch_size": bs,
                "seq_length": L_est,
                **inf, **trn
            })

            del model, batch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=BASE_DIR / "configs" / "train-config.yaml", type=str, help="Path to YAML config.")
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[2, 8, 32, 64, 128, 256, 512])
    ap.add_argument("--precisions", type=str, nargs="+", default=["fp16", "bf16"])
    ap.add_argument("--warmup-steps", type=int, default=3)
    ap.add_argument("--measure-steps", type=int, default=10)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=str, default="bench_results.csv")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required for this benchmark."

    results = benchmark(
        config_path=args.config,
        batch_sizes=tuple(args.batch_sizes),
        precisions=tuple(args.precisions),
        device=args.device,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
    )

    df = pd.DataFrame(results)
    print(df)
    df.to_csv(args.out, index=False)
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
