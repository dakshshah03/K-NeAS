#!/usr/bin/env python3
"""
Repeated Training and Evaluation Script for NeAS-2
Trains a given configuration N times and reports mean/std of 2D & 3D metrics.
"""
import os
import sys
import argparse
import copy
import gc
import random
import numpy as np
import torch

_proj_root = os.path.abspath(os.path.dirname(__file__))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

def parse_early_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu", type=str, default=None)
    args, _ = parser.parse_known_args()
    return args

early_args = parse_early_args()
if early_args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = early_args.gpu
    print(f"Setting CUDA_VISIBLE_DEVICES = {early_args.gpu}")

from src.config import load_config
from src.trainer import Trainer
from src.dataset.tigre import TIGREDataset
from src.render.render import render_image
from src.utils.util import get_psnr, get_ssim, get_psnr_3d, get_ssim_3d
from src.eval_validation_metrics import (
    _build_models_from_checkpoint,
    sample_3d_volume_from_models,
    normalize_volume
)

def evaluate_checkpoint(ckpt_path, val_pickle, device="cuda", n_samples=128, chunk_size=4096):
    """Computes average 2D projection and 3D volume metrics for a checkpoint."""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not os.path.exists(val_pickle):
        raise FileNotFoundError(f"Validation dataset not found: {val_pickle}")

    ckpt = torch.load(ckpt_path, map_location=device)
    
    if ckpt.get('encoding_type', 'freq') == 'hash' and device == 'cpu':
        raise RuntimeError("Checkpoint uses hash encoding which requires CUDA; run with --device cuda or use a freq-encoded checkpoint.")

    sdf_model, att_model, s_param, num_materials = _build_models_from_checkpoint(ckpt, device)
    sdf_model.eval()
    att_model.eval()

    val_ds = TIGREDataset(val_pickle, n_rays=1024, type='val', device=device)
    n_views = val_ds.n_samples
    s_tensor = torch.tensor(s_param, device=device)

    proj_psnrs = []
    proj_ssims = []

    with torch.no_grad():
        for i in range(n_views):
            rays = val_ds.rays[i].to(device)
            projs = val_ds.projs[i].to(device)

            soft_selector = ckpt.get('args', {}).get('network', {}).get('soft_selector', True)
            pred_img = render_image(rays, sdf_model, att_model, s_tensor,
                                    n_samples, chunk_size=chunk_size,
                                    tau=None, num_materials=num_materials,
                                    soft_selector=soft_selector)

            # GT projection is exp(-projs)
            gt_proj = torch.exp(-projs)

            p_psnr = get_psnr(pred_img, gt_proj)
            p_ssim = get_ssim(pred_img, gt_proj)

            proj_psnrs.append(float(p_psnr.item()))
            proj_ssims.append(float(p_ssim))

    if device.startswith('cuda'):
        torch.cuda.empty_cache()

    pred_volume = sample_3d_volume_from_models(
        sdf_model, att_model, s_param, val_ds.voxels, num_materials, 
        soft_selector=ckpt.get('args', {}).get('network', {}).get('soft_selector', True), 
        chunk_size=chunk_size, device=device
    )
    gt_volume = val_ds.image.cpu().numpy()
    pred_volume = normalize_volume(pred_volume, gt_volume)

    vol_psnr = float(get_psnr_3d(pred_volume, gt_volume))
    vol_ssim = float(get_ssim_3d(pred_volume, gt_volume))

    return {
        "avg_2d_proj_psnr": np.mean(proj_psnrs),
        "avg_2d_proj_ssim": np.mean(proj_ssims),
        "vol_3d_psnr": vol_psnr,
        "vol_3d_ssim": vol_ssim
    }

def main():
    parser = argparse.ArgumentParser(description="Train a config multiple times and report mean & std metrics.")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--gpu", type=str, default=None, help="GPU index to use (e.g. '0')")
    parser.add_argument("--runs", type=int, default=3, help="Number of repetitions to train (default: 3)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed (default: 42)")
    parser.add_argument("--n_samples", type=int, default=128, help="Number of samples for rendering (default: 128)")
    parser.add_argument("--chunk_size", type=int, default=4096, help="Chunk size for rendering (default: 4096)")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    base_cfg["_config_path"] = args.config

    original_expname = base_cfg["exp"]["expname"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = []

    for run_idx in range(1, args.runs + 1):
        run_seed = args.seed + run_idx
        random.seed(run_seed)
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)
        torch.cuda.manual_seed_all(run_seed)

        print("\n" + "="*50)
        print(f"Starting Run {run_idx}/{args.runs} (Seed: {run_seed}) for config: {args.config}")
        print("="*50 + "\n")

        run_cfg = copy.deepcopy(base_cfg)
        run_cfg["exp"]["expname"] = f"{original_expname}_run_{run_idx}"

        trainer = Trainer(run_cfg, device=device)
        trainer.start()

        ckpt_path = trainer.ckptdir
        val_pickle = run_cfg["exp"]["datadir"]

        if hasattr(trainer, "writer") and trainer.writer is not None:
            trainer.writer.close()

        del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\nEvaluating Run {run_idx}/{args.runs}...")
        metrics = evaluate_checkpoint(
            ckpt_path=ckpt_path,
            val_pickle=val_pickle,
            device=device,
            n_samples=args.n_samples,
            chunk_size=args.chunk_size
        )
        print(f"Run {run_idx} Results:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        results.append(metrics)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "X"*50)
    print("FINAL SUMMARY REPORT")
    print("X"*50 + "\n")
    print(f"Config: {args.config}")
    print(f"Total Runs: {args.runs}\n")

    metric_keys = ["avg_2d_proj_psnr", "avg_2d_proj_ssim", "vol_3d_psnr", "vol_3d_ssim"]
    summary_data = {}

    for k in metric_keys:
        values = [r[k] for r in results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        summary_data[k] = (mean_val, std_val)
        
        print(f"{k}:")
        print(f"  Mean: {mean_val:.4f}")
        print(f"  Std:  {std_val:.4f}")
        print(f"  Runs: {[f'{v:.4f}' for v in values]}")
        print("-" * 30)

    report_dir = os.path.dirname(ckpt_path)
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "repeated_runs_summary.txt")
    with open(report_path, "w") as f:
        f.write(f"Config: {args.config}\n")
        f.write(f"Total Runs: {args.runs}\n\n")
        for k in metric_keys:
            mean_val, std_val = summary_data[k]
            values = [r[k] for r in results]
            f.write(f"{k}:\n")
            f.write(f"  Mean: {mean_val:.4f}\n")
            f.write(f"  Std:  {std_val:.4f}\n")
            f.write(f"  Runs: {values}\n")
            f.write("-" * 30 + "\n")
    print(f"\nSummary report saved to: {report_path}")

if __name__ == "__main__":
    main()
