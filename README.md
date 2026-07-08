# $K$-NeAS: Scalable Multi-Material CT Reconstruction Using Neural SDFs

This repository contains the source code for the paper: **$K$-NeAS: Scalable Multi-Material CT Reconstruction Using Neural SDFs**.

---

## Abstract

> Computed Tomography (CT) carries significant ionizing radiation risks, driving the need for sparse-view reconstruction. Implicit scene representations (ISRs) address this by recovering continuous volumetric attenuation fields directly from sparse projections, and recent geometry-aware extensions jointly model surface geometry alongside attenuation to improve fidelity and enable clean tissue segmentation without manual thresholding. However, these methods remain limited by manually tuned attenuation bounds and rigid two-material constraints. This paper proposes $K$-NeAS, a unified and scalable architecture for automated, multi-material surface reconstruction. We replace independent material networks with a shared latent backbone and introduce a fully differentiable $K$-material sequential soft selector to model an arbitrary number of overlapping tissues. To eliminate manual tuning, we automate attenuation bounding using a Gaussian Mixture Model (GMM) and implement a scheduled auxiliary floater loss to mitigate geometric hallucinations common under extreme sparsity. Evaluated across four clinical Cone-Beam CT (CBCT) datasets, $K$-NeAS successfully scales to arbitrary material counts, achieving superior 3D volumetric fidelity at $K=3$ materials on complex multi-tissue regions such as the Abdomen ($33.28\text{ dB}$ 3D PSNR vs. $31.40\text{ dB}$ single-material NeAS baseline, a $+1.88\text{ dB}$ improvement). Furthermore, our model exhibits enhanced robustness under sparse-sampling conditions, outperforming baseline 3D PSNR by up to $1.17\text{ dB}$ under 5- and 10-view constraints.

---

## Key Contributions

- a soft, fully differentiable sequential occupancy filter, built on a shared attenuation backbone with $K$ lightweight prediction heads, that resolves material membership as a pointwise function of local SDF occupancy for an arbitrary number of materials
- an unsupervised Gaussian Mixture Model (GMM) that automatically estimates per-material attenuation bounds by sampling the volume of a converged single-material prior, eliminating manual scene-specific tuning
- a scheduled auxiliary floater regularization that suppresses spurious empty space geometry during the early stages of optimization


---

## Experimental Results

### 1. Quantitative Comparison (Average of 3 Runs)

| Scene | Config | NeAS (2D PSNR) | $K$-NeAS (2D PSNR) | NeAS (2D SSIM) | $K$-NeAS (2D SSIM) | NeAS (3D PSNR) | $K$-NeAS (3D PSNR) | NeAS (3D SSIM) | $K$-NeAS (3D SSIM) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Abdomen** | 1M | 46.744 | 46.850 | 0.992 | 0.993 | 31.399 | 32.648 | 0.820 | 0.845 |
| | 2M | 47.098 | 47.204 | 0.993 | 0.993 | 31.391 | 32.722 | 0.823 | 0.846 |
| | 3M | --- | **47.687** | --- | **0.994** | --- | **33.276** | --- | **0.858** |
| | 4M | --- | 47.346 | --- | **0.994** | --- | 32.916 | --- | 0.850 |
| **Chest** | 1M | 45.949 | 45.230 | 0.991 | 0.991 | 31.508 | 31.396 | 0.913 | 0.908 |
| | 2M | **46.609** | 45.672 | 0.992 | 0.992 | 32.171 | 31.784 | 0.918 | 0.915 |
| | 3M | --- | 46.036 | --- | **0.993** | --- | **32.181** | --- | **0.920** |
| | 4M | --- | 45.427 | --- | 0.992 | --- | 31.592 | --- | 0.911 |
| **Foot** | 1M | 42.230 | 42.717 | 0.981 | **0.983** | 31.007 | 31.579 | **0.900** | 0.893 |
| | 2M | 42.224 | 42.570 | 0.981 | 0.982 | 31.092 | 31.388 | 0.891 | 0.889 |
| | 3M | --- | **42.800** | --- | **0.983** | --- | **31.603** | --- | 0.889 |
| | 4M | --- | 42.717 | --- | **0.983** | --- | 31.462 | --- | 0.887 |
| **Jaw** | 1M | **39.451** | 34.321 | **0.966** | 0.953 | **34.099** | 31.672 | **0.882** | 0.797 |
| | 2M | 37.472 | 34.366 | 0.963 | 0.956 | 33.514 | 31.828 | 0.832 | 0.808 |
| | 3M | --- | 34.332 | --- | 0.955 | --- | 31.847 | --- | 0.806 |
| | 4M | --- | 34.541 | --- | 0.956 | --- | 31.879 | --- | 0.810 |

### 2. Sparse-View Chest Reconstruction Comparison (2-Material Configurations)

| Views | NeAS (2D PSNR) | $K$-NeAS (2D PSNR) | NeAS (2D SSIM) | $K$-NeAS (2D SSIM) | NeAS (3D PSNR) | $K$-NeAS (3D PSNR) | NeAS (3D SSIM) | $K$-NeAS (3D SSIM) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **5** | 30.473 | 30.813 | 0.899 | 0.913 | 20.596 | **21.767** | 0.502 | 0.554 |
| **10** | 35.058 | 35.997 | 0.940 | 0.948 | 23.414 | **24.486** | 0.638 | 0.673 |
| **20** | 40.307 | 40.788 | 0.972 | 0.973 | 26.424 | 27.112 | 0.760 | 0.775 |
| **50** | **46.609** | 45.672 | 0.992 | 0.992 | 32.171 | 31.784 | 0.918 | 0.915 |

### 3. Component Ablations on Foot Dataset (2-Material Config, Single Run)

| Configuration | 2D PSNR $\uparrow$ | 2D SSIM $\uparrow$ | 3D PSNR $\uparrow$ | 3D SSIM $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: |
| $K$-Selector Only | 41.938 | 0.991 | 31.071 | 0.884 |
| $K$-Selector + Floater Regularization | 43.115 | 0.983 | 31.623 | 0.893 |
| $K$-Selector + Shared Backbone Body | 42.544 | 0.981 | 31.245 | 0.883 |
| **$K$-NeAS** (Full Model) | **43.245** | **0.983** | **31.630** | **0.890** |

---

## Architectural Ablations

This codebase implements the 4 configurations shown in the component ablations above, which are located under `config/ablations/foot/`:

1. **1_KSelector**: K-Material Soft Selector without floater regularization or shared latent space (uses independent networks).
2. **2_KSelector_Floater**: K-Material Soft Selector with Floater Regularization but independent networks.
3. **3_KSelector_Shared**: K-Material Soft Selector with Shared Latent Space but without Floater Regularization.
4. **4_KNEAS** (Full Model): K-Material Soft Selector with both Shared Latent Space and Floater Regularization active simultaneously.

---


## Data Preparation

1. Download the clinical CBCT dataset from the official [NAF Google Drive Folder](https://drive.google.com/drive/folders/1BJYR4a4iHpfFFOAdbEe5O_7Itt1nukJd).
2. Create a `data` directory in the root of the project and place the TIGRE-formatted `.pickle` files inside:
```bash
mkdir data
# Place files like data/foot_50.pickle inside
```

---

## Scripts in Project Root Directory

This repository provides Python scripts and bash wrappers for training, evaluation, comparison, and analysis. Below are the usage details for each script.

### 1. Training Scripts

#### [train.py](train.py)
Used to train a single configuration manually.
- **Usage**:
  ```bash
  python train.py --config <PATH_TO_CONFIG>
  ```
- **Example**:
  ```bash
  python train.py --config config/foot_configs/foot_50_1m_hash.yaml
  ```

#### [train_repeated.py](train_repeated.py)
Trains a given configuration $N$ times (with different random seeds) and reports the mean and standard deviation of 2D and 3D metrics. A summary report is saved to the run's checkpoint directory as `repeated_runs_summary.txt`.
- **Usage**:
  ```bash
  python train_repeated.py --config <PATH_TO_CONFIG> [options]
  ```
- **Options**:
  - `--config`: (Required) Path to the YAML config file.
  - `--gpu`: GPU index to use (e.g., `0`).
  - `--runs`: Number of repetitions to train (default: `3`).
  - `--seed`: Base random seed (default: `42`).
  - `--n_samples`: Number of samples for rendering (default: `128`).
  - `--chunk_size`: Chunk size for rendering (default: `4096`).
- **Example**:
  ```bash
  python train_repeated.py --config config/foot_configs/foot_50_1m_hash.yaml --runs 3 --gpu 0
  ```

---

### 2. Evaluation Scripts

#### [eval_4_views.py](eval_4_views.py)
Generates ground-truth and predicted projection PNGs for a chosen set of validation views from a checkpoint.
- **Usage**:
  ```bash
  python eval_4_views.py --checkpoint <PATH_TO_CHECKPOINT> --gt_pickle <PATH_TO_VAL_PICKLE> --views <VIEW_NUMBERS> [options]
  ```
- **Options**:
  - `--checkpoint`: (Required) Path to the checkpoint `.pth` file.
  - `--gt_pickle`: (Required) Path to validation dataset pickle file.
  - `--views`: Space-separated list of 1-indexed validation view numbers to evaluate.
  - `--device`: Device to use (e.g., `cuda` or `cpu`, default: `cuda`).
- **Outputs**:
  - Saved under `<checkpoint_dir>/gt/view_NNN.png` and `<checkpoint_dir>/preds/view_NNN.png`.
- **Example**:
  ```bash
  python eval_4_views.py --checkpoint checkpoints/foot_50_4m_hash/checkpoint_epoch_1000.pth --gt_pickle data/foot_50.pickle --views 1 15 30 45
  ```

#### [run_eval.sh](run_eval.sh)
Runs evaluation across all 4 anatomies (`chest`, `foot`, `jaw`, `abdomen`) and sizes (`1m`, `2m`, `3m`, `4m`) for the `hash` variant at a given epoch (default: `1000`). It compiles the metrics into a summary CSV.
- **Usage**:
  ```bash
  ./run_eval.sh [--gpu DEVICE]
  ```
- **Example**:
  ```bash
  ./run_eval.sh --gpu cuda:0
  ```

#### [run_eval_ablations.sh](run_eval_ablations.sh)
Runs evaluation for the 4 foot ablations at epoch 1000 and generates a summary CSV.
- **Usage**:
  ```bash
  ./run_eval_ablations.sh [--gpu DEVICE] <anatomy>
  ```
- **Example**:
  ```bash
  ./run_eval_ablations.sh --gpu cuda:0 foot
  ```

#### [generate_comparison_images.sh](generate_comparison_images.sh)
Shell script wrapper that runs `eval_4_views.py` across multiple anatomies (`chest`, `foot`, `jaw`, `abdomen`) for size `2m` and variant `hash` (by default) to generate comparison projection images.
- **Usage**:
  ```bash
  ./generate_comparison_images.sh [--gpu DEVICE] [--views "VIEWS"] [--epoch N]
  ```
- **Example**:
  ```bash
  ./generate_comparison_images.sh --gpu cuda:1 --views "1 15 30 45" --epoch 1000
  ```

---

### 3. Pipeline & Automation Scripts

#### [run_ablations.sh](run_ablations.sh)
Executes targeted ablations sequentially (1 to 4) for a specified region and automatically registers and updates logging to Weights & Biases (WandB).
- **Usage**:
  ```bash
  ./run_ablations.sh [options]
  ```
- **Options**:
  - `-g, --gpu`: Target GPU ID (default: `0`).
  - `-r, --region`: Anatomy region: `abdomen`, `chest`, `foot`, or `jaw` (default: `foot`). Note: Ablation configs are loaded from `config/ablations/<region>/`.
  - `-a, --ablations`: Comma-separated list of ablations to run (e.g., `1,2,3,4`).
  - `-p, --project`: WandB project name (default: `neas_ablations`).
  - `-k, --key`: WandB API Key (optional if already logged in).
- **Example**:
  ```bash
  ./run_ablations.sh -g 0 -r foot -a 1,3,4 -p "foot_ablations"
  ```

#### [run_experiments_parallel.sh](run_experiments_parallel.sh)
Distributes multi-config training jobs across all available GPUs (or a specified list of GPU IDs) in parallel. It supports running single training runs or repeated runs.
- **Usage**:
  ```bash
  ./run_experiments_parallel.sh [options]
  ```
- **Options**:
  - `-g, --gpus`: Comma-separated list of GPU IDs (e.g. `0,1,2`). If omitted, auto-detects and uses all available GPUs.
  - `-r, --repeat`: Number of runs/repetitions per configuration (default: `1`). If $> 1$, uses `train_repeated.py` internally.
- **Example**:
  ```bash
  ./run_experiments_parallel.sh -g "0,1" -r 3
  ```

---

### 4. Attenuation & Config Analysis

#### [analyze_attenuation.py](analyze_attenuation.py)
Samples the prediction volume from a trained 1M-NeAS model checkpoint, fits Gaussian Mixture Models (GMM) via EM to find optimal material boundaries $K$, and outputs configuration parameters (alpha/beta/weights) directly formatted to copy and paste into a multi-material configuration file.
- **Usage**:
  ```bash
  python analyze_attenuation.py --checkpoint <CHECKPOINT_PATH> --config <CONFIG_PATH> [options]
  ```
- **Options**:
  - `--checkpoint`: (Required) Path to the trained checkpoint.
  - `--config`: (Required) Path to matching configuration YAML file.
  - `--output`: Output path for visualization PNG.
  - `--device`: Target device (default: `cuda`).
  - `--k_max`: Max number of materials to consider (default: `4`).
  - `--air_threshold`: Air/background classification threshold (default: `0.01`).
- **Example**:
  ```bash
  python analyze_attenuation.py --checkpoint checkpoints/foot_50_1m_hash/checkpoint_epoch_500.pth --config config/foot_configs/foot_50_1m_hash.yaml --output attenuation_analysis_foot.png
  ```

---

## References & Citation

If you use this work or the datasets, please cite the following papers:

```bibtex
@inproceedings{zha2022naf,
  title={NAF: Neural Attenuation Fields for Sparse-View CBCT Reconstruction},
  author={Zha, Ruyi and Zhang, Yanhao and Li, Hongdong},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={442--452},
  year={2022},
  organization={Springer}
}

@misc{zhu2025neas3dreconstructionxray,
      title={NeAS: 3D Reconstruction from X-ray Images using Neural Attenuation Surface}, 
      author={Chengrui Zhu and Ryoichi Ishikawa and Masataka Kagesawa and Tomohisa Yuzawa and Toru Watsuji and Takeshi Oishi},
      year={2025},
      eprint={2503.07491},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2503.07491}, 
}
```
