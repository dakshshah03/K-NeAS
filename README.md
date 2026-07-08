# NeAS-2: Multi-Material Neural Attenuation Fields

This repository contains the source code for the proposed architecture **K-Material SDFs for Neural Attenuation Fields**.

It builds upon the Neural Attenuation Fields (NeAS) pipeline, extending it to support robust multi-tissue reconstruction from sparse-view CBCT through unified shared latents, continuous differentiable $K$-material soft composition, dataset-driven (GMM) attenuation bounding, and scheduled floater suppression.

---

## Architectural Ablations

This codebase implements 4 specific architectural configurations (ablations) to isolate the impact of different methodologies:

1. **1_KSelector**: K-Material Soft Selector without floater regularization or shared latent space (uses independent networks).
2. **2_KSelector_Floater**: K-Material Soft Selector with Floater Regularization but independent networks.
3. **3_KSelector_Shared**: K-Material Soft Selector with Shared Latent Space but without Floater Regularization.
4. **4_KNEAS** (Full Model): K-Material Soft Selector with both Shared Latent Space and Floater Regularization active simultaneously.

These configurations are defined under `config/ablations/foot/`.

---

## Data Preparation

Create a `data` directory in the root of the project and place the TIGRE-formatted `.pickle` files inside:
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
