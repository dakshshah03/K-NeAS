# NeAS-2: Multi-Material Neural Attenuation Fields

This repository contains the source code for the proposed architecture in Daksh Shah's Senior Thesis: **[K-Material SDFs for Neural Attenuation Fields](https://dakshshah.com/files/Shah_Daksh_Senior_Thesis.pdf)**.

It builds upon an implementation of the original [NeAS](https://arxiv.org/abs/2503.07491) pipeline, extending it to support robust multi-tissue reconstruction from sparse-view CBCT through unified shared latents, continuous differentiable $K$-material soft composition, dataset-driven (GMM) attenuation bounding, and scheduled floater suppression.

## Architectural Ablations

This repository implements 6 specific architectural ablations to isolate the impact of our proposed methodologies against the baseline:

1. **NeAS** (Base model with original piecewise selector and manually tuned rigid thresholds)
2. **NeAS + GMM** (Data-driven GMM attenuation bound selection instead of manual thresholds)
3. **NeAS + GMM + Shared Latent Space** (Unified backbone MLP instead of independent networks)
4. **NeAS + GMM + K-Material Soft Selector** (Fully differentiable multi-layer compositing versus piecewise hard selector bounds)
5. **NeAS + GMM + Floater Regularization** (Scheduled mask-based supervision on zero-attenuation/air rays)
6. **NeAS Full** (All proposed extensions active simultaneously)

## Data Preparation
Create a `data` directory in the root of the project if it doesn't exist, and place the TIGRE-formatted `.pickle` files inside (e.g., `foot_50.pickle`).

## Running Ablations

You can execute targeted ablations sequentially and sync them directly to Weights & Biases using the provided `run_ablations.sh` runner script:

```bash
./run_ablations.sh -g <GPU_ID> -a <Ablations> -p <WandB_Project> -k <WandB_API_Key>
```

**Options**:
- `-g, --gpu`: Target GPU ID (e.g., `0`)
- `-a, --ablations`: Comma-separated list of ablations to run (e.g., `1,3,6` or `1,2,3,4,5,6`)
- `-p, --project`: WandB project name to log runs to (default: `neas_ablations`)
- `-k, --key`: Your WandB API Auth Key

**Example**:
```bash
./run_ablations.sh -g 1 -a 1,6 -p "foot_ablations" -k "YOUR_WANDB_TOKEN"
```

## Running Standard Training

To train a single configuration manually without the batch script:

```bash
python train.py --config config/ablations/6_NeAS_Full.yaml
```

*Outputs, including model checkpoints and evaluation iterations (PSNR, SSIM, rendering samples), will be saved sequentially down into the `checkpoints/<expname>` folders.*

## TODOs

- [ ] **Brain Dataset Setup:** Integration and testing for the Brain dataset are currently underway to validate performance on low-contrast soft-tissue structures within the skull. Pipeline loader scripts and configs will be updated once testing is complete.
