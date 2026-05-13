#!/bin/bash

# Default arguments
GPU="0"
ABLATIONS="1,2,3,4,5,6"
WANDB_PROJECT="neas_ablations"
WANDB_KEY=""

usage() {
  echo "Usage: $0 [options]"
  echo "  -g, --gpu           GPU ID to use (default: 0)"
  echo "  -a, --ablations     Comma-separated list of ablations to run (1-6). e.g., 1,2,6 (default: 1,2,3,4,5,6)"
  echo "  -p, --project       WandB project name (default: neas_ablations)"
  echo "  -k, --key           WandB API key (required, or ensure wandb is already logged in)"
  exit 1
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -g|--gpu) GPU="$2"; shift ;;
        -a|--ablations) ABLATIONS="$2"; shift ;;
        -p|--project) WANDB_PROJECT="$2"; shift ;;
        -k|--key) WANDB_KEY="$2"; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

if [ -n "$WANDB_KEY" ]; then
  export WANDB_API_KEY="$WANDB_KEY"
  # Log in to wandb locally so it's ready
  wandb login "$WANDB_KEY"
else
  echo -e "\e[33mWarning: WandB API key not provided (-k). Assuming you are already logged in.\e[0m"
fi

export CUDA_VISIBLE_DEVICES="$GPU"

declare -A config_map
config_map[1]="1_NeAS.yaml"
config_map[2]="2_NeAS_GMM.yaml"
config_map[3]="3_NeAS_GMM_Shared.yaml"
config_map[4]="4_NeAS_GMM_Soft.yaml"
config_map[5]="5_NeAS_GMM_Floater.yaml"
config_map[6]="6_NeAS_Full.yaml"

IFS=',' read -ra ABL_ARRAY <<< "$ABLATIONS"

for abl in "${ABL_ARRAY[@]}"; do
    if [[ -n "${config_map[$abl]}" ]]; then
        cfg_file="config/ablations/${config_map[$abl]}"
        echo ""
        echo -e "\e[34m========================================================\e[0m"
        echo -e "\e[32mRunning Ablation $abl: $cfg_file (GPU: $GPU)\e[0m"
        echo -e "\e[34m========================================================\e[0m"
        
        # Inject wandb configuration directly into the yaml safely
        # Use python to robustly update yaml configurations
        uv run -c "
import yaml
with open('$cfg_file', 'r') as f:
    cfg = yaml.safe_load(f)
cfg['log']['use_wandb'] = True
cfg['log']['wandb_project'] = '$WANDB_PROJECT'
with open('$cfg_file', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)
"
        
        # Run the training script
        uv run train.py --config "$cfg_file"
        
        echo -e "\e[32mAblation $abl completed.\e[0m"
    else
        echo -e "\e[31mUnknown ablation number: $abl. Skipping...\e[0m"
    fi
done

echo "All specified ablations finished."
