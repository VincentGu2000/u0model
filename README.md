# GR00T N1.5 — U0 Robot Policy

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)

A fine-tuning and deployment framework for the [NVIDIA GR00T N1.5](https://huggingface.co/nvidia/GR00T-N1.5-3B) Vision-Language-Action (VLA) model, adapted for the **U0 underwater robot**. This project provides tools for data loading, LoRA/full fine-tuning, evaluation, and inference.

> **Note**: This project is forked from [NVIDIA Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) and customized for U0 robot applications.

## Features

- **Fine-Tuning**: LoRA and full fine-tuning support for GR00T N1.5 with multi-GPU training
- **Evaluation**: Policy evaluation with per-trajectory MSE metrics and visualization
- **Inference**: HTTP and ZMQ server modes for real-time robot deployment
- **Data**: Compatible with LeRobot data format

## Project Structure

```
├── gr00t/                  # Core library
│   ├── data/               # Dataset loading and transforms
│   ├── eval/               # Evaluation wrappers and services
│   ├── experiment/         # Training runner and data configs
│   ├── model/              # Model architecture (backbone, action head, policy)
│   └── utils/              # Utility functions
├── scripts/                # Training, evaluation, and inference scripts
├── deployment_scripts/     # TensorRT deployment tools (from upstream, experimental)
├── examples/               # Robot-specific modality configs
│   └── U0bot/              # U0 robot modality configurations
├── demo_data/              # Example dataset for quick start
└── tests/                  # Unit tests
```

---

## 1. Installation

Clone the repo:

```sh
git clone https://gitee.com/vincent-gu/u0.git
cd u0
```

Create a new conda environment and install the dependencies. We recommend Python 3.10:

```sh
conda create -n gr00t python=3.10
conda activate gr00t
pip install --upgrade setuptools
pip install -e .[base]
```

Install flash-attn module:

```sh
pip install --no-build-isolation flash-attn==2.7.1.post4
```

> **Note**: Make sure your CUDA version is 12.4. Otherwise, you may have trouble properly configuring the flash-attn module. For GPUs with sm_120 like RTX PRO 6000 Blackwell, try CUDA 13.0. If you encounter issues, try the following solutions.

For CUDA 13.0:

```sh
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple

pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.0/flash_attn-2.8.3+cu130torch2.10-cp310-cp310-linux_x86_64.whl
```

For CUDA 12.4:

```sh
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.1.post4/flash_attn-2.7.1.post4+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

Installing CUDA 12.4 (optional):

```sh
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
sudo sh cuda_12.4.0_550.54.14_linux.run
```

In the TUI interface, accept the EULA, uncheck "Driver" (keep it unselected) if you already have the driver, ensure that "CUDA Toolkit" is selected, and keep the installation path as default (`/usr/local/cuda-12.4`).

---

## 2. Quick Setup: Configure Local Paths

This project uses environment variables for all file paths (model weights, datasets, etc.) so that the same commands work across different machines.

```sh
# Copy the example config and edit with your actual paths
cp .env.example .env
# Then edit .env, for example:
#   MODEL_BASE_DIR=/data/models
#   DATA_BASE_DIR=/data/datasets
#   ROS_WS_DIR=/home/user/ros_ws
```

Before running any command in this README, load your path configuration:

```sh
source .env
```

> **Tip**: If you use `make` commands, the `Makefile` will automatically load `.env` for you.

---

## 3. Download Model Weights

Choose one of the following options based on your needs:

### Option A: Use Pre-Fine-Tuned Weights (No Training Required)

If you do not need to fine-tune the model yourself, you can directly download our fine-tuned U0 weights from Hugging Face:

```sh
hf download Vincent2025hello/u0_final --local-dir $MODEL_BASE_DIR/u0_final --local-dir-use-symlinks False
```

After downloading, simply point `--model-path` to `$MODEL_BASE_DIR/u0_final` in the evaluation and inference steps below.

### Option B: Download NVIDIA Pretrained Weights (For Self Fine-Tuning)

If you want to fine-tune the model yourself, download the NVIDIA GR00T N1.5 pretrained weights first:

```sh
hf download nvidia/GR00T-N1.5-3B --local-dir $MODEL_BASE_DIR/GR00T-N1.5-3B --local-dir-use-symlinks False
```

> **Note**: Self fine-tuning requires GPU resources and training data. See the [5. Fine-Tuning](#5-fine-tuning) section below for details.

## 4. Download Dataset (Optional)

We also open-sourced the USIM training dataset on Hugging Face. If you plan to fine-tune the model, you can download it with:

```sh
hf download Vincent2025hello/usim --local-dir $DATA_BASE_DIR/usim --local-dir-use-symlinks False
```

> **Tip**: The `demo_data/` directory in this repo contains a small sample dataset for quick testing. For full training, use the USIM dataset above.

---

## 5. Fine-Tuning

For a full list of options, run:

```sh
python scripts/gr00t_finetune.py --help
```

### 5.1 Fine-Tuning Demo

Run the fine-tuning demo on a single GPU server with the following configuration:

```sh
source .env
python scripts/gr00t_finetune.py \
    --dataset-path ./demo_data/robot_sim.PickNPlace \
    --output-dir $MODEL_BASE_DIR/finetuned-model-demo \
    --base-model-path $MODEL_BASE_DIR/GR00T-N1.5-3B \
    --num-gpus 1 \
    --max-steps 500 \
    --batch-size 64 \
    --lora-rank 64 \
    --lora_alpha 128 \
    --data-config fourier_gr1_arms_only \
    --report-to tensorboard \
    --target-loss-weight 0
```

> **Note**: When `lora-rank` is greater than 0, LoRA fine-tuning mode is enabled. A larger `lora-rank` value means more trainable parameters and potentially better performance.

### 5.2 LoRA Fine-Tuning on Multi-GPU

Create a training session:

```sh
tmux new -s my_training "source .env && source ~/miniconda3/bin/activate gr00t && python scripts/gr00t_finetune.py \
    --dataset-path $DATA_BASE_DIR/usim/train \
    --output-dir $MODEL_BASE_DIR/finetuned-model-lora \
    --base-model-path $MODEL_BASE_DIR/GR00T-N1.5-3B \
    --num-gpus 4 \
    --max-steps 10000 \
    --save-steps 5000 \
    --batch-size 32 \
    --lora_rank 64 \
    --lora_alpha 128 \
    --data-config u0_bot \
    --report-to tensorboard"
```

### 5.3 Full Fine-Tuning with Visual Tuning

Create a training session:

```sh
tmux new -s my_training "source .env && source ~/miniconda3/bin/activate gr00t && python scripts/gr00t_finetune.py \
    --dataset-path $DATA_BASE_DIR/usim/train \
    --output-dir $MODEL_BASE_DIR/finetuned-model-full \
    --base-model-path $MODEL_BASE_DIR/GR00T-N1.5-3B \
    --num-gpus 2 \
    --max-steps 22000 \
    --save-steps 11000 \
    --batch-size 32 \
    --tune-visual \
    --data-config u0_bot \
    --report-to tensorboard"
```

Reattach to the training session:

```sh
tmux attach -t my_training
```

View training logs:

```sh
tensorboard --logdir="$MODEL_BASE_DIR/"
```

---

## 6. Evaluation

For a full list of options, run:

```sh
python scripts/eval_policy.py --help
```

Evaluate on the entire test dataset (automatically iterates all trajectories and computes per-trajectory steps):

```sh
source .env
python scripts/eval_policy.py \
    --model-path $MODEL_BASE_DIR/u0_final \
    --dataset-path $DATA_BASE_DIR/usim/test \
    --data-config u0_bot \
    --embodiment-tag new_embodiment \
    --modality-keys joint_pos pwm \
    --video-backend torchvision_av \
    --save-csv-path results/eval_results_u0_final.csv
```

Evaluate specific trajectories with fixed steps:

```sh
source .env
python scripts/eval_policy.py \
    --model-path $MODEL_BASE_DIR/u0_final \
    --dataset-path $DATA_BASE_DIR/usim/test \
    --data-config u0_bot \
    --embodiment-tag new_embodiment \
    --modality-keys joint_pos pwm \
    --video-backend torchvision_av \
    --save-csv-path results/eval_results_u0_final.csv \
    --save-plot-path results/plots \
    --action_horizon 16 \
    --trajs 5
```

> **Note**: When `--steps` and `--trajs` are omitted, the script automatically evaluates all trajectories and uses `trajectory_length - action_horizon` as the step count for each trajectory. Use `--save-csv-path` to save per-trajectory results (traj_id, traj_length, eval_steps, action_mse, target_mse) to a CSV file.

---

## 7. Inference

### 7.1 Parallel Evaluation with Multi-GPU

For automated evaluation pipelines, please refer to the [fish-vla](https://gitee.com/vincent-gu/fish-vla) project. To run parallel evaluation across multiple GPUs, use the following script:

```sh
python scripts/launch_multi_gpu.py \
    --num-instances 5 \
    --base-port 8000 \
    --gpus 1,2,3 \
    --model-path $MODEL_BASE_DIR/u0_final/ \
    --data-config u0_bot \
    --embodiment-tag new_embodiment \
    --host 0.0.0.0
```

For a full list of options, run:

```sh
python scripts/launch_multi_gpu.py --help
```

Press `Ctrl+C` to stop all instances simultaneously.

### 7.2 Single Model Inference

First, start the ROS environment:

```sh
conda activate ros_env
cd $ROS_WS_DIR
source devel/setup.bash
rosrun bluerov2_control mapper_setup.py shallowfixed
roslaunch stonefish_bluerov2 bluerov2_eval.launch
```

For a full list of options, run:

```sh
python scripts/inference_service_u0.py --help
```

Start the inference service (HTTP mode):

```sh
conda activate gr00t
pip install uvicorn fastapi json-numpy requests
source .env
python scripts/inference_service_u0.py \
    --data-config u0_bot \
    --embodiment-tag new_embodiment \
    --device cuda:0 \
    --server --http-server --host 0.0.0.0 --port 8000 \
    --model-path $MODEL_BASE_DIR/u0_final/
```

Publish task description:

```sh
rostopic pub /task_description std_msgs/String "Pick up the red cylinder."
```

---

## Deployment

The `deployment_scripts/` directory contains TensorRT deployment tools inherited from the upstream project. These scripts are **experimental** and have not been fully verified in the U0 robot setup. Use at your own discretion. See [deployment_scripts/README.md](deployment_scripts/README.md) for details.

## Citation

If you use this work, please cite:

```bibtex
@misc{gu2025usimu0visionlanguageactiondataset,
      title={USIM and U0: A Vision-Language-Action Dataset and Model for General Underwater Robots}, 
      author={Junwen Gu and Zhiheng Wu and Pengxuan Si and Shuang Qiu and Yukai Feng and Luoyang Sun and Laien Luo and Lianyi Yu and Jian Wang and Zhengxing Wu},
      year={2025},
      eprint={2510.07869},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2510.07869}, 
}
```

## Acknowledgments

This project is built on top of [NVIDIA Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T). We thank the NVIDIA GEAR team for open-sourcing the GR00T model and framework.

## License

This project is licensed under the [Apache License 2.0](LICENSE).
