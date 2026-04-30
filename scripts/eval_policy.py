# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import warnings
warnings.filterwarnings(
    "ignore",
    message="The video decoding and encoding capabilities of torchvision are deprecated"
)

from dataclasses import dataclass, field
from typing import List, Literal, Optional
from pathlib import Path

import numpy as np
import tyro

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING
from gr00t.eval.robot import RobotInferenceClient
from gr00t.experiment.data_config import load_data_config
from gr00t.model.policy import BasePolicy, Gr00tPolicy
from gr00t.utils.eval import calc_mse_for_single_trajectory

warnings.simplefilter("ignore", category=FutureWarning)

"""
Example command:

NOTE: provide --model_path to load up the model checkpoint in this script,
        else it will use the default host and port via RobotInferenceClient

python scripts/eval_policy.py --plot --model-path nvidia/GR00T-N1.5-3B

"""


@dataclass
class ArgsConfig:
    """Configuration for evaluating a policy."""

    host: str = "localhost"
    """Host to connect to."""

    port: int = 5555
    """Port to connect to."""

    plot: bool = False
    """Whether to plot the images."""

    modality_keys: List[str] = field(default_factory=lambda: ["right_arm", "left_arm"])
    """Modality keys to evaluate."""

    data_config: str = "fourier_gr1_arms_only"
    """
    Data config to use, e.g. so100, fourier_gr1_arms_only, unitree_g1, etc.
    Or a path to a custom data config file. e.g. "module:ClassName" format.
    See gr00t/experiment/data_config.py for more details.
    """

    steps: Optional[int] = None
    """Number of steps to evaluate. If None, automatically use (trajectory_length - action_horizon) for each trajectory."""

    trajs: Optional[int] = None
    """Number of trajectories to evaluate. If None, evaluate all trajectories in the dataset."""

    start_traj: int = 0
    """Start trajectory index to evaluate (0-based index into dataset.trajectory_ids)."""

    action_horizon: int = None
    """Action horizon to evaluate. If None, will use the data config's action horizon."""

    video_backend: Literal["decord", "torchvision_av"] = "decord"
    """Video backend to use for various codec options. h264: decord or av: torchvision_av"""

    dataset_path: str = "demo_data/robot_sim.PickNPlace/"
    """Path to the dataset."""

    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "gr1"
    """Embodiment tag to use."""

    model_path: str = None
    """Path to the model checkpoint."""

    denoising_steps: int = 4
    """Number of denoising steps to use."""

    save_plot_path: str = None
    """Path to save the plot."""

    save_csv_path: str = None
    """Path to save the per-trajectory evaluation results as a CSV file."""

    plot_state: bool = False
    """Whether to plot the state."""


def main(args: ArgsConfig):
    # Create save_plot_path directory if specified
    if args.save_plot_path is not None:
        Path(args.save_plot_path).mkdir(parents=True, exist_ok=True)

    # Create save_csv_path directory if specified
    if args.save_csv_path is not None:
        Path(args.save_csv_path).parent.mkdir(parents=True, exist_ok=True)

    data_config = load_data_config(args.data_config)

    # Set action_horizon from data config if not provided
    if args.action_horizon is None:
        args.action_horizon = len(data_config.action_indices)
        print(f"Using action_horizon={args.action_horizon} from data config '{args.data_config}'")

    if args.model_path is not None:
        import torch

        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()

        policy: BasePolicy = Gr00tPolicy(
            model_path=args.model_path,
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=args.embodiment_tag,
            denoising_steps=args.denoising_steps,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    else:
        policy: BasePolicy = RobotInferenceClient(host=args.host, port=args.port)

    # Get the supported modalities for the policy
    modality = policy.get_modality_config()
    print("Current modality config: \n", modality)

    # Create the dataset
    dataset = LeRobotSingleDataset(
        dataset_path=args.dataset_path,
        modality_configs=modality,
        video_backend=args.video_backend,
        video_backend_kwargs=None,
        transforms=None,  # We'll handle transforms separately through the policy
        embodiment_tag=args.embodiment_tag,
    )

    print(len(dataset))
    # Make a prediction
    obs = dataset[0]
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            print(k, v.shape)
        else:
            print(k, v)

    for k, v in dataset.get_step_data(0, 0).items():
        if isinstance(v, np.ndarray):
            print(k, v.shape)
        else:
            print(k, v)

    # Get actual trajectory IDs and lengths from the dataset
    all_traj_ids = dataset.trajectory_ids        # actual episode indices (may be non-sequential)
    all_traj_lengths = dataset.trajectory_lengths  # corresponding lengths
    total_trajs = len(all_traj_ids)

    print("Total trajectories:", total_trajs)
    print("Trajectory IDs:", all_traj_ids)
    print("Trajectory lengths:", all_traj_lengths)
    print("Running with modality keys:", args.modality_keys)

    # Determine the range of trajectories to evaluate
    start_idx = args.start_traj
    if args.trajs is None:
        end_idx = total_trajs
        print(f"Evaluating ALL {total_trajs} trajectories (from index {start_idx})")
    else:
        end_idx = min(start_idx + args.trajs, total_trajs)
        print(f"Evaluating {end_idx - start_idx} trajectories (index {start_idx} to {end_idx - 1})")

    all_mse = []
    all_mse_target = []
    all_traj_steps = []

    # Open CSV file for real-time writing (write header immediately)
    csv_file = None
    csv_writer = None
    if args.save_csv_path is not None:
        import csv
        csv_file = open(args.save_csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["traj_id", "traj_length", "eval_steps", "action_mse", "target_mse"])
        csv_file.flush()

    for idx in range(start_idx, end_idx):
        traj_id = int(all_traj_ids[idx])
        traj_length = int(all_traj_lengths[idx])

        # Auto-compute steps: trajectory_length - action_horizon to avoid padded tail data
        if args.steps is None:
            steps = max(traj_length - args.action_horizon, 0)
        else:
            steps = min(args.steps, max(traj_length - args.action_horizon, 0))

        if steps == 0:
            print(f"Skipping trajectory {traj_id} (length={traj_length}, too short for action_horizon={args.action_horizon})")
            continue

        print(f"[{idx - start_idx + 1}/{end_idx - start_idx}] Running trajectory id={traj_id}, length={traj_length}, steps={steps}")
        # Build per-trajectory plot path: {save_plot_path}/traj_{episode_idx}.png
        traj_plot_path = None
        if args.save_plot_path is not None:
            traj_plot_path = str(Path(args.save_plot_path) / f"traj_{idx}.png")
        mse, mse_target = calc_mse_for_single_trajectory(
            policy,
            dataset,
            traj_id,
            modality_keys=args.modality_keys,
            steps=steps,
            action_horizon=args.action_horizon,
            plot=args.plot,
            plot_state=args.plot_state,
            save_plot_path=traj_plot_path,
        )
        print(f"  Trajectory {traj_id} MSE: {mse:.6f}, MSE target: {mse_target:.6f}")
        all_mse.append(mse)
        all_mse_target.append(mse_target)
        all_traj_steps.append(steps)

        # Release memory to prevent OOM during long evaluation runs
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        # Write this trajectory's result to CSV immediately
        if csv_writer is not None:
            csv_writer.writerow([traj_id, traj_length, steps, f"{mse:.6f}", f"{mse_target:.6f}"])
            csv_file.flush()

    # Write summary and close CSV
    if csv_file is not None:
        csv_file.close()

    if len(all_mse) == 0:
        print("No trajectories were evaluated!")
    else:
        # Compute statistics
        all_traj_steps_arr = np.array(all_traj_steps, dtype=np.float64)
        all_mse_arr = np.array(all_mse)
        all_mse_target_arr = np.array(all_mse_target)

        simple_avg_mse = np.mean(all_mse_arr)
        simple_std_mse = np.std(all_mse_arr)
        simple_avg_mse_target = np.mean(all_mse_target_arr)
        simple_std_mse_target = np.std(all_mse_target_arr)

        weighted_mse = np.sum(all_mse_arr * all_traj_steps_arr) / np.sum(all_traj_steps_arr)
        weighted_mse_target = np.sum(all_mse_target_arr * all_traj_steps_arr) / np.sum(all_traj_steps_arr)

        # Print results
        print("\n========== Results ==========")
        print(f"Evaluated {len(all_mse)} trajectories")
        print(f"Action MSE    (simple avg): {simple_avg_mse:.6f} ± {simple_std_mse:.6f}")
        print(f"Target MSE    (simple avg): {simple_avg_mse_target:.6f} ± {simple_std_mse_target:.6f}")
        print(f"Action MSE  (weighted avg): {weighted_mse:.6f}")
        print(f"Target MSE  (weighted avg): {weighted_mse_target:.6f}")

        # Append summary rows to CSV
        if args.save_csv_path is not None:
            import csv
            with open(args.save_csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([])
                writer.writerow(["summary_type", "", "", "action_mse", "action_mse_std", "target_mse", "target_mse_std"])
                writer.writerow(["simple_avg", "", "", f"{simple_avg_mse:.6f}", f"{simple_std_mse:.6f}", f"{simple_avg_mse_target:.6f}", f"{simple_std_mse_target:.6f}"])
                writer.writerow(["weighted_avg", "", "", f"{weighted_mse:.6f}", "", f"{weighted_mse_target:.6f}", ""])
            print(f"Results saved to {args.save_csv_path}")

        print("Done")
    exit()


if __name__ == "__main__":
    # Parse arguments using tyro
    config = tyro.cli(ArgsConfig)
    main(config)
