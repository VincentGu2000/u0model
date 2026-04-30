#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
One-click launch of multiple inference service instances, each bound to a different GPU and port.
"""

import argparse
import datetime
import os
import signal
import subprocess
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description="One-click launch of multiple GR00T inference service instances (multi-GPU / multi-port)"
    )
    parser.add_argument(
        "--num-instances",
        type=int,
        default=4,
        help="Number of instances to launch (default: 4)",
    )
    parser.add_argument(
        "--base-port",
        type=int,
        default=8000,
        help="Port number for the first instance, subsequent instances increment (default: 8000)",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help=(
            "Comma-separated list of GPU IDs, e.g. '0,1,2,3'. "
            "If fewer GPUs are specified than num-instances, they will be used cyclically. "
            "If not specified, defaults to 0,1,...,num-instances-1"
        ),
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Model path",
    )
    parser.add_argument(
        "--data-config",
        type=str,
        default="u0_bot",
        help="Data configuration name",
    )
    parser.add_argument(
        "--embodiment-tag",
        type=str,
        default="new_embodiment",
        help="Embodiment tag",
    )
    parser.add_argument(
        "--denoising-steps",
        type=int,
        default=4,
        help="Denoising steps",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Service listen address",
    )
    parser.add_argument(
        "--http-server",
        action="store_true",
        default=True,
        help="Use HTTP server mode (enabled by default)",
    )
    parser.add_argument(
        "--script-path",
        type=str,
        default="scripts/inference_service_u0.py",
        help="Inference service script path (default: scripts/inference_service_u0.py)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Parse GPU list
    if args.gpus is not None:
        gpu_list = [int(g.strip()) for g in args.gpus.split(",")]
    else:
        gpu_list = list(range(args.num_instances))

    num_instances = args.num_instances

    if len(gpu_list) < num_instances:
        print(
            f"[WARNING] Number of GPUs ({len(gpu_list)}) is less than number of instances ({num_instances}), "
            f"GPUs will be used cyclically: {gpu_list}"
        )

    # Create log directory
    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build command for each instance
    processes = []
    log_files = []  # Save log file handles, close on exit
    for i in range(num_instances):
        gpu_id = gpu_list[i % len(gpu_list)]
        port = args.base_port + i

        cmd = [
            sys.executable,
            args.script_path,
            "--data-config", args.data_config,
            "--embodiment-tag", args.embodiment_tag,
            "--device", f"cuda:{gpu_id}",
            "--port", str(port),
            "--host", args.host,
            "--model-path", args.model_path,
            "--denoising-steps", str(args.denoising_steps),
            "--server",
        ]
        if args.http_server:
            cmd.append("--http-server")

        print(f"[Instance {i}] Launching: GPU=cuda:{gpu_id}, Port={port}, PID=...")
        print(f"         Command: {' '.join(cmd)}")

        # Create a separate log file for each instance (avoid subprocess deadlock from full PIPE buffer)
        log_path = os.path.join(log_dir, f"inference_{timestamp}_inst{i}_gpu{gpu_id}_port{port}.log")
        log_file = open(log_path, "w", buffering=1)  # Line buffering for real-time viewing
        log_files.append(log_file)

        env = os.environ.copy()
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout, write to log file uniformly
        )
        processes.append((i, gpu_id, port, proc, log_path))
        print(f"         PID={proc.pid}")
        print(f"         Log: {log_path}")

    print(f"\n{'='*60}")
    print(f"Launched {num_instances} inference service instances:")
    print(f"{'='*60}")
    for i, gpu_id, port, proc, log_path in processes:
        print(f"  Instance {i}: GPU=cuda:{gpu_id}, Port={port}, PID={proc.pid}")
        print(f"           Log: {log_path}")
    print(f"{'='*60}")
    print(f"Press Ctrl+C to stop all instances\n")

    # Register signal handler for graceful shutdown
    def shutdown(signum, frame):
        print(f"\nReceived signal {signum}, stopping all instances...")
        for i, gpu_id, port, proc, log_path in processes:
            if proc.poll() is None:
                print(f"  Stopping instance {i} (PID={proc.pid})...")
                proc.terminate()
        # Wait for processes to exit
        for i, gpu_id, port, proc, log_path in processes:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print(f"  Instance {i} (PID={proc.pid}) did not respond to terminate, force killing...")
                proc.kill()
        # Close log file handles
        for lf in log_files:
            try:
                lf.close()
            except Exception:
                pass
        print("All instances have been stopped.")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Monitor process status
    try:
        while True:
            all_dead = True
            for i, gpu_id, port, proc, log_path in processes:
                ret = proc.poll()
                if ret is None:
                    all_dead = False
                else:
                    print(f"[WARNING] Instance {i} (GPU=cuda:{gpu_id}, Port={port}, PID={proc.pid}) "
                          f"has exited, return code: {ret}")
                    print(f"         Check log: {log_path}")

            if all_dead:
                print("All instances have exited.")
                break

            time.sleep(5)
    except KeyboardInterrupt:
        shutdown(signal.SIGINT, None)


if __name__ == "__main__":
    main()
