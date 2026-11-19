"""Helper functions for the notebook perf_profiling_short.ipynb.

You do not have to read this file to do the exercises.
The functions start anemoi runs from the notebook and read the files that the profiler writes.
The functions that start a run are the same as in ../5-Anemoi-train/helpers.py,
but they use the configs/ folder of this directory.
"""

import codecs
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

# The functions that read a memory snapshot are in perf_helpers.py (the long profiling notebook).
from perf_helpers import load_snapshot, memory_timeline, peak_memory_gb, plot_memory_timelines  # noqa: F401

WORKSHOP_DIR = Path(__file__).resolve().parent
CONFIG_DIR = WORKSHOP_DIR / "configs"
CONFIG_NAME = "profiling_o48"
OUTPUT_ROOT = Path.home() / "anemoi-output" / "profiling"


def load_config(config_name=CONFIG_NAME, overrides=()):
    """Make a config as anemoi-training does (with Hydra), and resolve all ${...} references."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        config = compose(config_name=config_name, overrides=list(overrides))
    OmegaConf.resolve(config)
    return config


def run_anemoi(command, overrides=(), config_name=CONFIG_NAME, seed=42, cwd=WORKSHOP_DIR):
    """Run `anemoi-training <command>` from this folder and show its output in the notebook.

    This is the same as this command in a terminal (in the folder of this file):
        ANEMOI_BASE_SEED=42 anemoi-training <command> --config-path=$PWD/configs --config-name=<config_name> <overrides>

    Parameters
    ----------
    command : str
        "train" or "profile".
    overrides : list[str]
        Hydra overrides, for example ["model.encoder.num_chunks=16"].
    cwd : Path
        The folder in which the run starts.
    """
    executable = Path(sys.executable).parent / "anemoi-training"  # the one of the environment of this notebook
    cmd = [str(executable), command, f"--config-path={CONFIG_DIR}", f"--config-name={config_name}", *overrides]
    print(" ".join(cmd[1:]), flush=True)
    env = {**os.environ, "ANEMOI_BASE_SEED": str(seed), "PYTHONWARNINGS": "ignore"}
    # Read raw bytes. Then the carriage returns of the progress bar update it in place, and do not add lines.
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    with subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as process:
        while chunk := process.stdout.read1(4096):
            print(decoder.decode(chunk), end="", flush=True)
    if process.returncode != 0:
        raise RuntimeError(f"The run failed (exit code {process.returncode}). Refer to the output above.")


def run_experiment(name, overrides=(), command="profile"):
    """Run anemoi with the output in OUTPUT_ROOT/<name>, and return the folder of the profiler files.

    Each run writes its profiler files into OUTPUT_ROOT/<name>/profiler/<run id>/.
    This function returns the newest of these folders.

    The run starts in OUTPUT_ROOT/<name>. Thus, the files that the run writes into its start folder
    (the Hydra folder outputs/ and report.html) do not go into the folder of the notebook.
    PyTorch Lightning also writes a checkpoint into <start folder>/checkpoints/ (anemoi-training 0.9.0 does not
    switch this off). This function deletes it after the run, because it uses much disk space.
    """
    root = OUTPUT_ROOT / name
    root.mkdir(parents=True, exist_ok=True)
    try:
        run_anemoi(command, [*overrides, f"system.output.root={root}/"], cwd=root)
    finally:
        shutil.rmtree(root / "checkpoints", ignore_errors=True)
    if command == "profile":
        return find_profiler_dir(name)
    return None


def find_profiler_dir(name):
    """Return the newest profiler folder of an experiment, or None."""
    folders = [p.parent for p in (OUTPUT_ROOT / name).rglob("speed_profiler.csv")]
    return max(folders, key=lambda p: p.stat().st_mtime) if folders else None


def find_trace_file(profiler_dir):
    """Return the PyTorch profiler trace of the training steps in a profiler folder.

    The profiler can write more than one ``*.pt.trace.json`` file (one for training, one for validation).
    The training trace is the first file, thus we select the oldest file.
    """
    traces = sorted(Path(profiler_dir).glob("*.pt.trace.json"), key=lambda p: p.stat().st_mtime)
    return traces[0] if traces else None


def training_throughput(profiler_dir):
    """Return the average training throughput (iterations for each second) from the speed report."""
    df = pd.read_csv(Path(profiler_dir) / "speed_profiler.csv", index_col=0)
    return float(df.set_index("metric").loc["training_avg_throughput", "value"])


def dataloader_throughput(profiler_dir):
    """Return the average throughput of the training data loader (batches for each second) from the speed report."""
    df = pd.read_csv(Path(profiler_dir) / "speed_profiler.csv", index_col=0)
    return float(df.set_index("metric").loc["avg_training_dataloader_throughput", "value"])


def summary_table(experiments, memory=True):
    """Make a table with the throughput (and the peak memory) of each experiment."""
    rows = []
    for name, profiler_dir in experiments.items():
        row = {"experiment": name, "throughput [it/s]": round(training_throughput(profiler_dir), 2)}
        if memory:
            row["peak memory [GB]"] = round(peak_memory_gb(profiler_dir), 2)
        rows.append(row)
    return pd.DataFrame(rows).set_index("experiment")
