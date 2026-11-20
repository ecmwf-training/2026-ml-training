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


# A launcher for `anemoi-training`. It is the same as the `anemoi-training` command, with two fixes.
#
# Fix 1 (data loader with 0 workers):
# With num_workers=0, the PyTorch DataLoader stops with a ValueError if prefetch_factor is not None,
# or if persistent_workers is True. The anemoi config schema accepts only an integer for prefetch_factor,
# thus you cannot set it to null in the config.
# Also, anemoi prepares its dataset in worker_init_fn (per_worker_init() sets self.rng and the sample range).
# With num_workers=0, no worker process starts, thus worker_init_fn does not run,
# and the data loader stops with "'MultiDataset' object has no attribute 'rng'".
# The launcher changes the DataLoader. If num_workers is 0, it sets prefetch_factor=None and persistent_workers=False,
# and it calls per_worker_init(n_workers=1, worker_id=0) on the dataset one time, in the main process.
# If num_workers is more than 0, the DataLoader does not change.
#
# Fix 2 (empty memory report, `profile` only):
# If the memory profiler records no events, get_memory_profiler_df() returns "" and not a DataFrame.
# Then the report stops the run with "'str' object has no attribute 'to_markdown'".
# This occurs after the training and after the profiler writes its files.
# The launcher replaces get_memory_profiler_df() with a version that always returns a DataFrame.
#
# The run occurs in a different process, thus we must do the replacements in that process.
# We do not change the anemoi-training code.
_PROFILE_LAUNCHER = """
import inspect
import random
import sys
import pandas as pd
import torch
import torch.utils.data

_dataloader_init = torch.utils.data.DataLoader.__init__
_dataloader_signature = inspect.signature(_dataloader_init)

def dataloader_init(self, *args, **kwargs):
    bound = _dataloader_signature.bind(self, *args, **kwargs)
    if bound.arguments.get("num_workers", 0) == 0:
        bound.arguments["prefetch_factor"] = None
        bound.arguments["persistent_workers"] = False
        # No worker process starts, thus worker_init_fn does not run. Do its work here, in this process.
        # per_worker_init() sets the random seeds. Keep the random state of this process as it was.
        dataset = bound.arguments["dataset"]
        if hasattr(dataset, "per_worker_init"):
            python_state = random.getstate()
            with torch.random.fork_rng(devices=[]):
                dataset.per_worker_init(n_workers=1, worker_id=0)
            random.setstate(python_state)
    _dataloader_init(*bound.args, **bound.kwargs)

torch.utils.data.DataLoader.__init__ = dataloader_init

from anemoi.training.diagnostics.profilers import BenchmarkProfiler

_get_memory_profiler_df = BenchmarkProfiler.get_memory_profiler_df

def get_memory_profiler_df(self):
    memory_df = _get_memory_profiler_df(self)
    if isinstance(memory_df, pd.DataFrame):
        return memory_df
    print("Memory Profiling: the profiler recorded no events. The memory report is empty.", flush=True)
    return pd.DataFrame()

BenchmarkProfiler.get_memory_profiler_df = get_memory_profiler_df

from anemoi.training.__main__ import main
sys.argv = ["anemoi-training", *sys.argv[1:]]
main()
"""


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
    # Use the launcher above. It is the same command, but num_workers=0 is permitted,
    # and an empty memory report does not stop the run.
    cmd = [sys.executable, "-c", _PROFILE_LAUNCHER, *cmd[1:]]
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
