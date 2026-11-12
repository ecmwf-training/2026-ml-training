import logging
import datetime
import numpy as np

LOGGER = logging.getLogger(__name__)

def load_saved_state(file) -> dict:
    # Get the date from the filename: f"inputstate-{date.strftime('%Y%m%d_%H')}.npz"
    date_str = file.stem.split("-")[2]
    date = datetime.datetime.strptime(date_str, "%Y%m%d_%H")

    with np.load(file, allow_pickle=False) as data:
        fields = {k: data[k] for k in data.files}

    state = {"date": date, "fields": fields}
    return state


def save_state(state, outfile):
    np.savez(outfile, **state["fields"])

def find_nearest_point_index(lat_arr, lon_arr, lat0, lon0):
    """
    Works with 1D arrays of point coordinates (possibly from an unstructured grid).
    Returns the index of the point closest to (lat0, lon0) using great-circle distance.
    """
    # ensure arrays are 1D and same length
    lats = np.ravel(lat_arr)
    lons = np.ravel(lon_arr)
    lons = np.where(lons > 180, lons - 360, lons)

    # convert to radians
    lat_r = np.deg2rad(lats)
    lon_r = np.deg2rad(lons)
    lat0_r = np.deg2rad(lat0)
    lon0_r = np.deg2rad(lon0)

    # haversine formula for central angle (vectorized)
    dlat = lat_r - lat0_r
    dlon = lon_r - lon0_r
    a = np.sin(dlat / 2.0)**2 + np.cos(lat_r) * np.cos(lat0_r) * np.sin(dlon / 2.0)**2
    central_angle = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return int(np.argmin(central_angle))

# ---------------------------------------------------------------------------------------------------------
# Training notebooks
# ---------------------------------------------------------------------------------------------------------
import codecs
import os
import subprocess
import sys
from pathlib import Path

COURSE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = COURSE_DIR / "configs"


def load_config(config_name, overrides=()):
    """Build a training config the way anemoi-training does (Hydra) and resolve all ${...} references."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        config = compose(config_name=config_name, overrides=list(overrides))
    OmegaConf.resolve(config)
    return config


def config_diff(config_a, config_b, skip=("diagnostics",)):
    """List the (key, value in a, value in b) of all settings that differ between two configs."""
    from omegaconf import OmegaConf

    def flatten(node, prefix=""):
        if not isinstance(node, dict):
            return {prefix: node}
        flat = {}
        for key, value in node.items():
            flat.update(flatten(value, f"{prefix}.{key}" if prefix else str(key)))
        return flat

    a = flatten(OmegaConf.to_container(config_a))
    b = flatten(OmegaConf.to_container(config_b))
    keys = sorted(k for k in a.keys() | b.keys() if not k.startswith(tuple(skip)))
    return [(k, a.get(k, "-"), b.get(k, "-")) for k in keys if a.get(k, "-") != b.get(k, "-")]


def run_training(config_name, overrides=(), seed=42):
    """Run `anemoi-training train` from the course folder and stream its output into the notebook.

    Same as running this in a terminal (in the folder of this file):
        ANEMOI_BASE_SEED=42 anemoi-training train --config-path=$PWD/configs --config-name=<config_name> <overrides>
    """
    executable = Path(sys.executable).parent / "anemoi-training"  # the one of this notebook's environment
    command = [str(executable), "train", f"--config-path={CONFIG_DIR}", f"--config-name={config_name}", *overrides]
    print(" ".join(command[1:]), flush=True)
    env = {**os.environ, "ANEMOI_BASE_SEED": str(seed), "PYTHONWARNINGS": "ignore"}
    # Read raw bytes so that the progress bar's carriage returns update it in place instead of adding lines
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    with subprocess.Popen(command, cwd=COURSE_DIR, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as process:
        while chunk := process.stdout.read1(4096):
            print(decoder.decode(chunk), end="", flush=True)
    if process.returncode != 0:
        raise RuntimeError(f"Training failed (exit code {process.returncode}), see the output above.")
