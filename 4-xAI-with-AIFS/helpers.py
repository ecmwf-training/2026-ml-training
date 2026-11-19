import datetime
import logging
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

LOGGER = logging.getLogger(__name__)

R = 6371.0  # Earth radius in km
GRID_RESOLUTION = "O96"

# Variables of the `inference-aifs-crps-o96` checkpoint (see `anemoi-inference inspect <ckpt> --variables`)
PARAM_SFC = ["2t", "tcw", "lsm", "z", "slor", "sdor"]  # lsm, z, slor and sdor are constant in time
PARAM_PL = {  # parameter -> pressure levels
    "t": [850],
    "gh": [1000, 700, 500, 300, 250],
}

# The metadata of this checkpoint is inconsistent with the model it contains: it lists 13 input steps
# and an integer (instead of a list) as output index, whereas the model was trained with 2 input steps
# (t-6h and t) and predicts 1 step (t+6h). We patch the metadata when loading the checkpoint.
PATCH_METADATA = {
    "metadata_inference": {
        "data": {
            "timesteps": {
                "input_relative_date_indices": [0, 1],
                "output_relative_date_indices": [2],
            }
        }
    }
}


def load_ckpt(path: str = "inference-aifs-crps-o96.ckpt") -> str:
    """Return the path to the local checkpoint, checking that it exists."""
    ckpt_file = Path(path)
    if not ckpt_file.exists():
        raise FileNotFoundError(
            f"Checkpoint {ckpt_file} not found. Place the `inference-aifs-crps-o96.ckpt` file next to the notebook."
        )
    return str(ckpt_file)


def load_saved_state(file) -> dict:
    # Get the date from the filename: f"inputstate-{}-{date.strftime('%Y%m%d_%H')}.npz"
    date_str = file.stem.split("-")[2]
    date = datetime.datetime.strptime(date_str, "%Y%m%d_%H")

    with np.load(file, allow_pickle=False) as data:
        fields = {k: data[k] for k in data.files}

    state = {"date": date, "fields": fields}
    return state

def load_state(file):
    with np.load(file, allow_pickle=False) as data:
        fields = {k: data[k] for k in data.files}

    return {"fields": fields, "latitudes": fields.pop("latitudes"), "longitudes": fields.pop("longitudes")}

def save_state(state, outfile):
    np.savez(outfile, **state["fields"], longitudes = state["longitudes"], latitudes = state["latitudes"])

def open_data_client(source: str = "ecmwf"):
    """ECMWF Open Data client. `source="aws"` reads the archive of past dates."""
    # Imported here, as it is only needed to download data (loading packages from Lustre is slow)
    from ecmwf.opendata import Client

    # The 06 and 18 UTC runs moved from stream "scda" to "oper", but ecmwf-opendata 0.3.22 still requests
    # "scda" for them unless `infer_stream_keyword=False`
    return Client(source=source, model="ifs", preserve_request_order=True, infer_stream_keyword=False)


def latest_date() -> datetime.datetime:
    """Date of the latest forecast available on ECMWF Open Data."""
    return open_data_client().latest()


def get_open_data(param, date, resolution: str, levelist=[], source: str | None = None) -> dict:
    # Imported here, as they are only needed to download data (loading packages from Lustre is slow)
    import earthkit.data as ekd
    import earthkit.regrid as ekr

    # `source` is only needed for past dates (e.g. "aws"): the default server only keeps the latest days
    client = open_data_client(source or "ecmwf")
    fields = defaultdict(list)
    # Get the data for the current date and the previous date
    for date in [date - datetime.timedelta(hours=6), date]:
        # Without `levtype`, "z" would also match geopotential on pressure levels
        levtype = "pl" if levelist else "sfc"
        target = Path('.') / f"opendata-{date:%Y%m%d%H}-{levtype}-{'-'.join(param)}.grib2"
        client.retrieve(date=date, step=0, param=param, levtype=levtype, levelist=levelist, target=str(target))
        data = ekd.from_source("file", str(target))
        for f in data:
            # Open data is between -180 and 180, we need to shift it to 0-360
            assert f.to_numpy().shape == (721, 1440)
            values = np.roll(f.to_numpy(), -f.shape[1] // 2, axis=1)
            # Interpolate the data to from 0.25 to grid
            values = ekr.interpolate(values, {"grid": (0.25, 0.25)}, {"grid": resolution})
            # Add the values to the list
            name = f"{f.metadata('param')}_{f.metadata('levelist')}" if levelist else f.metadata("param")
            fields[name].append(values)

    # Create a single matrix for each parameter
    for param, values in fields.items():
        fields[param] = np.stack(values)

    return fields


def rename_keys(state: dict, mapping: dict) -> dict:
    for old_key, new_key in mapping.items():
        if old_key in state:
            state[new_key] = state.pop(old_key)
        else:
            LOGGER.warning(f"Key {old_key} not found in state, cannot rename to {new_key}")

    return state


def transform_GH_to_Z(fields: dict, levels: list[str]) -> dict:
    for level in levels:
        fields[f"z_{level}"] = fields.pop(f"gh_{level}") * 9.80665

    return fields


def load_opendata_state(resolution: str, date: str, source: str | None = None) -> dict:
    fields = {}
    fields.update(get_open_data(param=PARAM_SFC, date=date, resolution=resolution, source=source))
    for param, levels in PARAM_PL.items():
        fields.update(get_open_data(param=[param], date=date, resolution=resolution, levelist=levels, source=source))

    fields = transform_GH_to_Z(fields, PARAM_PL["gh"])

    return dict(date=date, fields=fields)


def save_state(state, outfile):
    np.savez(outfile, **state["fields"])


def load_input_state(date, resolution: str = GRID_RESOLUTION, source: str | None = None) -> dict:
    """Load the input state (2 time steps: date-6h and date) from ECMWF open data.

    The state is saved to a local file the first time, and reloaded from it afterwards.
    For dates older than a few days, use `source="aws"` to read the open data archive.
    """
    file = Path(f"inputstate-{resolution}-{date.strftime('%Y%m%d_%H')}.npz")
    if file.exists():
        input_state = load_saved_state(file)
        LOGGER.info(f"Input state loaded from file {file}")
    else:
        input_state = load_opendata_state(resolution=resolution, date=date, source=source)
        LOGGER.info(f"State created for date {date}")
        save_state(input_state, file)
        LOGGER.info(f"State saved to file {file}")

    return input_state


def haversine(coords: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    """Compute haversine distance between multiple coordinates and a single point.

    Args:
        coords: (N, 2) tensor of [lat, lon] in degrees
        point: (2,) tensor of [lat, lon] in degrees

    Returns:
        (N,) tensor of distances in kilometers
    """
    coords_rad = torch.deg2rad(coords)
    point_rad = torch.deg2rad(point)

    lat1, lon1 = coords_rad[:, 0], coords_rad[:, 1]
    lat2, lon2 = point_rad[0], point_rad[1]

    dlat, dlon = lat2 - lat1, lon2 - lon1

    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

    return R * c


def compute_sensitivities_statistics(sensitivities) -> pd.DataFrame:
    df = pd.DataFrame()
    for i, (varname, array) in enumerate(sensitivities["fields"].items()):
        var_type = "pl" if "_" in varname and varname.split("_")[1].isdigit() else "sfc"
        pl = varname.split("_")[-1] if var_type == "pl" else None
        name = varname.split("_")[0] if var_type == "pl" else varname
        df[2 * i] = (
            varname,
            var_type,
            name,
            pl,
            -6,
            np.sum(array[0] != 0) / array[0].size * 100,
            np.min(array[0]),
            np.max(array[0]),
        )
        df[2 * i + 1] = (
            varname,
            var_type,
            name,
            pl,
            0,
            np.sum(array[1] != 0) / array[1].size * 100,
            np.min(array[1]),
            np.max(array[1]),
        )
    df = df.T
    df.columns = ["var", "type", "varname", "pl", "hour", "num_nonzero", "min", "max"]
    df = df.astype({"num_nonzero": "float32", "min": "float32", "max": "float32"})
    df = df.set_index(["var", "type", "varname", "pl", "hour"])
    df = df.round({"num_nonzero": 2}).reset_index()
    return df
