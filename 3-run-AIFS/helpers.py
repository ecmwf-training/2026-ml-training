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


def area_weights(latitudes):
    """Normalised area weight of each point of a reduced Gaussian grid (e.g. the octahedral O48/O96 grids).

    The rows of constant latitude are (almost) equally spaced, so the area of a point is proportional to
    cos(latitude) divided by the number of points in its row. A plain cos(latitude) would be wrong here,
    because rows near the poles have far fewer points than rows near the equator.
    """
    lats = np.asarray(latitudes)
    _, row, points_per_row = np.unique(lats, return_inverse=True, return_counts=True)
    weights = np.cos(np.deg2rad(lats)) / points_per_row[row]
    return weights / weights.sum()


def patch_ens_single_device_inference():
    """Backport of the anemoi-models 0.12.1 fix for running ensemble models on a single device.

    anemoi-models 0.12.0 indexes `grid_shard_shapes[dataset_name]` even when it is None (i.e. when the
    model is not split across GPUs), so AIFS-ENS checkpoints crash during inference on one CPU/GPU.
    Remove this once the course environment uses anemoi-models >= 0.12.1.
    """
    from anemoi.models.models.ens_encoder_processor_decoder import AnemoiEnsModelEncProcDec

    if getattr(AnemoiEnsModelEncProcDec, "_single_device_patch", False):
        return
    original = AnemoiEnsModelEncProcDec._assemble_input

    def _assemble_input(self, x, fcstep, batch_ens_size, grid_shard_shapes=None, model_comm_group=None, dataset_name=None):
        if grid_shard_shapes is None:
            grid_shard_shapes = {dataset_name: None}
        return original(self, x, fcstep, batch_ens_size, grid_shard_shapes, model_comm_group, dataset_name)

    AnemoiEnsModelEncProcDec._assemble_input = _assemble_input
    AnemoiEnsModelEncProcDec._single_device_patch = True


def patch_graph_attention_backend_for_device():
    """Let checkpoints trained on GPU with the triton graph-attention backend run on CPU.

    The backend of the graph-transformer layers is chosen when the model is built (triton when a GPU is
    available) and stored in the checkpoint, and triton kernels only run on GPUs. This switches a layer
    to the equivalent pure-PyTorch ("pyg") implementation the first time it sees CPU tensors.
    """
    from anemoi.models.layers.block import GraphTransformerBaseBlock
    from anemoi.models.layers.conv import GraphTransformerConv

    if getattr(GraphTransformerBaseBlock, "_device_backend_patch", False):
        return
    original = GraphTransformerBaseBlock.apply_gt

    def apply_gt(self, query, key, value, edges, edge_index, size):
        if self.graph_attention_backend == "triton" and not query.is_cuda:
            self.graph_attention_backend = "pyg"
            self.conv = GraphTransformerConv(out_channels=self.out_channels_conv)
        return original(self, query, key, value, edges, edge_index, size)

    GraphTransformerBaseBlock.apply_gt = apply_gt
    GraphTransformerBaseBlock._device_backend_patch = True