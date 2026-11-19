# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from abc import ABC
from abc import abstractmethod
from types import MappingProxyType as frozendict
from typing import Any

import torch
from anemoi.inference.checkpoint import Checkpoint

LOG = logging.getLogger(__name__)


class Perturbation(ABC):
    """Base Perturbation"""

    def __init__(
        self,
        checkpoint: str,
        patch_metadata: dict[str, Any] = {},
        **kwargs,
    ) -> None:
        """Initialize the Perturbation.

        Parameters
        ----------
        checkpoint : str
            Path to the checkpoint.
        patch_metadata : dict, optional
            Patch to apply to the metadata of the checkpoint.
        """
        # anemoi-inference 0.9.0 reads the metadata of the (single) dataset of the checkpoint
        self._metadata = Checkpoint(checkpoint, patch_metadata=patch_metadata)

    @property
    def number_of_grid_points(self) -> int:
        return self._metadata.number_of_grid_points

    @property
    def variable_to_output_tensor_index(self) -> frozendict:
        """Return the mapping between variable name and output tensor index."""
        return self._metadata.variable_to_output_tensor_index

    @property
    def input_tensor_index_to_variable(self) -> frozendict:
        """Return the mapping between input tensor index and variable name."""
        return self._metadata.input_tensor_index_to_variable

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (
            1,
            1,
            self.number_of_grid_points,
            len(self.variable_to_output_tensor_index),
        )

    @property
    def coords(self) -> torch.Tensor:
        # `.copy()`: the supporting arrays are read-only
        lats = torch.from_numpy(self._metadata.latitudes.copy())
        lons = torch.from_numpy(self._metadata.longitudes.copy())
        return torch.stack([lats, lons], dim=-1)

    @abstractmethod
    def create(self, *args, **kwargs) -> torch.Tensor:
        pass
