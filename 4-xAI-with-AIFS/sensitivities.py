# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import logging
from collections.abc import Generator
from typing import Any
from typing import Callable

import numpy as np
import torch
from anemoi.inference.runners.simple import SimpleRunner
from anemoi.inference.types import FloatArray
from anemoi.inference.types import State
from anemoi.utils.dates import frequency_to_timedelta as to_timedelta
from anemoi.utils.timer import Timer

LOG = logging.getLogger(__name__)


class SensitivitiesRunner(SimpleRunner):
    """Sensitivities runner.

    Computes the backward sensitivities (vector-Jacobian product) of a perturbation of the output of
    the model (one forward step) with respect to the input of the model. Only single-dataset
    checkpoints and a lead time of one model step are supported.
    """

    def __init__(
        self,
        checkpoint: str,
        *,
        return_unnormalised_sensitivities: bool = True,
        perturb_normalised_space: bool = False,
        seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the SensitivitiesRunner.

        Parameters
        ----------
        checkpoint : str
            Path to the checkpoint.
        return_unnormalised_sensitivities : bool
            Return the sensitivities with respect to the unnormalised inputs.
        perturb_normalised_space : bool
            Apply the output perturbation in normalised space.
        seed : int, optional
            Seed for the random number generator. Ensemble models (e.g. AIFS-CRPS) inject noise in the
            processor, so the sensitivities are those of one ensemble member. Fixing the seed makes them
            reproducible.
        **kwargs : dict
            Keyword arguments passed to the `SimpleRunner` (e.g. `device`, `precision`, `patch_metadata`).
        """
        super().__init__(checkpoint, **kwargs)
        # anemoi-inference 0.9.0 runs a single dataset, but models from anemoi-models >= 0.12 take and
        # return a dictionary {dataset_name: tensor}, which `wrap_model` relies on.
        assert self.checkpoint._metadata.multi_dataset, "Only checkpoints of anemoi-models >= 0.12 are supported."
        self.return_unnormalised_sensitivities = return_unnormalised_sensitivities
        self.perturb_normalised_space = perturb_normalised_space
        self.noise_seed = seed
        self.perturbation = None
        # Forecast of the last model step, with shape (batch, ensemble, values, variables).
        # It is in physical units if `perturb_normalised_space=False`, otherwise normalised.
        self.last_prediction = None

    @property
    def dataset_name(self) -> str:
        """Name of the (single) dataset of the checkpoint."""
        return self.checkpoint.name

    def _allow_unsharded_input(self, model: torch.nn.Module) -> None:
        """Work around a bug in anemoi-models 0.12.0 for ensemble models.

        `AnemoiEnsModelEncProcDec._assemble_input` indexes `grid_shard_shapes[dataset_name]`
        even when the model is not sharded over several GPUs (`grid_shard_shapes=None`), which raises
        a `TypeError`. Here we treat `None` as "no sharding" for the dataset.
        """
        inner = model.model
        if getattr(inner, "_unsharded_input_patched", False):
            return

        original = inner._assemble_input

        def _assemble_input(x, fcstep, batch_ens_size, grid_shard_shapes=None, **kwargs):
            if grid_shard_shapes is None:
                grid_shard_shapes = {kwargs["dataset_name"]: None}
            return original(
                x, fcstep=fcstep, batch_ens_size=batch_ens_size, grid_shard_shapes=grid_shard_shapes, **kwargs
            )

        inner._assemble_input = _assemble_input
        inner._unsharded_input_patched = True

    def wrap_model(self, model: torch.nn.Module) -> Callable:
        """Wrap the model to be used for sensitivities."""
        name = self.dataset_name
        self._allow_unsharded_input(model)

        def model_wrapper(x: torch.Tensor) -> torch.Tensor:
            x = x[:, :, None, ...]  # add dummy ensemble dimension as 3rd index
            if self.return_unnormalised_sensitivities:
                x = model.pre_processors[name](x, in_place=False)
            # Multi-dataset models take and return a dictionary {dataset_name: tensor}
            y_hat = model.model({name: x}, fcstep=0)[name]
            if not self.perturb_normalised_space:
                y_hat = model.post_processors[name](y_hat, in_place=False)
            return y_hat

        return model_wrapper

    def perturb_prediction_linearly(
        self, output: torch.Tensor, idx: int, perturbation_perc: float = 0.01
    ) -> torch.Tensor:
        """Perturb the output."""
        # Use a perturbation of 1% of the forecasted value
        pert = torch.zeros_like(output.clone())
        pert[..., idx] = perturbation_perc * output[..., idx]

        return pert

    def sensitivities_step(
        self, model: torch.nn.Module, input_tensor_torch: torch.Tensor, perturbation: torch.Tensor
    ) -> torch.Tensor:
        """Compute the sensitivities of one model step.

        Parameters
        ----------
        model : torch.nn.Module
            The model.
        input_tensor_torch : torch.Tensor
            The input tensor, with shape (batch, time, values, variables).
        perturbation : torch.Tensor
            The perturbation of the output, with shape (batch, ensemble, values, variables).

        Returns
        -------
        torch.Tensor
            The sensitivities, with shape (time, values, variables).
        """
        model_func = self.wrap_model(model)

        # Prepare input tensor
        if not self.return_unnormalised_sensitivities:
            with torch.no_grad():
                input_tensor_torch = model.pre_processors[self.dataset_name](input_tensor_torch, in_place=False)

        input_tensor_torch.requires_grad_(True)

        def vjp() -> torch.Tensor:
            if self.noise_seed is not None:
                torch.manual_seed(self.noise_seed)

            with torch.enable_grad():
                with torch.autocast(device_type=self.device.type, dtype=self.autocast):
                    y_pred, t_dx_output = torch.autograd.functional.vjp(
                        model_func,
                        input_tensor_torch,
                        v=perturbation,
                        create_graph=False,
                        strict=False,
                    )
            self.last_prediction = y_pred.detach().float().cpu()
            return t_dx_output

        # This is needed to avoid issues with activation checkpointing.
        # The first time the function is called, you may get a checkpointing error.
        try:
            t_dx_output = vjp()
        except torch.utils.checkpoint.CheckpointError:
            LOG.warning("Checkpointing error occurred.")
            t_dx_output = vjp()

        return t_dx_output[0, ...]  # (time, values, variables)

    def forecast(
        self,
        lead_time: str,
        input_tensor_numpy: FloatArray,
        input_state: State,
    ) -> Generator[State, None, None]:
        """Compute the sensitivities.

        Parameters
        ----------
        lead_time : str
            The lead time. Only one model step is supported.
        input_tensor_numpy : FloatArray
            The input tensor, with shape (multi_step_input, variables, values).
        input_state : State
            The input state.

        Yields
        ------
        State
            The state whose fields are the sensitivities (time, values) with respect to each input variable.
        """
        start = input_state["date"]

        lead_time = to_timedelta(lead_time)
        n_steps = len(list(self.forecast_stepper(start, lead_time)))
        if n_steps != 1:
            raise ValueError(
                f"Sensitivities are only supported for one model step ({self.checkpoint.timestep}), "
                f"got lead_time={lead_time}."
            )

        self.model.eval()

        # Create pytorch input tensor: (batch, time, values, variables)
        input_tensor_torch = torch.from_numpy(np.swapaxes(input_tensor_numpy, -2, -1)[np.newaxis, ...]).to(
            self.device
        )

        if self.perturbation is None:
            raise ValueError("A perturbation must be provided, use `SensitivitiesRunner.run`.")
        output_perturbation = self.perturbation.create(self.model).to(self.device)

        new_state = input_state.copy()  # We should not modify the input state
        new_state["fields"] = dict()
        new_state["step"] = to_timedelta(0)

        for s, (step, date, _, _) in enumerate(self.forecast_stepper(start, lead_time)):
            with Timer(f"Computing sensitivities for {date}"):
                y_pred = self.sensitivities_step(self.model, input_tensor_torch, perturbation=output_perturbation)

            new_state["date"] = date
            new_state["previous_step"] = new_state["step"]
            new_state["step"] = step

            # Update state
            for i in range(y_pred.shape[-1]):
                new_state["fields"][self.checkpoint.input_tensor_index_to_variable[i]] = y_pred[:, :, i]

            if self.verbosity > 0:
                self._print_input_tensor("Sensitivities tensor", y_pred[None])

            yield new_state

    def run(
        self,
        *,
        input_state: State,
        perturbation: "Perturbation",
        lead_time: str | int | datetime.timedelta,
        return_numpy: bool = True,
    ) -> Generator[State, None, None]:
        """Run the model.

        Parameters
        ----------
        input_state : State
            The input state.
        perturbation : Perturbation
            The output perturbation
        lead_time : Union[str, int, datetime.timedelta]
            The lead time.
        return_numpy : bool, optional
            Whether to return the output state fields as numpy arrays, by default True.
            Otherwise, it will return torch tensors.

        Returns
        -------
        Generator[State, None, None]
            The state with the sensitivities.
        """
        self.perturbation = perturbation
        try:
            yield from super().run(input_state=input_state, lead_time=lead_time, return_numpy=return_numpy)
        finally:
            self.perturbation = None
