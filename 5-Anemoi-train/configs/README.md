# Training configs of the course

anemoi-training 0.9.0 configs for the two training sessions, on the O48 toy dataset and one GPU.

| File | Session | Notebook |
|---|---|---|
| `deterministic_o48.yaml` | 1: deterministic model, MSE loss | `../training_aifs-deterministic.ipynb` |
| `crps_o48.yaml` | 2: ensemble model, (multi-scale) CRPS loss | `../training_aifs-ens.ipynb` |

All other files are the default configs of anemoi-training 0.9.0 (`anemoi-training config generate`), plus
`data/toy.yaml`, `dataloader/toy_o48.yaml` and `system/course.yaml` for the course. `crps_o48.yaml` is the
course version of the pretraining stage of the O48 AIFS-CRPS model used in the inference session
(anemoi-configs, branch `training-course-anemoi-training-0.9`, `training-course-crps-o48-6h-pretrain.yaml`).

## Training in a terminal

From the `5-Anemoi-train` folder (the multi-scale loss reads its matrices from `loss_matrices/` relative to it):

```bash
export ANEMOI_BASE_SEED=42
anemoi-training train --config-path=$PWD/configs --config-name=deterministic_o48
```

Override any setting after the config name, e.g. `training.max_steps=200 diagnostics.log.mlflow.run_name=short`.
Checkpoints, plots and MLflow logs go to `~/anemoi-output/` (`system.output.root`).

## MLflow

```bash
mlflow ui --backend-store-uri=$HOME/anemoi-output/logs/mlflow --port=5111
```

and open `https://<course server>/user/<your user>/proxy/5111/`.
