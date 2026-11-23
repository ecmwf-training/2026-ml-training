# 2026-ml-training
This repo contains code examples and resources for ECMWF's [Machine Learning-based Weather Forecasting]([url](https://events.ecmwf.int/event/546/)) training course running from 16 to 20 November 2026. 

To set up the Python environment (on your machine with GPU access), see [SETUP.md](SETUP.md).

## A note on Datasets

The notebooks in this repo use a variety of datasets, including
```
aifs-ea-an-oper-0001-mars-o48-2010-2022-6h-v1-toy.zarr
aifs-ea-an-oper-0001-mars-o96-6h-v8-profiling-toy.zarr
```

These datasets are not publicly available. If you are trying to follow the notebooks on a system without these datasets, ECMWF provides an open dataset of ERA5
reanalysis data on an O96 reduced Gaussian grid. You can find more information, including how to download the dataset [here](https://anemoi.readthedocs.io/projects/training/en/latest/user-guide/download-era5-o96.html). 
