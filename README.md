# SciEye: A System for Finding the Underlying Datasets for Scientific Figures

Authors: Ziyue "Alan" Xiang, Edward J. Delp

Direct all correspondence to Edward J. Delp, [ace@ecn.purdue.edu](mailto:ace@ecn.purdue.edu).

## Installation and Configuration

- OS: Ubuntu 20.04

- Set up `conda` environment

  - `conda create -n data-graph-matching`

  - `conda env update -n data-graph-matching --file environment.yaml`

- Acquire the JSON service key from Google Cloud Platform and save it in `/notebook` as `api_key.json` ([tutorial](https://cloud.google.com/vision/docs/before-you-begin))

- Download model checkpoints from <https://darknet.ecn.purdue.edu/~xiang71/scieye/scieye_ckpt_v01.zip> and extract them to `/ckpt`

## Demo

Please see `/notebook/demo.ipynb`

## Parameters

The parameters of many steps are defined in `/data_graph_matching/param.py`.

- `parallel_n_jobs` controls the number of parallel jobs; setting it to 1 can be beneficial for debugging
