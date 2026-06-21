# Machine-Learning

This repository contains a final project for the Green Data Science Master's Module on Machine Learning. 

The goal of this project is to detect vegetation changes from Sentinal-2 data in Portugal. 

The project applies a deep learning model developed by [reference], developed to detect fires from NDVI time series data.

In this project, the initial model is fine tuned to predict vegetation changes from both fire and agriculture. 

## Running the interactive app locally

`app.py` is a local web app for exploring the burned-area predictions. You pick a
model, a before and after scene pair, the window overlap, and the voting
strictness, and see the result on an interactive map.

### Requirements

- The `veg-s2s` conda environment (see `environment.yml`), which now includes
  `streamlit` and `folium`. Create or update it with:
  ```bash
  conda env update -f environment.yml      # or: conda env create -f environment.yml
  conda activate veg-s2s
  ```
- The prediction rasters in `outputs/predictions/` (produced by the inference
  pipeline, see `inference/README.md`). The bundled runs, EfficientNet-B2 and
  Swin-YNet at 50% and 75% overlap for 2025-07-07 to 2025-10-15, load instantly.
- To run *new* configurations from the app, the local `models/` folder and the
  HDF5 cube must be present. Both are git-ignored and kept local.

### Run it

From the repository root, with the environment active:

```bash
streamlit run app.py
```

It opens in your browser at http://localhost:8501. To keep it reachable only from
your own machine (not the local network), bind it to localhost:

```bash
streamlit run app.py --server.address 127.0.0.1
```

### Using it

- The sidebar holds the controls: model, before date, after date, window overlap,
  and a voting-strictness slider.
- Moving the **voting-strictness** slider re-thresholds the existing run instantly,
  with no model run. Changing the model, dates, or overlap to a combination that is
  not already in `outputs/predictions/` shows a **Run** button that launches the
  model, with a green progress bar.
- Toggle the **ICNF ground truth** and the **Portugal boundary** as map overlays,
  and use the inset map in the corner to locate the view.
