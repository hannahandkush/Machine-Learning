"""Tile-wide burned-area inference for T29TPG.

Deploys the fine-tuned Swin-YNet change-detection model
(`models/updated_model/bacdm_predict`) over the Sentinel-2 HDF5 cube and
reassembles per-chip predictions into georeferenced burned-area rasters.

See `inference/run.py` for the CLI entry point and the project plan for the
overall design.
"""
