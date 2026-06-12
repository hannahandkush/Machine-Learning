# Project Proposal

## Project Title
Burned Area Detection in Portugal from Sentinel-2 10-Band Time Series Using a Deep Learning Change Detection Model

## Project Category
Remote sensing / geospatial image classification - burned area mapping and binary change detection ("other: satellite time-series burned area detection"; closest standard categories: *image segmentation*, *binary classification*)

## Team Members
- Hannah Nathanson - *[Student ID]*
- Danilo III Ortañez Gonzales - *[Student ID]*

*(Replace the bracketed placeholders with your actual student ID numbers before submission.)*

## Project Plan

### Problem Statement
Wildfires are one of the most damaging and recurrent hazards in Portugal, and timely, spatially accurate burned area maps underpin a cascade of critical downstream responses. ICNF currently produces authoritative burned-area perimeters (`ardida`) through manual satellite image interpretation, a process whose lag between fire occurrence and published polygon directly delays aerial firefighting resource allocation, emergency logistics, and legal and insurance triggers. A model that reliably reproduces this delineation from Sentinel-2 imagery alone would reduce that lag and provide a scalable alternative to manual mapping. Beyond immediate response, accurate burned perimeters are the entry point for post-fire erosion and flood risk modelling: burned soils lose their infiltration capacity and become highly erodible, so the spatial precision of the burn boundary propagates directly into debris-flow early warning systems and hydrological damage assessments. This project applies a pre-trained Swin Transformer change detection model, developed by the MLC research group for 10-band Sentinel-2 surface reflectance, to map burned areas across tile T29TPG in northeastern Portugal. Model predictions are assessed against ICNF's 2025 burned-area polygons, providing a spatially explicit accuracy assessment on Portuguese data.

### Challenges
The principal challenge is class imbalance: burned pixels represent a small and spatially irregular fraction of any Sentinel-2 tile, so standard accuracy metrics are misleading and the model must be evaluated with metrics that are robust to skewed class distributions. Sentinel-2 time series are also interrupted by cloud cover and swath gaps, so per-scene quality filtering using the completeness categories already computed in the data exploration stage is a prerequisite before any chip-level inference. A further challenge is the selection of appropriate before/after acquisition pairs: the model is a change detector that compares a pre-fire chip against a post-fire chip, so the quality of the prediction depends on choosing cloud-free acquisitions that bracket each fire event as closely as possible in time. Finally, the ICNF ground-truth layer captures burned perimeters at the polygon level; translating these to per-pixel labels on the HDF5 chip grid requires careful rasterisation to avoid label leakage at polygon boundaries.

### Dataset
The primary dataset is an HDF5 cube of Sentinel-2 Level-2A surface reflectance for tile T29TPG (EPSG:32629, 10 m resolution, 10 spectral bands B2-B12, a year-long acquisition record for 2025, organised into 256x256-pixel chips). Exploratory analysis has already characterised the chunk layout, per-scene cloud-cover statistics, and the spatial distribution of usable acquisitions. Ground truth comes from ICNF's 2025 burned-area polygons (`ardida_2025`), obtained via their public WFS service, which provide official fire-perimeter polygons for mainland Portugal and serve as the reference labels for evaluation.

### Method or Algorithm
The model is a pre-trained Swin Transformer provided by the MLC research group via a standardised `predict(before_chip, after_chip, path_to_weights)` interface. Both input chips are 10x256x256 uint16 arrays containing surface reflectance values in the range 0-10000 across bands B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, with NoData encoded as 65535. Preprocessing - conversion from uint16 to uint8 and normalisation - is handled internally by the predict function. The output is a binary 256x256 array where 0 is background (not burned) and 1 is burned.

The inference pipeline proceeds as follows: (1) per-scene quality filtering to identify cloud-free acquisitions from the HDF5 time series, (2) selection of before/after chip pairs that bracket known fire events as closely as possible, (3) extraction of chip arrays and forward pass through the Swin Transformer to produce binary prediction maps, and (4) georeferencing and mosaicking of prediction chips into a tile-wide burned area map for evaluation. The pipeline is designed to be modular so that the predict function and weights can be swapped without changes to the surrounding infrastructure.

### Evaluation
The model is used purely for inference on T29TPG data: no training or fine-tuning is performed on the study tile, so no train/test split is required. Evaluation is against ICNF `ardida_2025` polygons rasterised to the 10 m chip grid, with edge pixels excluded by eroding polygon interiors by one pixel before rasterisation to avoid ambiguous boundary labels.

Because burned pixels are a small fraction of the tile, overall accuracy is not used as a primary metric: a trivial classifier that predicts no burned pixels would score above 95% yet have zero utility (Campagnolo, T6). Evaluation instead centres on the confusion matrix and the metrics derived from its four cells (TP, TN, FP, FN), all computed on the burned class. The primary metrics are precision (complement of commission error), recall (complement of omission error), F1 score computed on the burned class only, and the Matthews Correlation Coefficient (MCC). MCC is preferred as the single-number summary because it incorporates all four confusion matrix entries and is invariant to class swapping, making it the most balanced measure for an imbalanced binary problem (Campagnolo, T6; Raschka, Ch. 6). Since the model outputs a hard binary prediction rather than a probability, threshold-dependent analysis replaces ROC/AUC. Per-acquisition metrics will also be reported to assess how performance varies with scene cloud fraction and the temporal gap between the before/after acquisition and the actual fire date.

---
*Word count (description only): ~490 words*
