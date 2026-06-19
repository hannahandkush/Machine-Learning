import h5py
import pandas as pd
import numpy as np

def get_timestamps():
    try:
        with h5py.File("data/hdf5/T29TPG.h5", "r") as f:
            ts = f["timestamps"][:]
            return [t.decode('utf-8') if isinstance(t, bytes) else t for t in ts]
    except Exception as e:
        print(f"Error: {e}")
        return None

ts = get_timestamps()
if ts:
    for t in ts:
        print(t)
