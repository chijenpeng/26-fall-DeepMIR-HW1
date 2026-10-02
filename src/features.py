"""Assemble design matrices from the saved feature files."""
import numpy as np
from config import FEATURES


def load_handcrafted(ds, path=None):
    z = np.load(path or FEATURES / f"{ds}_handcrafted.npz", allow_pickle=True)
    return z["X"], list(z["names"]), z


def load_mert(ds, path=None, model="MERT-v1-95M", chunk_sec=5):
    return np.load(path or FEATURES / f"{ds}_mert_{model}_{chunk_sec}s.npz", allow_pickle=True)


def mert_vector(z, layers, n_chunks=None, use_std=True, chunk_offset=0):
    """Recording-level vector: average chunk statistics over the first n_chunks chunks,
    concatenate the requested layers (and optionally the time-std stats)."""
    mean, std = z["mean"].astype(np.float32), z["std"].astype(np.float32)
    sl = slice(chunk_offset, None if n_chunks is None else chunk_offset + n_chunks)
    mu = mean[:, sl].mean(1)[:, layers]            # [N, len(layers), D]
    parts = [mu.reshape(len(mu), -1)]
    if use_std:
        parts.append(std[:, sl].mean(1)[:, layers].reshape(len(mu), -1))
    return np.concatenate(parts, 1)


def load_lang(ds, path=None, model="whisper-small"):
    return np.load(path or FEATURES / f"{ds}_lang_{model}.npz", allow_pickle=True)
