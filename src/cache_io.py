"""Cached BAR-ATH loader (avoids re-parsing the 195 MB ATD text on every experiment)."""

import os
import pickle

from src.atd_io import load_match

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE = os.path.join(ROOT, "data", "cache")


def load_match_cached(game="BAR-ATH", data_dir=None):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, "{}_match.pkl".format(game))
    if data_dir is None and os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    data = load_match(game=game, data_dir=data_dir)
    if data_dir is None:
        with open(path, "wb") as fh:
            pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return data
