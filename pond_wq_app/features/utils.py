"""
features/utils.py
=================
Shared utility functions for feature extraction.
"""

import numpy as np


def safex(x):
    """Convert a value to float, returning 0.0 for NaN/Inf/errors."""
    try:
        x = float(x)
        if np.isnan(x) or np.isinf(x):
            return 0.0
        return x
    except Exception:
        return 0.0
