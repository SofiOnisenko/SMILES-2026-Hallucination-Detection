from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    random_state: int = 42,
):
    """
    Creates stratified 5-fold splits with a separate validation subset
    """

    indices = np.arange(len(y))

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )

    splits = []

    for train_val_idx, test_idx in skf.split(indices, y):

        split_point = int(0.8 * len(train_val_idx))

        train_idx = train_val_idx[:split_point]
        val_idx = train_val_idx[split_point:]

        splits.append(
            (
                train_idx,
                val_idx,
                test_idx,
            )
        )

    return splits