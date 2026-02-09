from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from backend.app.core import config as config_module
from backend.app.services.storage import load_dataset, load_meta, save_dataset


def test_save_and_load_dataset(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(config_module.settings, "data_dir", str(tmp_path))
    monkeypatch.setattr(config_module.settings, "max_saved_datasets", 100)

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-01-01", "2021-01-02"]),
            "symbol": ["AAA", "AAA"],
            "close": [1.0, 2.0],
        }
    )

    meta = save_dataset(df)
    assert meta.dataset_id
    assert meta.row_count == 2

    m2 = load_meta(meta.dataset_id)
    assert m2.dataset_id == meta.dataset_id

    df2 = load_dataset(meta.dataset_id)
    assert set(df2.columns) == set(df.columns)
    assert len(df2) == 2

