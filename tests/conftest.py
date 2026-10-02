import logging
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr


@pytest.fixture
def sample_dataset() -> xr.Dataset:
    rng = np.random.default_rng(seed=42)
    return xr.Dataset(
        {
            "air": (
                ("time", "lat", "lon"),
                rng.random((12, 4, 5)).astype("float64"),
            )
        },
        coords={
            "time": pd.date_range("2000-01-01", periods=12, freq="MS"),
            "lat": np.linspace(-90, 90, 4).astype("float64"),
            "lon": np.linspace(0, 359, 5).astype("float64"),
        },
    )


@pytest.fixture
def sample_dataarray(sample_dataset: xr.Dataset) -> xr.DataArray:
    return sample_dataset["air"]


class IoContextStub:
    """Minimal stand-in for dagster Input/Output contexts.

    Only the attributes read by the IO managers are provided.
    """

    def __init__(
        self,
        dagster_type: Any = xr.Dataset,
        definition_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.dagster_type = dagster_type
        self.definition_metadata = definition_metadata


@pytest.fixture
def io_context() -> Callable[..., IoContextStub]:
    def make(
        dagster_type: Any = xr.Dataset,
        definition_metadata: Mapping[str, Any] | None = None,
    ) -> IoContextStub:
        return IoContextStub(dagster_type, definition_metadata)

    return make


class ResourceContextStub:
    """Minimal stand-in for dagster InitResourceContext."""

    def __init__(self) -> None:
        self.log = logging.getLogger("dagster-xarray-tests")


@pytest.fixture
def resource_context() -> ResourceContextStub:
    return ResourceContextStub()
