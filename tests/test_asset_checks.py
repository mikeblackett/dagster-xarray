import dagster as dg
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from dagster_xarray import build_xarray_frequency_check
from dagster_xarray.asset_checks import validate_freq


@pytest.fixture
def monthly_dataset() -> xr.Dataset:
    return xr.Dataset(
        {"t": ("time", np.arange(12, dtype="float64"))},
        coords={"time": pd.date_range("2000-01-01", periods=12, freq="MS")},
    )


def test_validate_freq_ok(monthly_dataset: xr.Dataset) -> None:
    validate_freq(monthly_dataset, "MS")
    validate_freq(monthly_dataset, ["D", "MS"])


def test_validate_freq_wrong(monthly_dataset: xr.Dataset) -> None:
    with pytest.raises(ValueError, match="invalid frequency"):
        validate_freq(monthly_dataset, "D")


def test_validate_freq_irregular() -> None:
    dataset = xr.Dataset(
        {"t": ("time", np.arange(4, dtype="float64"))},
        coords={
            "time": pd.to_datetime(
                ["2000-01-01", "2000-01-02", "2000-01-04", "2000-01-08"]
            )
        },
    )
    with pytest.raises(ValueError, match="could not infer"):
        validate_freq(dataset, "D")


def test_validate_freq_missing_time() -> None:
    dataset = xr.Dataset(
        {"t": ("x", np.arange(3, dtype="float64"))},
        coords={"x": np.arange(3, dtype="float64")},
    )
    with pytest.raises(KeyError):
        validate_freq(dataset, "D")


def test_default_name_and_blocking() -> None:
    checks = build_xarray_frequency_check(dg.AssetKey("some_asset"), freq="MS")
    key = dg.AssetCheckKey(dg.AssetKey("some_asset"), "check_freq")
    assert checks.check_keys == {key}
    assert checks.get_spec_for_check_key(key).blocking is True


def test_custom_name_and_blocking() -> None:
    checks = build_xarray_frequency_check(
        dg.AssetKey("some_asset"),
        freq=["MS"],
        name="my_check",
        blocking=False,
    )
    key = dg.AssetCheckKey(dg.AssetKey("some_asset"), "my_check")
    assert checks.check_keys == {key}
    assert checks.get_spec_for_check_key(key).blocking is False


def test_asset_from_assets_definition() -> None:
    @dg.asset
    def my_asset() -> int:
        return 1

    checks = build_xarray_frequency_check(my_asset, freq="MS")
    assert checks.check_keys == {dg.AssetCheckKey(my_asset.key, "check_freq")}


def test_asset_from_string() -> None:
    checks = build_xarray_frequency_check("some_asset", freq="MS")
    assert checks.check_keys == {
        dg.AssetCheckKey(dg.AssetKey("some_asset"), "check_freq")
    }
