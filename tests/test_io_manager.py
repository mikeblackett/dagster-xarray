from typing import Any

import dagster as dg
import pytest
import xarray as xr
from upath import UPath

from dagster_xarray import NetCDFXarrayIOManager, ZarrXarrayIOManager


def test_zarr_roundtrip_dataset(
    tmp_path: Any, sample_dataset: xr.Dataset, io_context: Any
) -> None:
    base = UPath(tmp_path) / "z"
    manager = ZarrXarrayIOManager(base_path=base)
    path = base / "asset.zarr"
    manager.dump_to_path(io_context(), sample_dataset, path)
    loaded = manager.load_from_path(io_context(), path)
    xr.testing.assert_identical(loaded, sample_dataset.drop_encoding())


def test_zarr_roundtrip_dataarray(
    tmp_path: Any, sample_dataarray: xr.DataArray, io_context: Any
) -> None:
    base = UPath(tmp_path) / "z"
    manager = ZarrXarrayIOManager(base_path=base)
    path = base / "asset.zarr"
    manager.dump_to_path(io_context(xr.DataArray), sample_dataarray, path)
    loaded = manager.load_from_path(io_context(xr.DataArray), path)
    xr.testing.assert_identical(loaded, sample_dataarray.drop_encoding())


def test_netcdf_roundtrip_dataset(
    tmp_path: Any, sample_dataset: xr.Dataset, io_context: Any
) -> None:
    base = UPath(tmp_path) / "nc"
    base.mkdir(parents=True, exist_ok=True)
    manager = NetCDFXarrayIOManager(base_path=base)
    path = base / "asset.nc"
    manager.dump_to_path(io_context(), sample_dataset, path)
    loaded = manager.load_from_path(io_context(), path)
    xr.testing.assert_identical(loaded, sample_dataset.drop_encoding())


def test_netcdf_roundtrip_dataarray(
    tmp_path: Any, sample_dataarray: xr.DataArray, io_context: Any
) -> None:
    base = UPath(tmp_path) / "nc"
    base.mkdir(parents=True, exist_ok=True)
    manager = NetCDFXarrayIOManager(base_path=base)
    path = base / "asset.nc"
    manager.dump_to_path(io_context(xr.DataArray), sample_dataarray, path)
    loaded = manager.load_from_path(io_context(xr.DataArray), path)
    xr.testing.assert_identical(loaded, sample_dataarray.drop_encoding())


def test_netcdf_remote_read_not_implemented(
    tmp_path: Any, io_context: Any
) -> None:
    manager = NetCDFXarrayIOManager(base_path=UPath(tmp_path))
    remote = UPath("s3://bucket/asset.nc")
    with pytest.raises(NotImplementedError, match="local-only"):
        manager.load_from_path(io_context(), remote)


def test_netcdf_remote_write_not_implemented(
    tmp_path: Any, sample_dataset: xr.Dataset, io_context: Any
) -> None:
    manager = NetCDFXarrayIOManager(base_path=UPath(tmp_path))
    remote = UPath("s3://bucket/asset.nc")
    with pytest.raises(NotImplementedError, match="local-only"):
        manager.dump_to_path(io_context(), sample_dataset, remote)


def test_open_options_merged_and_blacklisted(io_context: Any) -> None:
    manager = ZarrXarrayIOManager(
        base_path=UPath("/nonexistent"),
        open_options={
            "decode_times": True,
            "engine": "netcdf4",
            "backend_kwargs": {},
            "drop_variables": ["x"],
        },
    )
    options = manager._resolve_input_options(io_context())
    assert options == {"decode_times": True}


def test_definition_metadata_overrides_open_options(
    io_context: Any,
) -> None:
    manager = ZarrXarrayIOManager(
        base_path=UPath("/nonexistent"),
        open_options={"decode_times": True},
    )
    context = io_context(
        definition_metadata={
            "xarray/open": dg.MetadataValue.json(
                {"decode_times": False, "engine": "zarr"}
            )
        }
    )
    options = manager._resolve_input_options(context)
    assert options == {"decode_times": False}


def test_definition_metadata_plain_dict(
    io_context: Any,
) -> None:
    manager = ZarrXarrayIOManager(
        base_path=UPath("/nonexistent"),
    )
    context = io_context(
        definition_metadata={"xarray/open": {"mask_and_scale": False}}
    )
    options = manager._resolve_input_options(context)
    assert options == {"mask_and_scale": False}


def test_save_options_blacklisted(io_context: Any) -> None:
    manager = ZarrXarrayIOManager(
        base_path=UPath("/nonexistent"),
        save_options={
            "mode": "w-",
            "chunk_store": "other",
            "compute": False,
            "engine": "zarr",
        },
    )
    options = manager._resolve_output_options(io_context())
    assert options == {"mode": "w-"}


def test_get_metadata(sample_dataset: xr.Dataset, io_context: Any) -> None:
    manager = ZarrXarrayIOManager(base_path=UPath("/nonexistent"))
    metadata = manager.get_metadata(io_context(), sample_dataset)
    assert metadata["bytes"].value == sample_dataset.nbytes
    assert isinstance(metadata["in_memory_size"], dg.MetadataValue)
    assert str(metadata["in_memory_size"].value)
