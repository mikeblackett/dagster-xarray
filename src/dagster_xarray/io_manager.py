import os
from abc import ABC
from collections.abc import Callable, Mapping
from enum import StrEnum
from typing import Any, Literal, TypeIs
import dagster as dg
import xarray as xr
from upath import UPath

type NetcdfEngine = Literal["netcdf4", "scipy", "h5netcdf"]
type Engine = NetcdfEngine | Literal["zarr"]

BLACKLISTED_OPEN_DATASET_ARGS = ("drop_variables", "engine", "backend_kwargs")
BLACKLISTED_TO_NETCDF_ARGS = ("compute", "group", "path", "engine")
BLACKLISTED_TO_ZARR_ARGS = (
    "append_dim",
    "chunk_store",
    "compute",
    "group",
    "path",
    "region",
    "store",
)
BLACKLISTED_WRITE_ARGS = tuple(
    {*BLACKLISTED_TO_NETCDF_ARGS, *BLACKLISTED_TO_ZARR_ARGS}
)


class ZarrMode(StrEnum):
    CREATE = "w"
    CREATE_SAFE = "w-"


class NetCDFMode(StrEnum):
    CREATE = "w"


def _is_local_path(path: object) -> TypeIs[os.PathLike]:
    # netcdf4/h5netcdf/scipy need a real OS path;
    # UPath only implements os.PathLike for local filesystems (upath >= 0.3.0).
    return isinstance(path, os.PathLike)


def _unwrap_maybe_dagster_type(obj: dg.DagsterType | type) -> type:
    if isinstance(obj, dg.DagsterType):
        return obj.typing_type
    return obj


def _storage_options_or_none(path: UPath) -> dict[str, Any] | None:
    return dict(path.storage_options) or None


class XarrayIOManager[E: Engine](dg.UPathIOManager, ABC):
    """An base IOManager for reading and writing xarray objects"""

    default_engine: E
    open_options: Mapping[str, Any]
    save_options: Mapping[str, Any]

    def __init__(
        self,
        base_path: UPath | None = None,
        engine: E | None = None,
        open_options: Mapping[str, Any] | None = None,
        save_options: Mapping[str, Any] | None = None,
    ):
        super().__init__(base_path)
        self.engine = self.default_engine if engine is None else engine
        self.open_options = open_options or {}
        self.save_options = save_options or {}

    def get_metadata(
        self, context: dg.OutputContext, obj: xr.DataArray | xr.Dataset
    ) -> dict[str, dg.MetadataValue]:
        from dask.utils import format_bytes

        return {
            "bytes": dg.MetadataValue.int(obj.nbytes),
            "in_memory_size": dg.MetadataValue.text(format_bytes(obj.nbytes)),
        }

    def _resolve_input_options(
        self, context: dg.InputContext
    ) -> dict[str, Any]:
        raw = (context.definition_metadata or {}).get("xarray/open", {})
        changes = raw.value if isinstance(raw, dg.MetadataValue) else raw
        return {
            k: v
            for k, v in {**self.open_options, **changes}.items()
            if k not in BLACKLISTED_OPEN_DATASET_ARGS
        }

    def _resolve_output_options(
        self, context: dg.OutputContext
    ) -> dict[str, Any]:
        raw = (context.definition_metadata or {}).get("xarray/save", {})
        changes = raw.value if isinstance(raw, dg.MetadataValue) else raw
        return {
            k: v
            for k, v in {**self.save_options, **changes}.items()
            if k not in BLACKLISTED_WRITE_ARGS
        }

    def _get_xarray_open_method(
        self, context: dg.InputContext
    ) -> Callable[..., xr.DataArray | xr.Dataset]:
        typing_type = _unwrap_maybe_dagster_type(context.dagster_type)
        if typing_type is xr.DataArray:
            return xr.open_dataarray
        return xr.open_dataset


class NetCDFXarrayIOManager(XarrayIOManager[NetcdfEngine]):
    """An IOManager for reading and writing xarray objects via NetCDF."""

    extension = ".nc"
    default_engine = "netcdf4"

    def load_from_path(
        self,
        context: dg.InputContext,
        path: UPath,
    ) -> xr.Dataset | xr.DataArray:
        if not _is_local_path(path):
            raise NotImplementedError(
                "NetCDF reads are local-only for now."
                " Use the zarr manager for object storage."
            )
        kwargs = self._resolve_input_options(context)
        open_xarray = self._get_xarray_open_method(context)
        return open_xarray(path, engine=self.engine, **kwargs)

    def dump_to_path(
        self,
        context: dg.OutputContext,
        obj: xr.DataArray | xr.Dataset,
        path: UPath,
    ) -> None:
        if not _is_local_path(path):
            # TODO: (mike) Writing to remote store needs temp-file staging
            raise NotImplementedError(
                "NetCDF writes are local-only for now."
                " Use the zarr manager for object storage."
            )
        kwargs = self._resolve_output_options(context)
        mode = NetCDFMode(kwargs.pop("mode", "w")).value
        obj.drop_encoding().to_netcdf(
            path=path, compute=True, engine=self.engine, mode=mode, **kwargs
        )


class ZarrXarrayIOManager(XarrayIOManager[Literal["zarr"]]):
    """An IOManager for reading and writing xarray objects via Zarr."""

    extension = ".zarr"
    default_engine = "zarr"

    def load_from_path(
        self,
        context: dg.InputContext,
        path: UPath,
    ) -> xr.Dataset | xr.DataArray:
        kwargs = self._resolve_input_options(context)
        backend_kwargs = (
            {"storage_options": opts}
            if (opts := _storage_options_or_none(path))
            else None
        )
        open_xarray = self._get_xarray_open_method(context)
        return open_xarray(
            str(path),
            engine=self.engine,
            backend_kwargs=backend_kwargs,
            **kwargs,
        )

    def dump_to_path(
        self,
        context: dg.OutputContext,
        obj: xr.DataArray | xr.Dataset,
        path: UPath,
    ) -> None:
        kwargs = self._resolve_output_options(context)
        mode = ZarrMode(kwargs.pop("mode", "w")).value
        obj.drop_encoding().to_zarr(
            store=str(path),
            compute=True,
            mode=mode,
            storage_options=_storage_options_or_none(path),
            **kwargs,
        )
