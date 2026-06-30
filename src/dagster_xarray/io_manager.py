from abc import ABC, abstractmethod
from collections.abc import Mapping
from enum import StrEnum
import os
from typing import Any, Literal, TypeIs, cast

import dagster as dg
import xarray as xr
from upath import UPath

type NetcdfEngine = Literal["netcdf4", "scipy", "h5netcdf"]
type Engine = NetcdfEngine | Literal["zarr"]

DEFAULT_NETCDF_ENGINE: Engine = "netcdf4"
EXTENSION_FOR_ENGINE: dict[Engine, str] = {
    "zarr": ".zarr",
    "netcdf4": ".nc",
    "h5netcdf": ".nc",
    "scipy": ".nc",
}

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

LOCAL_UPATH_PROTOCOLS = ("", "file", "local", "memory")


class ZarrMode(StrEnum):
    CREATE = "w"
    CREATE_SAFE = "w-"


class NetCDFMode(StrEnum):
    CREATE = "w"


def _is_pathlike(path: object) -> TypeIs[os.PathLike]:
    return isinstance(path, os.PathLike)


class XarrayIOManager(dg.UPathIOManager, ABC):
    """An base IOManager for reading and writing xarray Datasets"""

    open_options: Mapping[str, Any]
    save_options: Mapping[str, Any]

    def __init__(
        self,
        base_path: UPath | None = None,
        engine: Engine | None = None,
        open_options: Mapping[str, Any] = {},
        save_options: Mapping[str, Any] = {},
    ):
        super().__init__(base_path)
        self._engine: Engine | None = engine
        self.extension = EXTENSION_FOR_ENGINE[self.engine]
        self.open_options = open_options
        self.save_options = save_options

    @property
    @abstractmethod
    def engine(self) -> Engine: ...

    def get_metadata(
        self, context: dg.OutputContext, obj: xr.DataArray | xr.Dataset
    ) -> dict[str, dg.MetadataValue]:
        from dask.utils import format_bytes

        return {
            "bytes": dg.MetadataValue.int(obj.nbytes),
            "file_size": dg.MetadataValue.text(format_bytes(obj.nbytes)),
        }

    def _resolve_input_options(
        self, context: dg.InputContext
    ) -> dict[str, Any]:
        raw = (context.metadata or {}).get("xarray/open", {})
        changes = raw.value if isinstance(raw, dg.MetadataValue) else raw
        return {
            k: v
            for k, v in (self.open_options | changes).items()
            if k not in BLACKLISTED_OPEN_DATASET_ARGS
        }

    def _resolve_output_options(self, context: dg.OutputContext) -> dict[str, Any]:
        raw = (context.metadata or {}).get("xarray/save", {})
        changes = raw.value if isinstance(raw, dg.MetadataValue) else raw
        return {
            k: v
            for k, v in (self.save_options | changes).items()
            if k not in BLACKLISTED_WRITE_ARGS
        }


class NetCDFXarrayIOManager(XarrayIOManager):
    """An IOManager for reading and writing xarray objects via NetCDF."""

    @property
    def engine(self) -> Engine:
        return DEFAULT_NETCDF_ENGINE if self._engine is None else self._engine

    def load_from_path(
        self,
        context: dg.InputContext,
        path: UPath,
    ) -> xr.Dataset:
        if not _is_pathlike(path):
            raise NotImplementedError(
                "NetCDF reads are local-only for now."
                " Use the zarr manager for object storage."
            )
        kwargs = self._resolve_input_options(context)
        return xr.open_dataset(path, engine=self.engine, **kwargs)

    def dump_to_path(
        self,
        context: dg.OutputContext,
        obj: xr.DataArray | xr.Dataset,
        path: UPath,
    ) -> None:
        if not _is_pathlike(path):
            # TODO: (mike) Writing to remote store needs temp-file staging
            raise NotImplementedError(
                "NetCDF writes are local-only for now."
                " Use the zarr manager for object storage."
            )
        kwargs = self._resolve_output_options(context)
        mode = NetCDFMode(kwargs.pop("mode", "w")).value
        engine = cast(NetcdfEngine, self.engine)
        obj.drop_encoding().to_netcdf(
            path=path, compute=True, engine=engine, mode=mode, **kwargs
        )


class ZarrXarrayIOManager(XarrayIOManager):
    """An IOManager for reading and writing xarray objects via Zarr."""

    @property
    def engine(self) -> Engine:
        return "zarr"

    def load_from_path(
        self,
        context: dg.InputContext,
        path: UPath,
    ) -> xr.Dataset:
        kwargs = self._resolve_input_options(context)
        backend_kwargs = self._resolve_backend_kwargs(path)
        return xr.open_dataset(
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
        obj.drop_encoding().to_zarr(store=path, compute=True, mode=mode, **kwargs)

    def _resolve_backend_kwargs(self, path: UPath) -> dict[str, Any]:
        opts = dict(path.storage_options)
        if not opts or path.protocol in LOCAL_UPATH_PROTOCOLS:
            return {}
        return {"storage_options": opts}
