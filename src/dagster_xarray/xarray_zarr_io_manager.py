from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Literal

import xarray as xr
from dagster import (
    InputContext,
    OutputContext,
    UPathIOManager,
)
from dask.array.rechunk import rechunk
from dask.distributed import Client
from upath import UPath
from xarray.core.types import T_Chunks

ZarrPersistenceModeType = Literal['w', 'w-', 'a', 'a-', 'r+']


class ZarrPersistenceMode(StrEnum):
    """Enum for Zarr write modes."""

    # Create (overwrite if exists)
    CREATE = 'w'
    # Create (fail if exists)
    CREATE_SAFE = 'w-'
    # Override all existing variables (create if not exists)
    APPEND = 'a'
    # Append only those variables that have `append_dim`
    APPEND_DIM = 'a-'
    # Modify existing array values only (fail if any metadata or shapes would change)
    MODIFY = 'r+'


@dataclass(frozen=True)
class ZarrStoreConfig:
    """Config for the Zarr store."""

    mode: ZarrPersistenceModeType = ZarrPersistenceMode.CREATE_SAFE
    """The mode to use when writing datasets. Defaults to 'w'."""

    consolidated: bool = True
    """Whether to consolidate the Zarr store. Defaults to True."""

    group: str | None = None
    """The group to use when writing datasets. Defaults to None."""


@dataclass(frozen=True)
class DaskClusterConfig:
    """Config for the Dask local cluster."""

    processes: bool = True
    """Whether to use processes (True) or threads (False). Defaults to True."""

    n_workers: int | None = None
    """The number of workers to start."""

    memory_limit: str | float | Literal['auto'] = 'auto'
    """The amount of memory to use *per worker*."""

    threads_per_worker: int | None = None
    """The number of threads to use per worker."""


class XarrayZarrIOManager(UPathIOManager):
    """An IOManager for reading and writing xarray datasets from Zarr stores.

    This IOManager uses Dask to parallelize read and write operations.

    Attributes
    ----------
    base_path: str | None
        The base path to use when reading or writing datasets.
    extension: str | None
        The extension to use when writing datasets. Defaults to '.zarr'.
    processes: bool | None
        Whether to use processes (True) or threads (False). Defaults to True.
    n_workers: int | None
        The number of workers to start.
    memory_limit: str | float | Literal['auto'] | None
        The amount of memory to use *per worker*.
    threads_per_worker: int | None
        The number of threads to use per worker.
    persistence_mode: ZarrPersistenceModeType
        The mode to use when writing datasets. Defaults to 'w'.
    """

    base_path: str | None = None
    """The base path to use when reading or writing datasets."""

    extension: str | None = '.zarr'
    """The extension to use when writing datasets. Defaults to '.zarr'."""

    processes: bool | None = True
    """Whether to use processes (True) or threads (False). Defaults to False."""

    n_workers: int | None = None
    """The number of workers to start."""

    memory_limit: str | float | Literal['auto'] | None = 'auto'
    """The amount of memory to use *per worker*."""

    threads_per_worker: int | None = None
    """The number of threads to use per worker."""

    persistence_mode: ZarrPersistenceMode = ZarrPersistenceMode.CREATE
    """The mode to use when writing datasets. Defaults to 'w'."""

    def __init__(
        self,
        base_path: UPath,
        *,
        processes: bool | None = None,
        n_workers: int | None = None,
        memory_limit: str | float | Literal['auto'] | None = None,
        threads_per_worker: int | None = None,
        persistence_mode: ZarrPersistenceModeType = ZarrPersistenceMode.CREATE,
    ) -> None:
        super().__init__(base_path)
        self.processes = processes
        self.n_workers = n_workers
        self.memory_limit = memory_limit
        self.threads_per_worker = threads_per_worker
        self.persistence_mode = ZarrPersistenceMode(persistence_mode)

    @property
    def _cluster_config(self) -> dict[str, Any]:
        return {
            'processes': self.processes,
            'n_workers': self.n_workers,
            'memory_limit': self.memory_limit,
            'threads_per_worker': self.threads_per_worker,
        }

    def get_cluster_config(self, context: OutputContext) -> DaskClusterConfig:
        metadata: dict[str, Any] = getattr(context, 'metadata', {})
        cluster_config = {
            key.removeprefix('cluster/'): value
            for key, value in metadata.items()
            if key.startswith('cluster/')
        }
        return DaskClusterConfig(**(self._cluster_config | cluster_config))

    def dump_to_path(
        self,
        context: OutputContext,
        obj: xr.Dataset | xr.DataArray,
        path: UPath,
    ) -> None:
        context.log.info(
            (
                f'{type(self).__name__}: Saving asset '
                f'{context.get_asset_identifier()} to {path!r}'
            )
        )
        cluster_config = self.get_cluster_config(context)
        metadata = getattr(context, 'metadata', {})
        mode = metadata.get('zarr/mode', self.persistence_mode)
        mode = ZarrPersistenceMode(mode).value
        consolidated = metadata.get('zarr/consolidated', True)
        group: str | None = metadata.get('zarr/group')
        chunks: T_Chunks = metadata.get('xarray/chunks', {})
        with Client(**asdict(cluster_config)) as client:
            address = client.dashboard_link if client.scheduler else None
            context.log.info(
                f'{type(self).__name__}: '
                f'Writing file to Zarr store using Dask cluster {address!r}'
            )
            if chunks:
                obj = obj.chunk(chunks)
            obj.pipe(balance_chunks).drop_encoding().to_zarr(
                store=path, mode=mode, group=group, consolidated=consolidated
            )
            obj.close()

    def load_from_path(
        self,
        context: InputContext,
        path: UPath,
    ) -> xr.Dataset:
        context.log.info(
            (
                f'{type(self).__name__}: Loading asset'
                f'{context.get_asset_identifier()} from {path}'
            )
        )
        metadata: dict[str, Any] = getattr(context, 'metadata', {})
        indexers: dict[str, Any] | None = metadata.get('xarray/indexers')
        indexers_method = metadata.get('xarray/indexers/method')
        chunks: T_Chunks = metadata.get('xarray/chunks', {})
        out = xr.open_dataset(
            path,
            chunks=chunks,
            # Avoid decoding data variables with `units: days` as `timedelta64`
            # See: https://github.com/pydata/xarray/issues/1621
            decode_timedelta=False,
            engine='zarr',
        )
        if indexers:
            out = out.sel(indexers=indexers, method=indexers_method)
        return out


def balance_chunks[T: xr.DataArray | xr.Dataset](
    obj: T, block_size_limit=1e8
) -> T:
    if isinstance(obj, xr.Dataset):
        return obj.map(balance_chunks, block_size_limit=block_size_limit)
    chunked = rechunk(
        obj.data,
        chunks='auto',
        balance=True,
        block_size_limit=block_size_limit,
    )
    return obj.copy(data=chunked)
