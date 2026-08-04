from typing import Literal, TypedDict

import dagster as dg
import dask.distributed as dd


class DaskLocalClusterKwargs(TypedDict):
    """Keyword arguments for configuring the DaskLocalCluster."""

    processes: bool
    n_workers: int | None
    threads_per_worker: int | None
    memory_limit: str | float | Literal["auto"]


class DaskClusterResource(dg.ConfigurableResource):
    """Resource for sharing a Dask LocalCluster across components.

    Args:
        processes (bool, optional): Whether to use processes (True) or threads (False). Defaults to True.
        n_workers (int | None, optional): The number of workers to start.
        threads_per_worker (int | None, optional): The number of threads to use per worker.
    """

    processes: bool = True
    n_workers: int | None = None
    threads_per_worker: int | None = None

    @property
    def client(self) -> dd.Client:
        return self._client

    def setup_for_execution(self, context: dg.InitResourceContext) -> None:
        assert context.log
        context.log.info(self.get_setup_log_message())

        self._client = dd.Client(
            n_workers=self.n_workers,
            processes=self.processes,
            threads_per_worker=self.threads_per_worker,
        )
        self._client.as_current()

        context.log.info(self.get_dashboard_log_message())

    def teardown_after_execution(self, context):
        assert context.log
        context.log.info(self.get_teardown_log_message())
        self._client.close()

    def get_setup_log_message(self) -> str:
        return f"starting dask LocalCluster using {self.__class__.__name__}..."

    def get_dashboard_log_message(self) -> str:
        return f"dask LocalCluster available at {self._client.dashboard_link}"

    def get_teardown_log_message(self) -> str:
        return f"closing dask LocalCluster {self._client}"
