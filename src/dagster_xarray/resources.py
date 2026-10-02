import dagster as dg
import dask.distributed as dd


class DaskClusterResource(dg.ConfigurableResource):
    """Resource for sharing a Dask LocalCluster across components.

    Args:
        processes (bool, optional): Whether to use processes (True) or threads (False). Defaults to True.
        n_workers (int | None, optional): The number of workers to start.
        threads_per_worker (int | None, optional): The number of threads to use per worker.
        memory_limit (str, optional): The memory limit per worker, as a
            dask size string (e.g. "2GB") or "auto". Defaults to the dask
            default ("auto").
    """

    processes: bool = True
    n_workers: int | None = None
    threads_per_worker: int | None = None
    memory_limit: str | None = None

    @property
    def client(self) -> dd.Client:
        if not hasattr(self, "_client"):
            raise RuntimeError(
                "DaskClusterResource is not set up yet. The client is "
                "only available during asset/job execution."
            )
        return self._client

    def setup_for_execution(self, context: dg.InitResourceContext) -> None:
        assert context.log
        context.log.info(self.get_setup_log_message())

        self._client = dd.Client(
            n_workers=self.n_workers,
            processes=self.processes,
            threads_per_worker=self.threads_per_worker,
            memory_limit=self.memory_limit,
        )
        self._client.as_current()

        context.log.info(self.get_dashboard_log_message())

    def teardown_after_execution(
        self, context: dg.InitResourceContext
    ) -> None:
        assert context.log
        if not hasattr(self, "_client"):
            return
        context.log.info(self.get_teardown_log_message())
        self._client.close()

    def get_setup_log_message(self) -> str:
        return f"starting dask LocalCluster using {self.__class__.__name__}..."

    def get_dashboard_log_message(self) -> str:
        return f"dask LocalCluster available at {self._client.dashboard_link}"

    def get_teardown_log_message(self) -> str:
        return f"closing dask LocalCluster {self._client}"
