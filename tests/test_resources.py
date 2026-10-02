from typing import Any

import dask.distributed as dd
import pytest

from dagster_xarray import DaskClusterResource


def test_defaults() -> None:
    resource = DaskClusterResource()
    assert resource.processes is True
    assert resource.n_workers is None
    assert resource.threads_per_worker is None
    assert resource.memory_limit is None


def test_client_before_setup_raises() -> None:
    resource = DaskClusterResource()
    with pytest.raises(RuntimeError, match="not set up"):
        _ = resource.client


def test_teardown_without_setup(resource_context: Any) -> None:
    resource = DaskClusterResource()
    resource.teardown_after_execution(resource_context)


def test_setup_and_teardown(resource_context: Any) -> None:
    resource = DaskClusterResource(
        processes=False, n_workers=1, threads_per_worker=1
    )
    resource.setup_for_execution(resource_context)
    try:
        assert isinstance(resource.client, dd.Client)
        assert resource.client.status == "running"
        cluster = resource.client.cluster
        assert cluster is not None
        assert cluster.status == dd.Status.running
    finally:
        resource.teardown_after_execution(resource_context)
    cluster = resource.client.cluster
    assert cluster is not None
    assert cluster.status == dd.Status.closed


def test_setup_with_memory_limit(resource_context: Any) -> None:
    resource = DaskClusterResource(
        processes=False,
        n_workers=1,
        threads_per_worker=1,
        memory_limit="1GB",
    )
    assert resource.memory_limit == "1GB"
    resource.setup_for_execution(resource_context)
    try:
        assert isinstance(resource.client, dd.Client)
    finally:
        resource.teardown_after_execution(resource_context)
