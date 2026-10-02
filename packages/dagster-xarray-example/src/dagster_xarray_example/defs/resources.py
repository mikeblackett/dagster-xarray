import dagster as dg
from dagster_xarray import ZarrXarrayIOManager, DaskClusterResource
from upath import UPath

from dagster_xarray_example.resources import XarrayTutorialIOManager


@dg.definitions
def defs():
    return dg.Definitions(
        resources={
            "zarr_io": ZarrXarrayIOManager(
                base_path=UPath().cwd().joinpath("data/out")
            ),
            "xarray_tutorial": XarrayTutorialIOManager(),
            "dask": DaskClusterResource(),
        },
    )
