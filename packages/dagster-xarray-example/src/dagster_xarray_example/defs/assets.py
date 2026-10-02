import dagster as dg
import xarray as xr

import dagster_xarray as dx

from dagster_xarray_example.defs import models

air_temperature_type = dx.pandera_schema_to_dagster_type(
    models.air_temperature
)

air_temperature = dg.AssetSpec(
    key="air_temperature",
    description="NCEP reanalysis subset",
    kinds={"xarray"},
).with_io_manager_key("xarray_tutorial")


@dg.asset(
    key_prefix="zarr",
    description="Monthly air temperature",
    ins={"air_temperature": dg.AssetIn()},
    io_manager_key="zarr_io",
    kinds={"zarr"},
    dagster_type=air_temperature_type,
    metadata=air_temperature_type.metadata,
    required_resource_keys={"dask"},
)
def monthly_air_temperature(
    context: dg.AssetExecutionContext, air_temperature: xr.Dataset
) -> xr.Dataset:
    return air_temperature.resample(time="MS").mean()


check_freq = dx.build_xarray_frequency_check(
    monthly_air_temperature, freq="MS"
)
