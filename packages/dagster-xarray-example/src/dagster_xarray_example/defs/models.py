import numpy as np
import pandera.xarray as pa

_air_temperature = pa.DataArraySchema(
    dtype=np.float64,
    dims=("time", "lat", "lon"),
)

air_temperature = pa.DatasetSchema(
    name="AirTemperature",
    data_vars={"air": _air_temperature},
    coords={
        "time": pa.Coordinate(dtype=np.datetime64, dimension=True),
        "lat": pa.Coordinate(dtype=np.float32, dimension=True),
        "lon": pa.Coordinate(dtype=np.float32, dimension=True),
    },
)
