import re
from typing import Any

import dagster as dg
import dagster._check as check
import numpy as np
import pandera
import pandera.xarray as pa
import pytest
import xarray as xr

from dagster_xarray import pandera_schema_to_dagster_type


def make_da_schema() -> pa.DataArraySchema:
    return pa.DataArraySchema(
        dtype=np.float64,
        dims=("time", "lat", "lon"),
    )


def make_ds_schema() -> pa.DatasetSchema:
    return pa.DatasetSchema(
        name="AirTemperature",
        data_vars={"air": make_da_schema()},
        coords={
            "time": pa.Coordinate(dtype=np.datetime64, dimension=True),
            "lat": pa.Coordinate(dtype=np.float64, dimension=True),
            "lon": pa.Coordinate(dtype=np.float64, dimension=True),
        },
    )


def check_value(dagster_type: dg.DagsterType, value: object) -> dg.TypeCheck:
    # The type check functions under test ignore the context.
    context: Any = None
    return dagster_type.type_check(context, value)


def schema_meta(dagster_type: dg.DagsterType) -> dict[str, Any]:
    value = dagster_type.metadata["schema"].value
    assert isinstance(value, dict)
    return value


def test_dataset_schema_name_typing_and_metadata(sample_dataset) -> None:
    dagster_type = pandera_schema_to_dagster_type(make_ds_schema())
    assert dagster_type._name == "AirTemperature"
    assert dagster_type.typing_type is xr.Dataset
    meta = schema_meta(dagster_type)
    assert meta["schema_type"] == "dataset"
    assert meta["title"] == "AirTemperature"
    assert "air" in meta["data_vars"]


def test_description_passthrough(sample_dataset) -> None:
    schema = make_ds_schema()
    schema.description = "Monthly air temperature"
    dagster_type = pandera_schema_to_dagster_type(schema)
    assert dagster_type._description == "Monthly air temperature"


def test_valid_dataset_passes(sample_dataset) -> None:
    dagster_type = pandera_schema_to_dagster_type(make_ds_schema())
    result = check_value(dagster_type, sample_dataset)
    assert result.success is True


def test_wrong_dtype_fails(sample_dataset) -> None:
    dagster_type = pandera_schema_to_dagster_type(make_ds_schema())
    bad = sample_dataset.assign_coords(
        lat=sample_dataset.lat.astype("float32")
    )
    result = check_value(dagster_type, bad)
    assert result.success is False
    assert "WRONG_DATATYPE" in str(result.description)


def test_dataset_type_rejects_dataarray(sample_dataset) -> None:
    dagster_type = pandera_schema_to_dagster_type(make_ds_schema())
    result = check_value(dagster_type, sample_dataset["air"])
    assert result.success is False
    assert result.description == "Must be a xarray Dataset, got DataArray."


def test_dataarray_type_rejects_dataset(sample_dataset) -> None:
    dagster_type = pandera_schema_to_dagster_type(make_da_schema())
    result = check_value(dagster_type, sample_dataset)
    assert result.success is False
    assert result.description == "Must be a xarray DataArray, got Dataset."


def test_non_xarray_value_rejected(sample_dataset) -> None:
    dagster_type = pandera_schema_to_dagster_type(make_ds_schema())
    result = check_value(dagster_type, 42)
    assert result.success is False
    assert "got int" in str(result.description)


def test_dataarray_schema_valid(sample_dataarray) -> None:
    dagster_type = pandera_schema_to_dagster_type(make_da_schema())
    assert dagster_type.typing_type is xr.DataArray
    result = check_value(dagster_type, sample_dataarray)
    assert result.success is True


def test_model_class_uses_config_title(sample_dataset) -> None:
    class AirModel(pa.DatasetModel):
        class Config:  # pyright: ignore[reportIncompatibleVariableOverride]
            title = "AirModelTitle"

    dagster_type = pandera_schema_to_dagster_type(AirModel)
    assert dagster_type._name == "AirModelTitle"
    assert schema_meta(dagster_type)["title"] == "AirModelTitle"


def test_model_class_falls_back_to_class_name(sample_dataset) -> None:
    class MyModel(pa.DatasetModel):
        pass

    dagster_type = pandera_schema_to_dagster_type(MyModel)
    assert dagster_type._name == "MyModel"
    assert schema_meta(dagster_type)["title"] == "MyModel"


def test_anonymous_schema_names(sample_dataset) -> None:
    first = pandera_schema_to_dagster_type(make_da_schema())
    second = pandera_schema_to_dagster_type(make_da_schema())
    first_name = first._name
    second_name = second._name
    assert isinstance(first_name, str)
    assert isinstance(second_name, str)
    assert re.fullmatch(r"DagsterPanderaXarray\d+", first_name)
    assert re.fullmatch(r"DagsterPanderaXarray\d+", second_name)
    assert first_name != second_name


def test_invalid_schema_inputs_rejected(sample_dataset) -> None:
    bad_inputs: tuple[object, ...] = (
        123,
        "nope",
        pandera.DataFrameSchema,
    )
    for bad in bad_inputs:
        with pytest.raises(check.CheckError):
            pandera_schema_to_dagster_type(bad)  # type: ignore[arg-type]
