import itertools
import json
from collections.abc import Callable

import dagster as dg
import dagster._check as check
import pandera.errors as pa_errors
import pandera.xarray as pa
import xarray as xr

type DagsterPanderaXarraySchema = pa.DataArraySchema | pa.DatasetSchema
type DagsterPanderaXarrayModel = pa.DataArrayModel | pa.DatasetModel

VALID_SCHEMA_CLASSES = (pa.DataArraySchema, pa.DatasetSchema)
VALID_MODEL_CLASSES = (pa.DataArrayModel, pa.DatasetModel)
VALID_XARRAY_CLASSES = (xr.DataArray, xr.Dataset)


def pandera_schema_to_dagster_type(
    schema: DagsterPanderaXarraySchema | DagsterPanderaXarrayModel,
) -> dg.DagsterType:
    name = _extract_name_from_pandera_schema(schema)
    norm_schema = (
        schema.to_schema()
        if isinstance(schema, type) and issubclass(schema, VALID_MODEL_CLASSES)
        else schema
    )
    assert isinstance(norm_schema, VALID_SCHEMA_CLASSES)
    metadata = _pandera_schema_to_metadata_value(norm_schema)
    type_check_fn = _pandera_schema_to_type_check_fn(norm_schema)
    typing_type = xr.DataArray if isinstance(schema, pa.DataArraySchema) else xr.Dataset
    return dg.DagsterType(
        type_check_fn=type_check_fn,
        name=name,
        description=norm_schema.description,
        metadata={"schema": metadata},
        typing_type=typing_type,
    )


def _extract_name_from_pandera_schema(
    schema: DagsterPanderaXarraySchema | DagsterPanderaXarrayModel,
) -> str:
    if isinstance(schema, type) and issubclass(schema, VALID_MODEL_CLASSES):
        return str(
            getattr(schema.Config, "title", None)
            or getattr(schema.Config, "name", None)
            or schema.__name__
        )
    elif isinstance(schema, VALID_SCHEMA_CLASSES):
        return str(
            schema.title or schema.name or next(_anonymous_schema_name_generator)
        )
    return next(_anonymous_schema_name_generator)


_anonymous_schema_name_generator = (
    f"DagsterPanderaXarray{i}" for i in itertools.count(start=1)
)


def _pandera_schema_to_type_check_fn(
    schema: DagsterPanderaXarraySchema,
) -> Callable[[dg.TypeCheckContext, object], dg.TypeCheck]:
    def type_check_fn(_context, value: object) -> dg.TypeCheck:
        if isinstance(value, VALID_XARRAY_CLASSES):
            try:
                if isinstance(schema, pa.DataArraySchema):
                    da = check.inst(
                        value, xr.DataArray, "Must be a xarray DataArray."
                    )
                    # `lazy` instructs pandera to capture every (not just the first) validation error
                    schema.validate(da, lazy=True)
                elif isinstance(schema, pa.DatasetSchema):
                    ds = check.inst(
                        value,
                        xr.Dataset,
                        "Must be a xarray Dataset.",
                    )
                    schema.validate(ds, lazy=True)
                else:
                    check.failed(
                        f"Unexpected schema/value type combination: {type(schema).__name__} / {type(value).__name__}"
                    )
            except pa_errors.SchemaErrors as error:
                return _pandera_errors_to_type_check(error)
            except Exception as error:
                return dg.TypeCheck(
                    success=False,
                    description=f"Unexpected error during validation: {error}",
                )
        else:
            return dg.TypeCheck(
                success=False,
                description=(
                    f"Must be one of {VALID_XARRAY_CLASSES},"
                    f" got {type(value).__name__}."
                ),
            )

        return dg.TypeCheck(success=True)

    return type_check_fn


def _pandera_errors_to_type_check(
    error: pa_errors.SchemaErrors,
) -> dg.TypeCheck:
    # TODO: (mike) add metadata to describe the error in the UI
    return dg.TypeCheck(
        success=False,
        description=str(error),
    )


def _pandera_schema_to_metadata_value(
    schema: DagsterPanderaXarraySchema,
) -> dg.MetadataValue:
    value = schema.to_json()
    if value is None:
        return dg.MetadataValue.null()
    # TODO: (mike) add markdown metadata to describe schema in the UI
    return dg.MetadataValue.json(json.loads(value))
