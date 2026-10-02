import itertools
import json
from collections.abc import Callable

import dagster as dg
import dagster._check as check
import pandera.errors as pa_errors
import pandera.xarray as pa
import xarray as xr

type DagsterPanderaXarraySchema = pa.DataArraySchema | pa.DatasetSchema
type DagsterPanderaXarrayModel = type[pa.DataArrayModel | pa.DatasetModel]

VALID_SCHEMA_CLASSES = (pa.DataArraySchema, pa.DatasetSchema)
VALID_MODEL_CLASSES = (pa.DataArrayModel, pa.DatasetModel)


def pandera_schema_to_dagster_type(
    schema: DagsterPanderaXarraySchema | DagsterPanderaXarrayModel,
) -> dg.DagsterType:
    if isinstance(schema, type) and issubclass(schema, VALID_MODEL_CLASSES):
        fallback_name = str(
            getattr(schema.Config, "title", None)
            or getattr(schema.Config, "name", None)
            or schema.__name__
        )
        norm_schema = schema.to_schema()
    else:
        fallback_name = None
        norm_schema = schema
    norm_schema = check.inst(norm_schema, VALID_SCHEMA_CLASSES)
    name = _extract_name_from_schema(norm_schema, fallback_name)
    metadata = _pandera_schema_to_metadata_value(norm_schema)
    type_check_fn = _pandera_schema_to_type_check_fn(norm_schema)
    typing_type = (
        xr.DataArray
        if isinstance(norm_schema, pa.DataArraySchema)
        else xr.Dataset
    )
    return dg.DagsterType(
        type_check_fn=type_check_fn,
        name=name,
        description=norm_schema.description,
        metadata={"schema": metadata},
        typing_type=typing_type,
    )


def _extract_name_from_schema(
    schema: DagsterPanderaXarraySchema,
    fallback_name: str | None,
) -> str:
    title = schema.title or schema.name
    if title:
        # `to_json()` only serializes `title`, so stamp `name` onto it to
        # keep the metadata in sync with the type name.
        if not schema.title:
            schema.title = str(title)
        return str(title)
    if fallback_name is not None:
        # `to_schema()` drops the model's `Config.title`, so stamp it back
        # onto the schema to keep the metadata in sync with the type name.
        schema.title = fallback_name
        return fallback_name
    return next(_anonymous_schema_name_generator)


_anonymous_schema_name_generator = (
    f"DagsterPanderaXarray{i}" for i in itertools.count(start=1)
)


def _pandera_schema_to_type_check_fn(
    schema: DagsterPanderaXarraySchema,
) -> Callable[[dg.TypeCheckContext, object], dg.TypeCheck]:
    def type_check_fn(_context, value: object) -> dg.TypeCheck:
        try:
            if isinstance(schema, pa.DataArraySchema):
                if not isinstance(value, xr.DataArray):
                    return _wrong_kind_type_check(value, expected=xr.DataArray)
                # `lazy` instructs pandera to capture every (not just the
                # first) validation error
                schema.validate(value, lazy=True)
            elif isinstance(schema, pa.DatasetSchema):
                if not isinstance(value, xr.Dataset):
                    return _wrong_kind_type_check(value, expected=xr.Dataset)
                schema.validate(value, lazy=True)
            else:
                check.failed(
                    f"Unexpected schema type: {type(schema).__name__}"
                )
        except pa_errors.SchemaErrors as error:
            return _pandera_errors_to_type_check(error)
        except Exception as error:  # noqa: BLE001
            return dg.TypeCheck(
                success=False,
                description=f"Unexpected error during validation: {error}",
            )
        return dg.TypeCheck(success=True)

    return type_check_fn


def _wrong_kind_type_check(value: object, expected: type) -> dg.TypeCheck:
    return dg.TypeCheck(
        success=False,
        description=f"Must be a xarray {expected.__name__},"
        f" got {type(value).__name__}.",
    )


def _pandera_errors_to_type_check(
    error: pa_errors.SchemaErrors,
) -> dg.TypeCheck:
    # TODO: (mike) add metadata to describe the error in the UI
    return dg.TypeCheck(success=False, description=str(error))


def _pandera_schema_to_metadata_value(
    schema: DagsterPanderaXarraySchema,
) -> dg.MetadataValue:
    value = schema.to_json()
    if value is None:
        return dg.MetadataValue.null()
    # TODO: (mike) add markdown metadata to describe schema in the UI
    return dg.MetadataValue.json(json.loads(value))
