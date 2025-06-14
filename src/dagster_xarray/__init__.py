import itertools
from collections.abc import Callable

import dagster as dg
import xarray as xr
from xarray_jsonschema import (
    DataArraySchema,
    DatasetSchema,
    SchemaError,
    ValidationError,
    XarraySchema,
)

from .xarray_zarr_io_manager import XarrayZarrIOManager


__all__ = [
    'DataArrayDagsterType',
    'DatasetDagsterType',
    'XarrayZarrIOManager',
    'xarray_schema_to_dagster_type',
]

VALID_XARRAY_CLASSES = (xr.DataArray, xr.Dataset)

DatasetDagsterType = dg.PythonObjectDagsterType(
    python_type=xr.Dataset,
    name='DatasetDagsterType',
    description='A multi-dimensional, in memory, array database.',
)

DataArrayDagsterType = dg.PythonObjectDagsterType(
    python_type=xr.DataArray,
    name='DataArrayDagsterType',
    description='N-dimensional array with labeled coordinates and dimensions.',
)


def xarray_schema_to_dagster_type(
    schema: DatasetSchema | DataArraySchema,
) -> dg.DagsterType:
    """Convert a xarray schema to a ``DagsterType``.

    The generated Dagster type will be given an automatically generated name:
    the schema's ``key`` attribute or ``title`` attribute. If both ``key``
    and ``title`` are not defined, a unique name of the form
    `DagsterXarray[DataArray|Dataset]<n>` will be generated.

    Metadata is also extracted from the schema and included in the Dagster
    type as a metadata dictionary. The extracted metadata includes:

    - A JSON representation of the schema.

    The returned `DagsterType` type will call the schema's `validate()` method
    in its type check function. If validation fails, the returned `TypeCheck`
    object will contain metadata about the error.

    Parameters
    ----------
    schema : DatasetSchema | DataArraySchema
        The xarray schema to convert.

    Returns
    -------
        DagsterType : Dagster Type constructed from the Xarray schema.
    """
    name = _extract_name_from_xarray_schema(schema)
    description = schema.description
    type_check_fn = _xarray_schema_to_dagster_type_check_fn(schema)
    typing_type = (
        xr.Dataset if isinstance(schema, DatasetSchema) else xr.DataArray
    )
    return dg.DagsterType(
        name=name,
        description=description,
        type_check_fn=type_check_fn,
        typing_type=typing_type,
        metadata={'schema': _xarray_schema_to_metadata_value(schema)},
    )


# call next() on these to generate a unique Dagster Type name for anonymous schemas
_anonymous_data_array_schema_name_generator = (
    f'DagsterXarrayDataArray{i}' for i in itertools.count(start=1)
)
_anonymous_dataset_schema_name_generator = (
    f'DagsterXarrayDataset{i}' for i in itertools.count(start=1)
)


def _extract_name_from_xarray_schema(schema: XarraySchema) -> str:
    """Return either the schema's title or a unique name for anonymous schemas."""
    name = schema.key or schema.title
    if name is None:
        match schema:
            case DatasetSchema():
                name = next(_anonymous_dataset_schema_name_generator)
            case DataArraySchema():
                name = next(_anonymous_data_array_schema_name_generator)
            case _:
                raise ValueError(f'Unexpected schema type: {type(schema)}')
    return name


def _xarray_schema_to_dagster_type_check_fn(
    schema: DatasetSchema | DataArraySchema,
) -> Callable[[dg.TypeCheckContext, object], dg.TypeCheck]:
    def _type_check_fn(
        _context: dg.TypeCheckContext,
        value: object,
    ) -> dg.TypeCheck:
        if isinstance(value, VALID_XARRAY_CLASSES):
            try:
                schema.validate(value)
            except (SchemaError, ValidationError) as error:
                type_check = _xarray_jsonschema_errors_to_type_check(error)
            except Exception as error:
                type_check = dg.TypeCheck(
                    success=False,
                    description=f'Unexpected error during validation: {error}',
                )
            else:
                type_check = dg.TypeCheck(success=True)
            return type_check
        else:
            return dg.TypeCheck(
                success=False,
                description=(
                    f'Expected one of {VALID_XARRAY_CLASSES},'
                    f' got {type(value).__name__}.'
                ),
            )

    return _type_check_fn


def _xarray_jsonschema_errors_to_type_check(
    error: ValidationError | SchemaError,
) -> dg.TypeCheck:
    """Convert a JSON schema error to a Dagster TypeCheck object."""
    # TODO: (mike) add metadata to describe the error in the UI
    return dg.TypeCheck(
        success=False,
        description=str(error),
    )


def _xarray_schema_to_metadata_value(
    schema: XarraySchema,
) -> dg.JsonMetadataValue:
    # TODO: (mike) Would be cool to have a Dagster integration like
    #  `dg.TableSchema, dg.TableColumn`... As a stopgap, we could create a
    #  markdown representation and pass it to DagsterType.description, which
    #  accepts markdown-formatted strings.
    return dg.MetadataValue.json(schema.to_dict())
