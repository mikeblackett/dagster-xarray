# TODO: Use @multi_asset_check once there is a way to pass the asset output to the check function.
# See: https://github.com/dagster-io/dagster/issues/21772from collections.abc import Sequence

from collections.abc import Sequence

import dagster as dg
import xarray as xr


def validate_freq(
    obj: xr.DataArray | xr.Dataset, freq: str | Sequence[str]
) -> None:
    time = obj.get_index("time")

    inferred_freq = xr.infer_freq(time)
    if inferred_freq is None:
        raise ValueError("could not infer a frequency.")

    freq = [freq] if isinstance(freq, str) else freq
    if inferred_freq not in freq:
        raise ValueError(
            f"invalid frequency. Expected {freq!r}, got {inferred_freq!r}."
        )


def build_xarray_frequency_check(
    asset: dg.AssetKey | dg.AssetsDefinition | str,
    *,
    freq: str | Sequence[str],
) -> dg.AssetChecksDefinition:

    freq = [freq] if isinstance(freq, str) else list(freq)

    @dg.asset_check(
        name="check_freq",
        description=(
            "Check that the dataset has the correct temporal frequency."
        ),
        asset=asset,
        blocking=True,
    )
    def _check(
        context: dg.AssetCheckExecutionContext,
        obj: dg.PythonObjectDagsterType(
            python_type=(xr.DataArray, xr.Dataset)
        ),  # type: ignore (HACK: dagster complains about unions)
    ) -> dg.AssetCheckResult:
        try:
            validate_freq(obj, freq=freq)
            passed = True
            metadata = {"valid_freqs": ", ".join(freq)}
        except BaseException as error:
            passed = False
            metadata = {"error": str(error)}
        return dg.AssetCheckResult(
            passed=passed,
            severity=dg.AssetCheckSeverity.ERROR,
            metadata=metadata,
        )

    return _check
