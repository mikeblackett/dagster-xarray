from pathlib import Path
from typing import Any

import dagster as dg
import xarray as xr

CACHE_DIR = Path().cwd().joinpath("data/in")


class XarrayTutorialIOManager(dg.ConfigurableIOManager):
    def load_input(self, context: dg.InputContext) -> xr.Dataset:
        return xr.tutorial.open_dataset(
            context.asset_key.path[-1], cache_dir=CACHE_DIR
        )

    def handle_output(self, context: dg.OutputContext, obj: Any):
        raise NotImplementedError("xarray tutorial datasets are readonly.")
