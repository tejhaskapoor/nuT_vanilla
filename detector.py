from abc import abstractmethod
from typing import Dict, Callable, Optional, Tuple, List, Union
import logging, os
import torch, torch.nn as nn
import pandas as pd
import torch
import os
import math
from .constants import ICECUBE_GEOMETRY_TABLE_DIR, KM3NeT_GEOMETRY_TABLE_DIR, PROMETHEUS_GEOMETRY_TABLE_DIR


class Detector(nn.Module):
    """Abstract base class for detector feature standardization.

    Each concrete subclass defines a ``feature_map()`` that returns a dict
    mapping feature column names to callables (normalization functions).
    The ``forward()`` method applies these functions column-by-column to the
    raw input tensor, returning a standardized tensor of the same shape.

    To skip standardization for a specific feature (e.g. during transfer
    learning), pass its name in ``replace_with_identity``.
    """

    def __init__(self, replace_with_identity: Optional[List[str]] = None) -> None:
        super().__init__()
        self._logger = logging.getLogger(self.__class__.__name__)
        self._replace_with_identity = replace_with_identity

    @abstractmethod
    def feature_map(self) -> Dict[str, Callable]: ...
    """List of features used/assumed by inheriting `Detector` objects."""

    def forward(self, input_features: torch.Tensor, input_feature_names: List[str]) -> torch.Tensor:
        return self._standardize(input_features, input_feature_names)

    def _standardize(self, input_features: torch.Tensor, input_feature_names: List[str]) -> torch.Tensor:
        fmap = self.feature_map()
        if self._replace_with_identity is not None:
            for f in self._replace_with_identity:
                fmap[f] = self._identity
        for idx, feature in enumerate(input_feature_names):
            try:
                input_features[:, idx] = fmap[feature](input_features[:, idx])
            except KeyError:
                self._logger.warning(f"No standardization function for '{feature}'")
                raise
        return input_features

    def _identity(self, x: torch.Tensor) -> torch.Tensor:
        return x

    @property
    def geometry_table(self) -> pd.DataFrame:
        """Public get method for retrieving a `Detector`s geometry table."""
        if not hasattr(self, '_geometry_table'):
            path = getattr(self, 'geometry_table_path', None)
            if path is None:
                raise AttributeError(f"{self.__class__.__name__} has no geometry_table_path.")
            self._geometry_table = pd.read_parquet(path)
        return self._geometry_table

    @property
    def string_index_name(self) -> str:
        return self.string_id_column

    @property
    def sensor_index_name(self) -> str:
        return self.sensor_id_column

    @property
    def sensor_position_names(self) -> List[str]:
        return self.xyz




################DETECTORS ORCA FILE##################################

class ORCA115_graphnet(Detector):
    """`Detector` class for ORCA-115 (legacy; kept for backward compatibility)."""

    geometry_table_path = os.path.join(
        ICECUBE_GEOMETRY_TABLE_DIR, "icecube86.parquet"
    )
    xyz = ["pos_x", "pos_y", "pos_z"]
    string_id_column = "string_id"
    sensor_id_column = "sensor_id"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension of input data."""
        feature_map = {
            "t": self._dom_time,
            "pos_x": self._dom_x,
            "pos_y": self._dom_y,
            "pos_z": self._dom_z,
            "dir_x": self._dir_x,
            "dir_y": self._dir_y,
            "dir_z": self._dir_z,
            "tot": self._tot,
        }
        return feature_map

    def _dom_x(self, x: torch.Tensor) -> torch.Tensor:
        return x / 100.0

    def _dom_y(self, x: torch.Tensor) -> torch.Tensor:
        return (x + 5) / 100.0  # +5 shifts origin to detector centre

    def _dom_z(self, x: torch.Tensor) -> torch.Tensor:
        # Normalise to [0, 1] over the vertical detector span (~163 m)
        return (x - 40 + 3) / (200 - 40 + 3)

    def _dom_time(self, x: torch.Tensor) -> torch.Tensor:
        return x / 2500.0  # scale to O(1) units (max event window ~2500 ns)

    def _tot(self, x: torch.Tensor) -> torch.Tensor:
        return x / 256.0  # time-over-threshold is 8-bit (0–255)

    def _dir_x(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def _dir_y(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def _dir_z(self, x: torch.Tensor) -> torch.Tensor:
        return x
    
class ORCA(Detector):
    """`Detector` class for ORCA."""

    def __init__(
        self, 
        ORCA115_norm: bool,
        configuration: Optional[Union[None, str]] = None, 
        du_selection: Optional[Union[None, Tuple[List[int], List[int]]]] = None,
        shift_coordinates: Optional[Union[None, bool]] = None,
    ) -> None:
        super().__init__()
        self.norm = ORCA115_norm
        if configuration is not None:
            self.configuration = os.path.join(KM3NeT_GEOMETRY_TABLE_DIR, configuration)
        if du_selection is not None:
            self.du_selection_ORCA115 = du_selection[0]
            self.du_selection_ORCAX = du_selection[1]
        if shift_coordinates is not None:
            self.shift = shift_coordinates
            if self.shift:
                self.x_shift, self.y_shift, self.z_shift = self._shift_to_ORCA115()
        else:
            self.shift = False
        
        self.geometry_table_path = os.path.join(
            KM3NeT_GEOMETRY_TABLE_DIR, "ORCA115.parquet"
        )
    
    geometry_table_path = os.path.join(
            KM3NeT_GEOMETRY_TABLE_DIR, "ORCA115.parquet"
    )
    xyz = ["pos_x", "pos_y", "pos_z"]
    string_id_column = "du_id"
    floor_id_column = "floor_id"
    sensor_id_column = "dom_id"
    
    def _shift_to_ORCA115(self):
        """
            Given a parquet detector file, it will shift those coordinate
            to ORCA115 coordinates. It needs to be provided:
                - Parquet detector file of a given configuration ORCAX.
                - DU ids in ORCA115.
                - DU ids in ORCAX.
        """
        
        ORCA115_df = pd.read_parquet(self.geometry_table_path)
        ORCAX_df = pd.read_parquet(self.configuration)
        
        ORCA115_DUs = ORCA115_df[ORCA115_df['du_id'].isin(self.du_selection_ORCA115)]
        ORCAX_DUs = ORCAX_df[ORCAX_df['du_id'].isin(self.du_selection_ORCAX)]
        
        x_shift = ORCAX_DUs['pos_x'].mean() - ORCA115_DUs['pos_x'].mean()
        y_shift = ORCAX_DUs['pos_y'].mean() - ORCA115_DUs['pos_y'].mean()
        z_shift = ORCAX_DUs['pos_z'].mean() - ORCA115_DUs['pos_z'].mean()
        
        return (x_shift, y_shift, z_shift)
    
    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension of input data."""
        
        feature_map = {
            "t": self._dom_time,
            "pos_x": self._dom_x,
            "pos_y": self._dom_y,
            "pos_z": self._dom_z,
            "dir_x": self._dir_x,
            "dir_y": self._dir_y,
            "dir_z": self._dir_z,
            "tot": self._tot,
            "trig": self._identity,
            "channel_id": self._identity,
            "dom_id": self._identity,
            "du_id": self._identity,
        }
        
        return feature_map
    
    def _dom_x(self, x: torch.tensor) -> torch.tensor:
        if self.shift:
            x = x - self.x_shift
        if self.norm:
            x = x / 100.0
        return x
        
    def _dom_y(self, x: torch.tensor) -> torch.tensor:
        if self.shift:
            x = x - self.y_shift
        if self.norm:
            x = (x + 5) / 100.0
        return x

    def _dom_z(self, x: torch.tensor) -> torch.tensor:
        if self.shift:
            x = x - self.z_shift
        if self.norm:
            x = ( x - 40 + 3) / (200 - 40 + 3)
        return x

    def _dom_time(self, x: torch.tensor) -> torch.tensor:
        if self.norm:
            x = x / 2500.0
        return x
    
    def _tot(self, x: torch.tensor) -> torch.tensor:
        if self.norm:
            x = x / 256.0
        return x

    def _dir_x(self, x: torch.tensor) -> torch.tensor:
        return x
    
    def _dir_y(self, x: torch.tensor) -> torch.tensor:
        return x
    
    def _dir_z(self, x: torch.tensor) -> torch.tensor:
        return x

    
    
class ORCA115(Detector):
    """`Detector` class for ORCA-115."""
    def __init__(
        self, 
        raw
    ) -> None:
        super().__init__()
        self.raw = raw
    
    geometry_table_path = os.path.join(
        ICECUBE_GEOMETRY_TABLE_DIR, "icecube86.parquet"
    )
    xyz = ["pos_x", "pos_y", "pos_z"]
    string_id_column = "string_id"
    sensor_id_column = "sensor_id"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension of input data."""
        feature_map = {
            "t": self._identity,
            "pos_x": self._identity,
            "pos_y": self._identity,
            "pos_z": self._identity,
            "dir_x": self._identity,
            "dir_y": self._identity,
            "dir_z": self._identity,
            "tot": self._identity,
            "trig": self._identity,
            "du_id": self._identity,
            "channel_id": self._identity,
            "dom_id": self._identity,
        }
        return feature_map
    
    
class ORCA_dev(Detector):
    """`Detector` class for ORCA."""

    def __init__(
        self, 
        configuration: Optional[str], 
        du_selection: Optional[Tuple[List[int], List[int]]],
        shift_coordinates: Optional[bool] = True,
    ) -> None:
        super().__init__()
        self.configuration = os.path.join(KM3NeT_GEOMETRY_TABLE_DIR, configuration)
        self.du_selection_ORCA115 = du_selection[0]
        self.du_selection_ORCAX = du_selection[1]
        self.shift = shift_coordinates
        
        self.geometry_table_path = os.path.join(
            KM3NeT_GEOMETRY_TABLE_DIR, "ORCA115.parquet"
        )
        
        if self.shift:
            self.x_shift, self.y_shift, self.z_shift = self._shift_to_ORCA115()
    
    geometry_table_path = os.path.join(
            KM3NeT_GEOMETRY_TABLE_DIR, "ORCA115.parquet"
    )
    xyz = ["pos_x", "pos_y", "pos_z"]
    string_id_column = "DU_id"
    floor_id_column = "floor_id"
    sensor_id_column = "dom_id"
    
    def _shift_to_ORCA115(self):
        ORCA115_df = pd.read_parquet(self.geometry_table_path)
        ORCAX_df = pd.read_parquet(self.configuration)
        
        ORCA115_DUs = ORCA115_df[ORCA115_df['DU_id'].isin(self.du_selection_ORCA115)]
        ORCAX_DUs = ORCAX_df[ORCAX_df['DU_id'].isin(self.du_selection_ORCAX)]
        
        x_shift = ORCAX_DUs['pos_x'].mean() - ORCA115_DUs['pos_x'].mean()
        y_shift = ORCAX_DUs['pos_y'].mean() - ORCA115_DUs['pos_y'].mean()
        z_shift = ORCAX_DUs['pos_z'].mean() - ORCA115_DUs['pos_z'].mean()
        
        return (x_shift, y_shift, z_shift)
    
    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension of input data."""
        
        feature_map = {
            "t": self._dom_time,
            "pos_x": self._dom_x,
            "pos_y": self._dom_y,
            "pos_z": self._dom_z,
            "dir_x": self._dir_xy,
            "dir_y": self._dir_xy,
            "dir_z": self._dir_z,
            "tot": self._tot,
            "trig": self._identity,
            "channel_id": self._identity,
            "dom_id": self._identity,
            "du_id": self._identity,
        }
        
        return feature_map
    
    def _dom_x(self, x: torch.tensor) -> torch.tensor:
        if self.shift:
            x = x - self.x_shift
        return x / 10.0
    
    def _dom_y(self, x: torch.tensor) -> torch.tensor:
        if self.shift:
            x = x - self.y_shift
        return x / 10.0

    def _dom_z(self, x: torch.tensor) -> torch.tensor:
        if self.shift:
            x = x - self.z_shift
        return (x - 117.5) / 7.75

    def _dom_time(self, x: torch.tensor) -> torch.tensor:
        return (x - 1800) / 180

    def _tot(self, x: torch.tensor) -> torch.tensor:
        return (x - 75) / 7.5

    def _dir_xy(self, x: torch.tensor) -> torch.tensor:
        return x * 10.0

    def _dir_z(self, x: torch.tensor) -> torch.tensor:
        return (x + 0.275) * 12.9


class ORCA6(Detector):
    """`Detector` class for ORCA-6."""

    geometry_table_path = os.path.join(
        ICECUBE_GEOMETRY_TABLE_DIR, "icecube86.parquet"
    )
    xyz = ["pos_x", "pos_y", "pos_z"]
    string_id_column = "string_id"
    sensor_id_column = "sensor_id"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension of input data."""
        feature_map = {
            "t": self._dom_time,
            "pos_x": self._dom_x,
            "pos_y": self._dom_y,
            "pos_z": self._dom_z,
            "dir_x": self._dir_xy,
            "dir_y": self._dir_xy,
            "dir_z": self._dir_z,
            "tot": self._tot,
        }
        return feature_map

    def _dom_x(self, x: torch.tensor) -> torch.tensor:
        return (x - 457.8) * 0.37

    def _dom_y(self, x: torch.tensor) -> torch.tensor:
        return (x - 574.1) * 1.04

    def _dom_z(self, x: torch.tensor) -> torch.tensor:
        return (x - 108.6) * 0.12

    def _dom_time(self, x: torch.tensor) -> torch.tensor:
        return (x - 1025) * 0.021

    def _tot(self, x: torch.tensor) -> torch.tensor:
        # return torch.log10(x)
        return (x - 117) * 0.085

    def _dir_xy(self, x: torch.tensor) -> torch.tensor:
        return x * 10.0

    def _dir_z(self, x: torch.tensor) -> torch.tensor:
        return (x + 0.23) * 12.9

class ORCA6_2_ORCA115(Detector):
    """`Detector class for ORCA-6 in ORCA115 coordinates."""

    geometry_table_path = os.path.join(
        ICECUBE_GEOMETRY_TABLE_DIR, "icecube86.parquet"
    )
    xyz = ["pos_x", "pos_y", "pos_z"]
    string_id_column = "string_id"
    sensor_id_column = "sensor_id"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension of input data."""
        feature_map = {
            "t": self._identity,
            "pos_x": self._shift_x_pos,
            "pos_y": self._shift_y_pos,
            "pos_z": self._shift_z_pos,
            "dir_x": self._identity,
            "dir_y": self._identity, 
            "dir_z": self._identity, 
            "tot": self._identity,
            "trig": self._identity,
            "channel_id": self._identity,
            "dom_id": self._identity,
            "du_id": self._identity,
            }
        return feature_map

    def _shift_x_pos(self, x: torch.tensor) -> torch.tensor:
        ORCA6_x_center = 458.41666159134473
        ORCA115_6_x_center = -50.8455
        return x - (ORCA6_x_center - ORCA115_6_x_center)

    def _shift_y_pos(self, x: torch.tensor) -> torch.tensor:
        ORCA6_y_center = 574.716668752937
        ORCA115_6_y_center = 94.01083333333335
        return x - (ORCA6_y_center - ORCA115_6_y_center)

    def _shift_z_pos(self, x: torch.tensor) -> torch.tensor:
        ORCA6_z_center = 113.86196451110592
        ORCA115_6_z_center = 117.1999120916717
        return x - (ORCA6_z_center - ORCA115_6_z_center)


class ORCA10_2_ORCA115(Detector):
    """`Detector class for ORCA-10 in ORCA115 coordinates."""

    geometry_table_path = os.path.join(
        ICECUBE_GEOMETRY_TABLE_DIR, "icecube86.parquet"
    )
    xyz = ["pos_x", "pos_y", "pos_z"]
    string_id_column = "string_id"
    sensor_id_column = "sensor_id"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension of input data."""
        feature_map = {
            "t": self._identity,
            "pos_x": self._shift_x_pos,
            "pos_y": self._shift_y_pos,
            "pos_z": self._shift_z_pos,
            "dir_x": self._identity,
            "dir_y": self._identity,
            "dir_z": self._identity,
            "tot": self._identity,
            }
        return feature_map

    def _shift_x_pos(self, x: torch.tensor) -> torch.tensor:
        ORCA10_x_center = 461.57000323
        ORCA115_10_x_center = -47.878099999999996
        return x - (ORCA10_x_center - ORCA115_10_x_center)

    def _shift_y_pos(self, x: torch.tensor) -> torch.tensor:
        ORCA10_y_center = 565.32
        ORCA115_10_y_center = 85.05930000000002
        return x - (ORCA10_y_center - ORCA115_10_y_center)

    def _shift_z_pos(self, x: torch.tensor) -> torch.tensor:
        ORCA10_z_center = 108.53536254
        ORCA115_10_z_center = 117.19991209167169
        return x - (ORCA10_z_center - ORCA115_10_z_center)



######################### PROMETHEUS DETECTOR #########################




class ORCA150SuperDense(Detector):
    """`Detector` class for Prometheus ORCA150SuperDense."""

    geometry_table_path = os.path.join(
        PROMETHEUS_GEOMETRY_TABLE_DIR, "orca_150.parquet"
    )
    xyz = ["sensor_pos_x", "sensor_pos_y", "sensor_pos_z"]
    string_id_column = "sensor_string_id"
    sensor_id_column = "sensor_id"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension."""
        feature_map = {
            "sensor_pos_x": self._sensor_pos_xy,
            "sensor_pos_y": self._sensor_pos_xy,
            "sensor_pos_z": self._sensor_pos_z,
            "t": self._t,
        }
        return feature_map

    def _sensor_pos_xy(self, x: torch.tensor) -> torch.tensor:
        return x / 100

    def _sensor_pos_z(self, x: torch.tensor) -> torch.tensor:
        return (x + 350) / 100

    def _t(self, x: torch.tensor) -> torch.tensor:
        return x / 1.05e04


class TRIDENT1211(Detector):
    """`Detector` class for Prometheus TRIDENT1211."""

    geometry_table_path = os.path.join(
        PROMETHEUS_GEOMETRY_TABLE_DIR, "trident.parquet"
    )
    xyz = ["sensor_pos_x", "sensor_pos_y", "sensor_pos_z"]
    string_id_column = "sensor_string_id"
    sensor_id_column = "sensor_id"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension."""
        feature_map = {
            "sensor_pos_x": self._sensor_pos_xy,
            "sensor_pos_y": self._sensor_pos_xy,
            "sensor_pos_z": self._sensor_pos_z,
            "t": self._t,
        }
        return feature_map

    def _sensor_pos_xy(self, x: torch.tensor) -> torch.tensor:
        return x / 1900

    def _sensor_pos_z(self, x: torch.tensor) -> torch.tensor:
        return x / 3000

    def _t(self, x: torch.tensor) -> torch.tensor:
        return x / 1.05e04


class IceCubeUpgrade7(Detector):
    """`Detector` class for Prometheus IceCubeUpgrade7."""

    geometry_table_path = os.path.join(
        PROMETHEUS_GEOMETRY_TABLE_DIR, "icecube_upgrade.parquet"
    )
    xyz = ["sensor_pos_x", "sensor_pos_y", "sensor_pos_z"]
    string_id_column = "sensor_string_id"
    sensor_id_column = "sensor_id"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension."""
        feature_map = {
            "sensor_pos_x": self._sensor_pos_xy,
            "sensor_pos_y": self._sensor_pos_xy,
            "sensor_pos_z": self._sensor_pos_z,
            "t": self._t,
        }
        return feature_map

    def _sensor_pos_xy(self, x: torch.tensor) -> torch.tensor:
        return x / 10

    def _sensor_pos_z(self, x: torch.tensor) -> torch.tensor:
        return x / 2000

    def _t(self, x: torch.tensor) -> torch.tensor:
        return x / 1.05e04


class WaterDemo81(Detector):
    """`Detector` class for Prometheus WaterDemo81."""

    geometry_table_path = os.path.join(
        PROMETHEUS_GEOMETRY_TABLE_DIR, "demo_water.parquet"
    )
    xyz = ["sensor_pos_x", "sensor_pos_y", "sensor_pos_z"]
    string_id_column = "sensor_string_id"
    sensor_id_column = "sensor_id"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension."""
        feature_map = {
            "sensor_pos_x": self._sensor_pos_xy,
            "sensor_pos_y": self._sensor_pos_xy,
            "sensor_pos_z": self._sensor_pos_z,
            "t": self._t,
        }
        return feature_map

    def _sensor_pos_xy(self, x: torch.tensor) -> torch.tensor:
        return x / 500

    def _sensor_pos_z(self, x: torch.tensor) -> torch.tensor:
        return x / 2000

    def _t(self, x: torch.tensor) -> torch.tensor:
        return x / 1.05e04


class BaikalGVD8(Detector):
    """`Detector` class for Prometheus BaikalGVD8."""

    geometry_table_path = os.path.join(
        PROMETHEUS_GEOMETRY_TABLE_DIR, "gvd.parquet"
    )
    xyz = ["sensor_pos_x", "sensor_pos_y", "sensor_pos_z"]
    string_id_column = "sensor_string_id"
    sensor_id_column = "sensor_id"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension."""
        feature_map = {
            "sensor_pos_x": self._sensor_pos_xy,
            "sensor_pos_y": self._sensor_pos_xy,
            "sensor_pos_z": self._sensor_pos_z,
            "t": self._t,
        }
        return feature_map

    def _sensor_pos_xy(self, x: torch.tensor) -> torch.tensor:
        return x / 10

    def _sensor_pos_z(self, x: torch.tensor) -> torch.tensor:
        return x / 1000

    def _t(self, x: torch.tensor) -> torch.tensor:
        return x / 1.05e04


class IceDemo81(Detector):
    """`Detector` class for Prometheus IceDemo81."""

    geometry_table_path = os.path.join(
        PROMETHEUS_GEOMETRY_TABLE_DIR, "demo_ice.parquet"
    )
    xyz = ["sensor_pos_x", "sensor_pos_y", "sensor_pos_z"]
    string_id_column = "sensor_string_id"
    sensor_id_column = "sensor_id"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension."""
        feature_map = {
            "sensor_pos_x": self._sensor_pos_xy,
            "sensor_pos_y": self._sensor_pos_xy,
            "sensor_pos_z": self._sensor_pos_z,
            "t": self._t,
        }
        return feature_map

    def _sensor_pos_xy(self, x: torch.tensor) -> torch.tensor:
        return x / 500

    def _sensor_pos_z(self, x: torch.tensor) -> torch.tensor:
        return x / 3000

    def _t(self, x: torch.tensor) -> torch.tensor:
        return x / 1.05e04


class ARCA115(Detector):
    """`Detector` class for Prometheus ARCA115."""

    geometry_table_path = os.path.join(
        PROMETHEUS_GEOMETRY_TABLE_DIR, "arca.parquet"
    )
    xyz = ["sensor_pos_x", "sensor_pos_y", "sensor_pos_z"]
    string_id_column = "sensor_string_id"
    sensor_id_column = "sensor_id"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension."""
        feature_map = {
            "sensor_pos_x": self._sensor_pos_xy,
            "sensor_pos_y": self._sensor_pos_xy,
            "sensor_pos_z": self._sensor_pos_z,
            "t": self._t,
        }
        return feature_map

    def _sensor_pos_xy(self, x: torch.tensor) -> torch.tensor:
        return x / 100

    def _sensor_pos_z(self, x: torch.tensor) -> torch.tensor:
        return x / 1000

    def _t(self, x: torch.tensor) -> torch.tensor:
        return x / 1.05e04


class ORCA150(Detector):
    """`Detector` class for Prometheus ORCA150."""

    geometry_table_path = os.path.join(
        PROMETHEUS_GEOMETRY_TABLE_DIR, "orca.parquet"
    )
    xyz = ["sensor_pos_x", "sensor_pos_y", "sensor_pos_z"]
    string_id_column = "sensor_string_id"
    sensor_id_column = "sensor_id"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension."""
        feature_map = {
            "sensor_pos_x": self._sensor_pos_xy,
            "sensor_pos_y": self._sensor_pos_xy,
            "sensor_pos_z": self._sensor_pos_z,
            "t": self._t,
        }
        return feature_map

    def _sensor_pos_xy(self, x: torch.tensor) -> torch.tensor:
        return x / 10

    def _sensor_pos_z(self, x: torch.tensor) -> torch.tensor:
        return x / 100

    def _t(self, x: torch.tensor) -> torch.tensor:
        return x / 1.05e04


class IceCube86Prometheus(Detector):
    """`Detector` class for Prometheus IceCube86."""

    geometry_table_path = os.path.join(
        PROMETHEUS_GEOMETRY_TABLE_DIR, "icecube86.parquet"
    )
    xyz = ["sensor_pos_x", "sensor_pos_y", "sensor_pos_z"]
    string_id_column = "sensor_string_id"
    sensor_id_column = "sensor_id"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension."""
        feature_map = {
            "sensor_pos_x": self._sensor_pos_xy,
            "sensor_pos_y": self._sensor_pos_xy,
            "sensor_pos_z": self._sensor_pos_z,
            "t": self._t,
        }
        return feature_map

    def _sensor_pos_xy(self, x: torch.tensor) -> torch.tensor:
        return x / 100

    def _sensor_pos_z(self, x: torch.tensor) -> torch.tensor:
        return x / 1000

    def _t(self, x: torch.tensor) -> torch.tensor:
        return x / 1.05e04


class IceCubeDeepCore8(Detector):
    """`Detector` class for Prometheus IceCubeDeepCore8."""

    geometry_table_path = os.path.join(
        PROMETHEUS_GEOMETRY_TABLE_DIR, "icecube_deepcore.parquet"
    )
    xyz = ["sensor_pos_x", "sensor_pos_y", "sensor_pos_z"]
    string_id_column = "sensor_string_id"
    sensor_id_column = "sensor_id"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension."""
        feature_map = {
            "sensor_pos_x": self._sensor_pos_xy,
            "sensor_pos_y": self._sensor_pos_xy,
            "sensor_pos_z": self._sensor_pos_z,
            "t": self._t,
        }
        return feature_map

    def _sensor_pos_xy(self, x: torch.tensor) -> torch.tensor:
        return x / 100

    def _sensor_pos_z(self, x: torch.tensor) -> torch.tensor:
        return x / 1000

    def _t(self, x: torch.tensor) -> torch.tensor:
        return x / 1.05e04


class IceCubeGen2(Detector):
    """`Detector` class for Prometheus IceCubeGen2."""

    geometry_table_path = os.path.join(
        PROMETHEUS_GEOMETRY_TABLE_DIR, "icecube_gen2.parquet"
    )
    xyz = ["sensor_pos_x", "sensor_pos_y", "sensor_pos_z"]
    string_id_column = "sensor_string_id"
    sensor_id_column = "sensor_id"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension."""
        feature_map = {
            "sensor_pos_x": self._sensor_pos_xyz,
            "sensor_pos_y": self._sensor_pos_xyz,
            "sensor_pos_z": self._sensor_pos_xyz,
            "t": self._t,
        }
        return feature_map

    def _sensor_pos_xyz(self, x: torch.tensor) -> torch.tensor:
        return x / 1000

    def _t(self, x: torch.tensor) -> torch.tensor:
        return x / 1.05e04


class PONETriangle(Detector):
    """`Detector` class for Prometheus PONE Triangle."""

    geometry_table_path = os.path.join(
        PROMETHEUS_GEOMETRY_TABLE_DIR, "pone_triangle.parquet"
    )
    xyz = ["sensor_pos_x", "sensor_pos_y", "sensor_pos_z"]
    string_id_column = "sensor_string_id"
    sensor_id_column = "sensor_id"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension."""
        feature_map = {
            "sensor_pos_x": self._sensor_pos_xyz,
            "sensor_pos_y": self._sensor_pos_xyz,
            "sensor_pos_z": self._sensor_pos_xyz,
            "t": self._t,
            "charge": self._charge,
            "string_id": self._identity,
            "is_signal": self._identity,
        }
        return feature_map

    def _sensor_pos_xyz(self, x: torch.tensor) -> torch.tensor:
        return x / 100

    def _t(self, x: torch.tensor) -> torch.tensor:
        return x / 1.05e04

    def _charge(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log10(1 + x)


class Prometheus(ORCA150SuperDense):
    """Reference to ORCA150SuperDense."""



#TK: Detector Class for Prometheus Detector
class PrometheusDetector(Detector):
    """ Generic Detector Class for the Paper."""

    geometry_table_path = os.path.join(
        PROMETHEUS_GEOMETRY_TABLE_DIR, "orca_150.parquet"
    )
    xyz = ["sensor_pos_x", "sensor_pos_y", "sensor_pos_z"]
    string_id_column = "sensor_string_id"
    sensor_id_column = "sensor_id"
    sensor_time_column = "t"
    charge_column = "charge"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension."""
        feature_map = {
            "sensor_pos_x": self._sensor_pos_xyz,
            "sensor_pos_y": self._sensor_pos_xyz,
            "sensor_pos_z": self._sensor_pos_xyz,
            "t": self._t,
            "charge": self._charge,
            "string_id": self._identity,
            "is_signal": self._identity
        }
        return feature_map

    def _sensor_pos_xyz(self, x: torch.Tensor) -> torch.Tensor:
        # Divide by detector size O(1000 m) to bring coordinates to O(1)
        return x / 1000

    def _t(self, x: torch.Tensor) -> torch.Tensor:
        # Divide by ~1 µs event window to bring time to O(1)
        return x / 10e5

    def _charge(self, x: torch.Tensor) -> torch.Tensor:
        # log1p compresses dynamic range of charge (PE counts span several decades)
        return torch.log10(1 + x)
