"""Class(es) for constructing training labels at runtime."""

from abc import ABC, abstractmethod
import logging
import torch

class Label(ABC):
    def __init__(self, key: str):
        self._key = key
        self._logger = logging.getLogger(self.__class__.__name__)

    @property
    def key(self) -> str:
        return self._key

    @abstractmethod
    def __call__(self, batch: dict) -> torch.Tensor:
        """Label-specific implementation."""




class Direction(Label):
    """Class for producing particle direction/pointing label."""

    def __init__(
        self,
        key: str = "direction",
        azimuth_key: str = "azimuth",
        zenith_key: str = "zenith",
    ):
        """Construct `Direction`.

        Args:
            key: The name of the field in `Data` where the label will be
                stored. That is, `batch[key] = label`.
            azimuth_key: The name of the pre-existing key in `batch` that will
                be used to access the azimiuth angle, used when calculating
                the direction.
            zenith_key: The name of the pre-existing key in `batch` that will
                be used to access the zenith angle, used when calculating the
                direction.
        """
        self._azimuth_key = azimuth_key
        self._zenith_key = zenith_key

        # Base class constructor
        super().__init__(key=key)

    def __call__(self, batch: dict) -> torch.Tensor:
        """Compute label for `batch`."""
        x = (torch.cos(batch[self._azimuth_key]) * torch.sin(
            batch[self._zenith_key]
        )).reshape(-1, 1)
        y = (torch.sin(batch[self._azimuth_key]) * torch.sin(
            batch[self._zenith_key]
        )).reshape(-1, 1)
        z = (torch.cos(batch[self._zenith_key])).reshape(-1, 1)
        return torch.cat((x, y, z), dim=1)


class Track(Label):
    """Class for producing NuMuCC label.

    Label is set to `1` if the event is a NuMu CC event, else `0`.
    """

    def __init__(
        self,
        key: str = "track",
        pid_key: str = "pid",
        interaction_key: str = "interaction_type",
    ):
        """Construct `Track` label.

        Args:
            key: The name of the field in `Data` where the label will be
                stored. That is, `batch[key] = label`.
            pid_key: The name of the pre-existing key in `batch` that will
                be used to access the pdg encoding, used when calculating
                the direction.
            interaction_key: The name of the pre-existing key in `batch` that
                will be used to access the interaction type (1 denoting CC),
                used when calculating the direction.
        """
        self._pid_key = pid_key
        self._int_key = interaction_key

        # Base class constructor
        super().__init__(key=key)

    def __call__(self, batch: dict) -> torch.Tensor:
        """Compute label for `batch`."""
        is_numu = torch.abs(batch[self._pid_key]) == 14
        is_cc = batch[self._int_key] == 1
        return (is_numu & is_cc).type(torch.int)


class Neutrino(Label):
    """Class for producing Neutrino label.

    Label is set to `1` if the event is a Neutrino event, else `0`.
    """

    def __init__(
        self,
        key: str = "neutrino",
        pid_key: str = "pid",
    ):
        """Construct `Neutrino` label.

        Args:
            key: The name of the field in `Data` where the label will be
                stored. That is, `batch[key] = label`.
            pid_key: The name of the pre-existing key in `batch` that will
                be used to access the pdg encoding, used when calculating
                the direction.
        """
        self._pid_key = pid_key

        # Base class constructor
        super().__init__(key=key)

    def __call__(self, batch: dict) -> torch.Tensor:
        """Compute label for `batch`."""
        # PDG code 13 = muon; everything else is treated as neutrino
        is_neutrino = torch.abs(batch[self._pid_key]) != 13
        return is_neutrino.type(torch.int)


class Muon(Label):
    """Class for producing Neutrino label.

    Label is set to `1` if the event is a Muon event, else `0`.
    """

    def __init__(
        self,
        key: str = "muon",
        pid_key: str = "pid",
    ):
        """Construct `Muon` label.

        Args:
            key: The name of the field in `Data` where the label will be
                stored. That is, `batch[key] = label`.
            pid_key: The name of the pre-existing key in `batch` that will
                be used to access the pdg encoding, used when calculating
                the direction.
        """
        self._pid_key = pid_key

        # Base class constructor
        super().__init__(key=key)

    def __call__(self, batch: dict) -> torch.Tensor:
        """Compute label for `batch`."""
        # PDG code 13 = muon
        is_muon = torch.abs(batch[self._pid_key]) == 13
        return is_muon.type(torch.int)


class Position(Label):
    """Class for producing particle direction/pointing label."""

    def __init__(
        self,
        key: str = "position",
        vrx_x_key: str = "pos_x",
        vrx_y_key: str = "pos_y",
        vrx_z_key: str = "pos_z",
    ):
        """Construct `Position`.

        Args:
            key: The name of the field in `Data` where the label will be
                stored. That is, `batch[key] = label`.
            vrx_x_key: The name of the pre-existing key in `batch` that will
                be used to access the interaction vertex x-position.
            vrx_y_key: The name of the pre-existing key in `batch` that will
                be used to access the interaction vertex y-position.
            vrx_z_key: The name of the pre-existing key in `batch` that will
                be used to access the interaction vertex z-position.
        """
        self._vrx_x_key = vrx_x_key
        self._vrx_y_key = vrx_y_key
        self._vrx_z_key = vrx_z_key

        # Base class constructor
        super().__init__(key=key)

    def __call__(self, batch: dict) -> torch.Tensor:
        """Compute label for `batch`."""
        x = batch[self._vrx_x_key].reshape(-1, 1)
        y = batch[self._vrx_y_key].reshape(-1, 1)
        z = batch[self._vrx_z_key].reshape(-1, 1)

        return torch.cat((x, y, z), dim=1)
