from abc import ABC, abstractmethod
from torch import nn


class SubModule(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def get_input_dim(self) -> int:
        """
        Get the input dimension of the module.

        Returns:
            int: Input dimension.
        """
        pass

    def get_output_dim(self) -> int:
        """
        Get the output dimension of the module.

        Returns:
            int: Output dimension.
        """
        pass
