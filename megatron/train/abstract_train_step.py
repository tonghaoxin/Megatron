""" Abstract class for batching, forward pass and loss handler. """
from abc import ABC
class AbstractTrainStep(ABC):
    """Abstract class for batching, forward pass and loss handler."""

    def __init__(self, name):
        """ initialize the class. """
        super().__init__()
        self.name = name
        self.model_output_class = None

    def get_batch_func(self):
        """ get batch function. """
        pass

    def get_forward_step_func(self):
        """ get forward function. """
        pass

    def get_loss_func(self):
        """ get loss function. """
        pass