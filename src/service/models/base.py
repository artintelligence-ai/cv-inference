import abc

from allenai_common import Registrable


class BaseModel(abc.ABC, Registrable):
    @abc.abstractmethod
    def forward(self, x):
        return NotImplementedError
