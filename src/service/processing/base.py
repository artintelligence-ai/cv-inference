import abc

from allenai_common import Registrable


class BaseProcessor(abc.ABC, Registrable):
    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        pass
