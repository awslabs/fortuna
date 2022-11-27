import abc

from fortuna.utils.builtins import HashableMixin


class Distribution(HashableMixin, abc.ABC):
    @abc.abstractmethod
    def sample(self, *args, **kwargs):
        pass

    def log_joint_prob(self, *args, **kwargs):
        pass
