from abc import ABC, abstractmethod
from typing import Iterable
from saf import Sentence, Token, Annotable


class Annotator(ABC):
    default_model = None

    @abstractmethod
    def annotate(self, items: Iterable[Annotable]):
        pass
