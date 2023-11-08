from __future__ import annotations

from dataclasses import dataclass
from random import shuffle
from textwrap import dedent

import numpy as np

_sp_keys_map = {
    "distractor1": "distractor_1",
    "distractor2": "distractor_2",
    "distractor(unsure)": "distractor_unsure"
}


@dataclass
class Instance:
    id: str
    question: str
    answer: str
    distractor_1: str
    distractor_2: str
    distractor_unsure: str
    label: int
    choice_list: list[str]
    choice_order: list[int]

    @staticmethod
    def from_dict(dictionary: dict[str, int | str | list[int] | list[str]]) -> Instance:
        dictionary = dictionary.copy()
        dictionary["question"] = dictionary["question"].replace("/n", " ")
        for old, new in _sp_keys_map.items():
            dictionary[new] = dictionary[old]
            del dictionary[old]
        return Instance(**dictionary)

    def __repr__(self) -> str:
        return dedent(f"""\
            Question: {self.question}
            A (correct): {self.answer}
            B: {self.distractor_1}
            C: {self.distractor_2}
            D (unsure): {self.distractor_unsure}\
        """)


@dataclass
class DataSet:
    instances: list[Instance]
    shuffle: bool = False

    @staticmethod
    def from_array(array: np.ndarray, shuffle: bool = False) -> DataSet:
        return DataSet(instances=list(map(lambda x: Instance.from_dict(x), array)), shuffle=shuffle)

    @staticmethod
    def from_file(path: str, shuffle: bool = False) -> DataSet:
        array = np.load(path, allow_pickle=True)
        return DataSet.from_array(array, shuffle)

    def __iter__(self) -> DataSet:
        self._iter_idx = -1
        self._iter_instances = self.instances.copy()
        if self.shuffle:
            shuffle(self._iter_instances)
        return self

    def __next__(self) -> Instance:
        self._iter_idx += 1
        if self._iter_idx >= self.__len__():
            raise StopIteration
        return self._iter_instances[self._iter_idx]

    def __repr__(self) -> str:
        return "; ".join(map(lambda x: x.question.strip(), self.instances))

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, item: int) -> Instance:
        return self.instances[item]
