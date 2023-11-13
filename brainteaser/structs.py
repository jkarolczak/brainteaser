from __future__ import annotations

import random
from dataclasses import dataclass
from textwrap import dedent
from typing import Optional

import numpy as np

_sp_keys_map = {
    "distractor1": "distractor_1",
    "distractor2": "distractor_2",
    "distractor(unsure)": "distractor_unsure"
}


@dataclass(kw_only=True)
class Instance:
    question: str
    choice_list: list[str]
    embedding: Optional[list[float]] = None

    @staticmethod
    def from_dict(dictionary: dict[str, str | list[str]]) -> Instance:
        return Instance(**dictionary)

    def embed(self) -> None:
        from openai import OpenAI

        response = OpenAI().embeddings.create(input=self.question, model="text-embedding-ada-002")
        self.embedding = response.data[0].embedding

    def __repr__(self) -> str:
        return dedent(f"""\
                Question: {self.question}
                A: {self.choice_list[0]}
                B: {self.choice_list[1]}
                C: {self.choice_list[2]}
                D: {self.choice_list[3]}\
            """)


@dataclass(kw_only=True)
class TrainingInstance(Instance):
    id: str
    answer: str
    distractor_1: str
    distractor_2: str
    distractor_unsure: str
    label: int
    choice_order: list[int]

    @staticmethod
    def from_dict(dictionary: dict[str, int | str | list[int] | list[str]]) -> TrainingInstance:
        dictionary = dictionary.copy()
        dictionary["question"] = dictionary["question"].replace("\n", " ")
        for old, new in _sp_keys_map.items():
            dictionary[new] = dictionary[old]
            del dictionary[old]
        return TrainingInstance(**dictionary)

    @property
    def answer_idx(self) -> int:
        return self.choice_order.index(0)

    def is_choice_correct(self, idx: int) -> bool:
        return idx == self.answer_idx

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
        instance_class = TrainingInstance if "answer" in array[0].keys() else Instance
        return DataSet(instances=list(map(lambda x: instance_class.from_dict(x), array)), shuffle=shuffle)

    @staticmethod
    def from_list(list_: list[Instance], shuffle: bool = False) -> DataSet:
        return DataSet(instances=list_, shuffle=shuffle)

    @staticmethod
    def from_file(path: str, shuffle: bool = False) -> DataSet:
        if ".npy" in path:
            array = np.load(path, allow_pickle=True)
            return DataSet.from_array(array, shuffle)
        if ".pkl" in path:
            import cloudpickle as pickle

            with open(path, "rb") as fp:
                dataset = pickle.load(fp)
                dataset.shuffle = shuffle
                return dataset

    @staticmethod
    def to_file(dataset: DataSet, path: str = "./dataset.pkl") -> None:
        import cloudpickle as pickle

        if ".pkl" not in path:
            path += ".pkl"
        with open(path, "wb") as fp:
            pickle.dump(dataset, fp)

    def shuffle_instances(self) -> DataSet:
        random.shuffle(self.instances)
        return self

    @property
    def correct_answers(self) -> list[int]:
        try:
            return [instance.answer_idx for instance in self.instances]
        except:
            raise RuntimeError("A list of correct answers can be generated only for DataSet with TrainingInstances")

    def __iter__(self) -> DataSet:
        self._iter_idx = -1
        self._iter_instances = self.instances.copy()
        if self.shuffle:
            rabdom.shuffle(self._iter_instances)
        return self

    def __next__(self) -> TrainingInstance | Instance:
        self._iter_idx += 1
        if self._iter_idx >= self.__len__():
            raise StopIteration
        return self._iter_instances[self._iter_idx]

    def __repr__(self) -> str:
        return "; ".join(map(lambda x: x.question.strip(), self.instances))

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, item: int) -> TrainingInstance | Instance:
        return self.instances[item]
