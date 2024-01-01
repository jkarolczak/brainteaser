import re
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
from scipy.spatial.distance import cosine

from structs import DataSet, Instance, TrainingInstance


class Solver(ABC):
    def solve(self, x: DataSet | Instance, show_progress: bool = True) -> int | list[int]:
        if isinstance(x, Instance):
            return self.solve_instance(x)
        if isinstance(x, DataSet):
            if show_progress:
                from tqdm.auto import tqdm

                return [self.solve_instance(i) for i in tqdm(x)]
            return [self.solve_instance(i) for i in x]
        raise TypeError("An input has to be an Instance or a DataSet")

    @abstractmethod
    def solve_instance(self, x: Instance) -> int:
        pass


class ZeroShotGPT(Solver):
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        super().__init__()
        self.model_name = model_name
        from openai import OpenAI

        self.client_cls = OpenAI
        self.messages = [
            {
                "role": "system",
                "content": "Solve the brain teaser. Return only the number assigned to the correct answer. "
                           "Don't provide the answer content"
            }
        ]

    @staticmethod
    def _format_question(instance: Instance) -> str:
        return "QUESTION: " + instance.question.strip() + " CHOICES: " + " ".join(
            [f"{i}) {choice.strip()}" for i, choice in enumerate(instance.choice_list)]
        ) + " ANSWER: "

    def solve_instance(self, instance: Instance, retry_counter: int = 3) -> int:
        messages = self.messages + [
            {
                "role": "user",
                "content": ZeroShotGPT._format_question(instance)
            }
        ]

        try:
            response = self.client_cls().chat.completions.create(
                model=self.model_name,
                messages=messages
            )
            full_answer = response.choices[0].message.content
            num_answer = int(re.compile(r"\d+").match(full_answer).group(0))

            return num_answer
        except Exception as e:
            if retry_counter > 1:
                print("An error occurred during generating response. Retrying...")
                return self.solve_instance(instance, retry_counter=retry_counter - 1)
            raise RuntimeError("The number of maximum retries has been reached.") from e


class FineTunedGPT(ZeroShotGPT):
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        super().__init__(model_name=model_name)

    def build_json(self, dataset: DataSet, path: str = "../data/fine-tuning.jsonl") -> None:
        import json
        from pathlib import Path

        lines = []
        for instance in dataset:
            content = ZeroShotGPT._format_question(instance)
            answer = instance.answer_idx
            lines.append(
                json.dumps(
                    {
                        "messages": self.messages +
                                    [{"role": "user", "content": content}] +
                                    [{"role": "assistant", "content": str(answer)}]
                    }
                )
            )
        with Path(path).open("wb") as fp:
            for line in lines:
                line += "\n"
                line = str.encode(line)
                fp.write(line)

    def fit(self, dataset: DataSet, dataset_path: str = "../data/fine-tuning.jsonl") -> None:
        from pathlib import Path

        client = self.client_cls()

        self.build_json(dataset, dataset_path)
        with Path(dataset_path).open("rb") as fp:
            response = client.files.create(
                file=fp,
                purpose="fine-tune"
            )
        training_file = response.id

        client.fine_tuning.jobs.create(
            training_file=training_file,
            model=self.model_name,
            hyperparameters={
                "n_epochs": 1
            }
        )


class ContextAwareZeroShotGPT(ZeroShotGPT):
    class Context(Enum):
        SENTENCE = "sentence"
        WORD = "word"

    def __init__(self, context: Context, model_name: str = "gpt-3.5-turbo"):
        super().__init__(model_name=model_name)
        match context:
            case self.Context.SENTENCE:
                description = "Solve brain teaser where the puzzle defying commonsense is centered on sentence snippets"
            case self.Context.WORD:
                description = ("Solve brain teaser where the answer violates the default meaning of the word and "
                               "focuses on the letter composition of the target question")
            case _:
                raise ValueError("The context type has to be one of ContextAwareZeroshotGPT.Context")

        self.messages = [
            {
                "role": "system",
                "content": f"{description}. Return only the number assigned to the correct answer. "
                           "Don't provide the answer content"
            }
        ]


class InContextGPT(ZeroShotGPT):
    class Context(Enum):
        SENTENCE = "sentence"
        WORD = "word"

    def __init__(self, context: Context, model_name: str = "gpt-3.5-turbo", file_name: str | None = None):
        super().__init__(model_name=model_name)

        match context:
            case self.Context.SENTENCE:
                file_name = file_name or "../data/WP-train.pkl"
            case self.Context.WORD:
                file_name = file_name or "../data/WP-train.pkl"
            case _:
                raise ValueError("The context type has to be one of ContextAwareZeroshotGPT.Context")

        self.dataset = DataSet.from_file(file_name)

    def _find_nn(self, instance: Instance) -> TrainingInstance | None:
        if not hasattr(instance, "embedding"):
            instance.embed()
        nn = (None, np.inf)
        for neighbour in self.dataset:
            distance = cosine(neighbour.embedding, instance.embedding)
            if distance < nn[1]:
                nn = (neighbour, distance)
        return nn[0]

    @staticmethod
    def format_nn_as_example(instance: TrainingInstance) -> str:
        return ZeroShotGPT._format_question(instance) + str(instance.answer_idx)

    def solve_instance(self, instance: Instance, retry_counter: int = 3) -> int:
        example = InContextGPT.format_nn_as_example(self._find_nn(instance))
        messages = [
            {
                "role": "system",
                "content": "Solve the brain teaser. Return only the number assigned to the correct answer. Don't provide the "
                           "answer content. Example: " + example
            },
            {
                "role": "user",
                "content": ZeroShotGPT._format_question(instance)
            }
        ]

        try:
            response = self.client_cls().chat.completions.create(
                model=self.model_name,
                messages=messages
            )
            full_answer = response.choices[0].message.content
            num_answer = int(re.compile(r"\d+").match(full_answer).group(0))

            return num_answer
        except Exception as e:
            if retry_counter > 1:
                print("An error occurred during generating response. Retrying...")
                return self.solve_instance(instance, retry_counter=retry_counter - 1)
            raise RuntimeError("The number of maximum retries has been reached.") from e
