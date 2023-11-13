import re
from abc import ABC, abstractmethod

from tqdm.auto import tqdm

from structs import DataSet, Instance


class Solver(ABC):
    def solve(self, x: DataSet | Instance, show_progress: bool = True) -> int | list[int]:
        if isinstance(x, Instance):
            return self.solve_instance(x)
        if isinstance(x, DataSet):
            if show_progress:
                return [self.solve_instance(i) for i in tqdm(x)]
            return [self.solve_instance(i) for i in x]
        raise TypeError("An input has to be Instance of DataSet")

    @abstractmethod
    def solve_instance(self, x: Instance) -> int:
        pass


class ZeroShotGPT(Solver):
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
        from openai import OpenAI

        self.client_cls = OpenAI
        self.messages = [
            {
                "role": "system",
                "content": "Solve the brain teaser. Return only the number assigned to the correct answer."
            }
        ]

    def solve_instance(self, instance: Instance) -> int:
        content = instance.question + "\n" + "\n".join(
            [f"{i}) {choice}" for i, choice in enumerate(instance.choice_list)]
        )
        messages = self.messages + [
            {
                "role": "user",
                "content": content
            }
        ]

        response = self.client_cls().chat.completions.create(
            model=self.model_name,
            messages=messages
        )

        full_answer = response.choices[0].message.content
        num_answer = int(re.compile(r"\d+").match(full_answer).group(0))

        return num_answer
