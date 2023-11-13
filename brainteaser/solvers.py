from abc import ABC, abstractmethod

from structs import DataSet, Instance


class Solver(ABC):
    def solve(self, x: DataSet | Instance) -> int | list[int]:
        if isinstance(x, Instance):
            return self.solve_instance(x)
        if isinstance(x, DataSet):
            return [self.solve_instance(i) for i in x]
        raise TypeError("An input has to be Instance of DataSet")

    @abstractmethod
    def solve_instance(self, x: Instance) -> int:
        pass


class ZeroShotGPT(Solver):
    def __init__(self):
        from openai import OpenAI

        self.client = OpenAI()
        self.messages = [
            {
                "role": "system",
                "content": "Solve the brain teaser. Return only the number corresponding to the correct answer."
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

        print(instance.__dir__())
        answer = 0
        print(instance.is_answer_correct(answer))
        return 1
