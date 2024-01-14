import re
from enum import Enum
import torch
import numpy as np
from transformers import (
    pipeline,
    AutoTokenizer,
    T5ForConditionalGeneration,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from scipy.spatial.distance import cosine
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from solvers import Solver
from structs import Instance, DataSet, TrainingInstance
import wandb

class Zephyr7BetaSolver(Solver):
    def __init__(
            self,
            model_name: str = "HuggingFaceH4/zephyr-7b-beta",
    ):
        super().__init__()
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.model_name = model_name
        self.pipeline = pipeline(
            "conversational", model=model_name, tokenizer=model_name, device=device
        )
        self.messages = [
            {
                "role": "system",
                "content": "Solve the brain teaser. Return only the number assigned to the correct answer. "
                           "Don't provide the answer content",
            }
        ]

    def solve_instance(self, instance: Instance, retry_counter: int = 3) -> int:
        content = (
                "QUESTION: "
                + instance.question.strip()
                + " CHOICES: "
                + " ".join(
            [
                f"{i}) {choice.strip()}"
                for i, choice in enumerate(instance.choice_list)
            ]
        )
                + " ANSWER: "
        )
        messages = self.messages + [{"role": "user", "content": content}]

        try:
            response = self.pipeline(messages)
            full_answer = str(response.generated_responses[0])
            # print(full_answer)
            num_answer = re.search(r"\d+", full_answer)
            # print(num_answer)
            num_answer = num_answer.group(0)
            # print(num_answer)

            num_answer = int(num_answer)

            return num_answer
        except Exception as e:
            if retry_counter > 1:
                print("An error occurred during generating response. Retrying...")
                return self.solve_instance(instance, retry_counter=retry_counter - 1)
            raise RuntimeError("The number of maximum retries has been reached.") from e


class InContextZephyr(Zephyr7BetaSolver):
    """
    YYY
    """
    class Context(Enum):
        SENTENCE = "sentence"
        WORD = "word"
    
    def __init__(
            self,
            context: Context,
            model_name: str = "HuggingFaceH4/zephyr-7b-beta",
    ):
        super().__init__(model_name=model_name)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            padding_side='left',
            add_eos_token=True)
        print("Tokenizer not set",self.tokenizer.pad_token)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print("Tokenizer set",self.tokenizer.pad_token)
        match context:
            case self.Context.SENTENCE:
                file_name = "WP-train.pkl"
            case self.Context.WORD:
                file_name = "WP-train.pkl"
            case _:
                raise ValueError("The context type has to be one of ...")
        
        self.dataset = DataSet.from_file(f"/home/ubuntu/repositories/brainteaser/data/{file_name}")


    def _find_nn(self, instance: Instance) -> TrainingInstance | None:
        if not hasattr(instance, "embedding"):
            instance.embed()
        nn = (None, np.inf)
        for neighbour in self.dataset:
            distance = cosine(neighbour.embedding, instance.embedding)
            if distance < nn[1]:
                nn = (neighbour, distance)
        return nn[0]

    def solve_instance(self, instance: Instance, retry_counter: int = 3) -> int:
        example = self._find_nn(instance)
        example_content = (
                "QUESTION: "
                + example.question.strip()
                + " CHOICES: "
                + " ".join(
            [
                f"{i}) {choice.strip()}"
                for i, choice in enumerate(example.choice_list)
            ]
        )
                + " ANSWER: " + str(example.answer_idx)
        )
        messages = [
            {
                "role": "system",
                "content": "Solve the brain teaser. Return only the number assigned to the correct answer. "
                           "Don't provide the answer content. Example: " + example_content,
            }
        ]
        # print(messages)
        content = (
                "QUESTION: "
                + instance.question.strip()
                + " CHOICES: "
                + " ".join(
            [
                f"{i}) {choice.strip()}"
                for i, choice in enumerate(instance.choice_list)
            ]
        )
                + " ANSWER: "
        )
        messages = messages + [{"role": "user", "content": content}]

        try:
            response = self.pipeline(messages)
            full_answer = str(response.generated_responses[0])
            # print(full_answer)
            num_answer = re.search(r"\d+", full_answer)
            # print(num_answer)
            num_answer = num_answer.group(0)
            # print(num_answer)

            num_answer = int(num_answer)

            return num_answer
        except:
            if retry_counter > 1:
                print("An error occurred during generating response. Retrying...")
                return self.solve_instance(instance, retry_counter=retry_counter - 1)
            # raise RuntimeError("The number of maximum retries has been reached.") from e
            print("Returning -1")
            return -1

    def load_fine_tuned(self, path):
        """
        load fine-tuned model from path
        """
        self.model = AutoModelForCausalLM.from_pretrained(path)        
        self.pipeline = pipeline(
            "conversational", model=self.model, tokenizer=self.tokenizer, device=self.device
        )

class FineTunedZephyr7BetaSolver(Solver):
    """
    Fine-tuned Zephyr-7B model.
    """

    def __init__(
            self,
            model_name: str = "HuggingFaceH4/zephyr-7b-beta",
    ):
        super().__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            padding_side='left',
            add_eos_token=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = None
        self.pipeline = None

        self.messages = {
            "role": "system",
            "content": "Solve the brain teaser. Return only the number assigned to the correct answer. "
                       "Don't provide the answer content",
        }

    def process_instance(self, instance: Instance):
        """
        Process an instance into a chat template.
        """
        content = "QUESTION: " + instance.question.strip() + " CHOICES: " + " ".join(
            [f"{i}) {choice.strip()}" for i, choice in enumerate(instance.choice_list)]
        ) + " ANSWER: "
        answer = instance.answer_idx

        chat = [
            self.messages,
            {"role": "user", "content": content},
            {"role": "assistant", "content": str(answer)}

        ]
        parsed = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
        return parsed

    def get_dataset(self, dataset: DataSet):
        """
        Process a dataset into a list of chat templates.
        """
        data = []
        for instance in dataset:
            data.append(self.process_instance(instance))
        tokenized_data = list(map(self.tokenizer, data))
        return tokenized_data

    def fine_tune(self,
                  train_dataset: DataSet,
                  eval_dataset: DataSet,
                  epochs: int = 1,
                  use_cpu: bool = True,
                  batch_size: int = 8
                ):
        """
        Fine-tune the model on a dataset.
        """
        wandb.init(
            project="brainteasers",
            name=f"{epochs}Shot-{self.model_name}-fine-tuning",
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
        )

        config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM")
        
        # Stabilize output layer and layernorms & prepare for 8bit training.
        self.model = prepare_model_for_kbit_training(self.model, 16)

        # Set PEFT adapter on model.
        self.model = get_peft_model(self.model, config)

        train_data = self.get_dataset(train_dataset)
        val_data = self.get_dataset(eval_dataset)

        training_config = TrainingArguments(
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            warmup_steps=10,
            optim="adamw_torch",
            learning_rate=2e-4,
            logging_steps=1,
            output_dir=".",
            overwrite_output_dir=True,
            report_to='wandb',
            load_best_model_at_end=True,
            evaluation_strategy='epoch',
            metric_for_best_model='eval_loss',
            greater_is_better=False,
            save_strategy="epoch",
            use_cpu=use_cpu,
        )
        
        # version with using steps instead of epochs

        # training_config = TrainingArguments(
        #     per_device_train_batch_size=batch_size,
        #     per_device_eval_batch_size=batch_size,
        #     max_steps=steps,
        #     warmup_steps=100,
        #     optim="adamw_torch",
        #     learning_rate=2e-4,
        #     logging_steps=100,
        #     output_dir=".",
        #     overwrite_output_dir=True,
        #     report_to='wandb',
        #     logging_strategy="steps",
        #     load_best_model_at_end=True,
        #     evaluation_strategy='steps',
        #     metric_for_best_model='eval_loss',
        #     greater_is_better=False,
        #     save_strategy="steps",
        #     save_steps=200,
        #     save_total_limit=5,
        #     use_cpu=use_cpu,
        # )

        # Setup collator.
        data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)

        # Setup trainer.
        trainer = Trainer(
            model=self.model,
            train_dataset=train_data,
            eval_dataset=val_data,
            data_collator=data_collator,
            args=training_config,
        )

        trainer.train()
        wandb.finish()

    def load_fine_tuned(self, path):
        """
        load fine-tuned model from path
        """
        self.model = AutoModelForCausalLM.from_pretrained(path)

    def solve_instance(self, instance: Instance, retry_counter: int = 3) -> int:

        self.pipeline = pipeline(
            "conversational", model=self.model, tokenizer=self.tokenizer, device=self.device
        )
        content = (
                "QUESTION: "
                + instance.question.strip()
                + " CHOICES: "
                + " ".join(
            [
                f"{i}) {choice.strip()}"
                for i, choice in enumerate(instance.choice_list)
            ]
        )
                + " ANSWER: "
        )
        messages = [self.messages] + [{"role": "user", "content": content}]

        try:
            # print("Messages:", messages)
            # print("---------------------------------------")
            response = self.pipeline(messages)
            full_answer = str(response.generated_responses[0])
            # print(f"Full answer:{full_answer}")
            # print("---------------------------------------")
            num_answer = re.search(r"\d+", full_answer)
            if num_answer is None:
                print(f"Returning -1 with answer {full_answer}")
                return -1
            num_answer = num_answer.group(0)
            num_answer = int(num_answer)
            # print("Num answer:", num_answer)
            # print("=======================================")
            return num_answer
        except Exception as e:
            if retry_counter > 1:
                print("An error occurred during generating response. Retrying...")
                return self.solve_instance(instance, retry_counter=retry_counter - 1)
            raise RuntimeError("The number of maximum retries has been reached.") from e
        
class Orca2Solver(Solver):
    """
    Orca-2 model.
    """
    def __init__(
            self,
            model_name: str = "microsoft/Orca-2-13b",
    ):
        super().__init__()

        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map='auto')

        # https://github.com/huggingface/transformers/issues/27132
        # please use the slow tokenizer since fast and slow tokenizer produces different tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=False,
            )

        self.system_message = "Solve the brain teaser. Return only the number assigned to the correct answer. Don't provide the answer content" 

    def solve_instance(self, instance: Instance, retry_counter: int = 3) -> int:
        """
        Solve an instance using the Orca-2 model.
        """
        content = "QUESTION: " + instance.question.strip() + " CHOICES: " + " ".join(
            [
                f"{i}) {choice.strip()}"
                for i, choice in enumerate(instance.choice_list)
            ]
        ) + " ANSWER: "
        
        prompt = f"<|im_start|>system\n{self.system_message}<|im_end|>\n<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant"

        inputs = self.tokenizer(prompt, return_tensors='pt')

        try:
            output_ids = self.model.generate(inputs["input_ids"].to(self.model.device))
            response = self.tokenizer.batch_decode(output_ids)[0]
            # print(response)
            full_answer = str(response)
            num_answer = re.search(r"\d+", full_answer)
            num_answer = num_answer.group(0)

            num_answer = int(num_answer)

            return num_answer
        except Exception as e:
            if retry_counter > 1:
                print("An error occurred during generating response. Retrying...")
                return self.solve_instance(instance, retry_counter=retry_counter - 1)
            raise RuntimeError("The number of maximum retries has been reached.") from e


class Notus7BSolver(Solver):
    def __init__(
            self,
            model_name: str = "argilla/notus-7b-v1",
    ):
        super().__init__()
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.model_name = model_name
        self.pipeline = pipeline(
            "text-generation", model=model_name, torch_dtype=torch.bfloat16, device=device
        )
        self.messages = [
            {
                "role": "system",
                "content": "Solve the brain teaser. Return only the number assigned to the correct answer. "
                           "Don't provide the answer content",
            }
        ]

    def solve_instance(self, instance: Instance, retry_counter: int = 3) -> int:
        content = (
                "QUESTION: "
                + instance.question.strip()
                + " CHOICES: "
                + " ".join(
            [
                f"{i}) {choice.strip()}"
                for i, choice in enumerate(instance.choice_list)
            ]
        )
                + " ANSWER: "
        )
        messages = self.messages + [{"role": "user", "content": content}]

        try:
            prompt = self.pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            outputs = self.pipeline(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
            full_answer = outputs[0]["generated_text"]
            # print(full_answer)
            num_answer = re.search(r"\d+", full_answer)
            # print(num_answer)
            num_answer = num_answer.group(0)
            # print(num_answer)

            num_answer = int(num_answer)

            return num_answer
        except Exception as e:
            if retry_counter > 1:
                print("An error occurred during generating response. Retrying...")
                return self.solve_instance(instance, retry_counter=retry_counter - 1)
            raise RuntimeError("The number of maximum retries has been reached.") from e


class ByT5Solver(Solver):
    def __init__(self, model_name: str = "google/byt5-xl"):
        super().__init__()
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to("cuda:0" if torch.cuda.is_available() else "cpu")

    def solve_instance(self, instance: Instance) -> int:
        input = (
                "Solve the brain teaser."
                +
                # " Return only the number assigned to the correct answer. " +
                "Don't provide the answer content. "
                + "QUESTION: "
                + instance.question.strip()
        )
        #          + " CHOICES: " + " ".join(
        #     [f"{i}) {choice.strip()}" for i, choice in enumerate(instance.choice_list)]
        # ) + " ANSWER: "

        model_inputs = self.tokenizer(input, padding="longest", return_tensors="pt").to(
            self.model.device
        )

        model_outputs = (
            self.model.generate(**model_inputs, max_new_tokens=200)
            .cpu()
            .flatten()
            .tolist()
        )
        print(input)
        print("--------------")
        decoded = self.tokenizer.decode(
            model_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        print(decoded)
        print("==============")
        return 0


