import os
import json
from pathlib import Path
from typing import Iterable, Union, Any


def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except:
                print("Error in loading:", line)
                exit()


def load_data(data_name, data_dir="./data"):
    data_file = f"{data_dir}/{data_name}/test.jsonl"
    assert os.path.exists(data_file)
    examples = list(load_jsonl(data_file))

    # add 'idx' in the first column
    if "idx" not in examples[0]:
        examples = [{"idx": i, **example} for i, example in enumerate(examples)]

    # dedepulicate & sort
    examples = sorted(examples, key=lambda x: x["idx"])
    return examples


def prepare_prompt(question, tokenizer, data_name):
    if data_name in ["gpqa"]:
        prefix = (
            "Answer the following multiple choice question. "
            "The last line of your response should be of the following format: "
            "'ANSWER: $LETTER' (without quotes) where LETTER is one of ABCD. "
            "Think step by step before answering.\n\n"
        )
        message = [
            {"role": "user", "content": prefix + "Question: " + question},
        ]
    else:
        system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Question: " + question},
        ]
    prompt = tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True,
    )
    # Some chat templates append <think> at the end by default, here make sure a unified template 
    if "<think>" not in prompt:
        prompt = prompt + "<think>\n"
    return prompt


def parse_question(example, data_name):
    question = ""
    if data_name in ["gpqa"]:
        options = example["choices"]
        assert len(options) == 4
        for i, (label, option) in enumerate(zip("ABCD", options)):
            options[i] = f"{label}. {str(option).strip()}\n"
        options = " ".join(options).strip()
        question = f"{example['question'].strip()}\n\n {options}"
    else:
        for key in ["question", "problem", "Question", "input"]:
            if key in example:
                question = example[key]
                break
    return question.strip()