import pandas as pd
import json
from tqdm import tqdm

from typing import Union, Any, Dict
from datasets.arrow_dataset import Batch
from datasets import Dataset

import os
import argparse
import datasets
from transformers.utils import logging, check_min_version
from transformers.utils.versions import require_version

from retro_reader import RetroReader
from retro_reader.constants import EXAMPLE_FEATURES


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.13.0.dev0")

require_version("datasets>=1.8.0")

logger = logging.get_logger(__name__)
logger.setLevel(logging.INFO)

def schema_integrate(example: Batch) -> Union[Dict, Any]:
    title = example["title"]
    question = example["question"]
    context = example["context"]
    guid = example["id"]
    classtype = [""] * len(title)
    dataset_name = source = ["govquad_v2"] * len(title)
    answers, is_impossible = [], []
    for answer_examples in example["answers"]:
        answer_examples = eval(answer_examples)
        if answer_examples["text"]:
            answers.append(answer_examples)
            is_impossible.append(False)
        else:
            answers.append({"text": [""], "answer_start": []})
            is_impossible.append(True)
    # The feature names must be sorted.
    return {
        "guid": guid,
        "question": question,
        "context": context,
        "answers": answers,
        "title": title,
        "classtype": classtype,
        "source": source,
        "is_impossible": is_impossible,
        "dataset": dataset_name,
    }


def load_and_convert(filepath):
    squad_dataset = pd.DataFrame(columns=["id", "title", "context", "question", "answers"])
    with open(filepath) as f:
        squad = json.load(f)
        for article in tqdm(squad["data"]): # {title, paragraphs}
            title = article.get("title", "").strip()
            for paragraph in article["paragraphs"]: # paragraph : {qas, context}
                context = paragraph["context"].strip()

                for qa in paragraph["qas"]: 
                # ans_qa : {question, id, answers=[{text, answer_start}], is_impossible}
                # unans_qa : {plausible_answer=[{text, answer_start}], question, id, answers=[], is_impossible}
                    question = qa["question"].strip()
                    id_ = qa["id"]
                    if qa["answers"]:
                        answer_starts = [answer["answer_start"] for answer in qa["answers"]]
                        answers = [answer["text"].strip() for answer in qa["answers"]]
                    else:
                        answer_starts = []
                        answers = []
                    squad_dataset.loc[len(squad_dataset)] = [id_, title, context, question, {"text": answers, "answer_start": answer_starts}]
    return squad_dataset


def main(args):
    # # ==== modified ====
    # # Load test dataset
    file_test = 'f_validation_8m'

    # # 전처리 부분 추가 필요
    # test_df = load_and_convert('data/' + file_test + '.json')

    # test_df.to_csv('data/df_' + file_test + '.csv', mode='w')
    test_df = pd.read_csv('data/' + file_test + '.csv')

    test_dataset = Dataset.from_dict(test_df)


    korquad = datasets.DatasetDict({
        "validation": test_dataset  # num_rows: 5800 -> 6424   (2002)
    })

    korquad = korquad.map(
        schema_integrate, 
        batched=True,
        remove_columns=korquad.column_names["validation"],
        features=EXAMPLE_FEATURES,
    )

    num_unanswerable_valid = sum(korquad["validation"]["is_impossible"])
    logger.warning(f"Number of unanswerable sample for SQuAD v2.0 validation dataset: {num_unanswerable_valid}")
    # num_rows in train: 29800 -> 49645, num_unanswerable in train:  16353
    # num_rows in valid:  5800 -> 6424, num_unanswerable in valid:  2002

    # Load Retro Reader
    # features: parse arguments
    #           make train/eval dataset from examples
    #           load model from 🤗 hub
    #           set sketch/intensive reader and rear verifier
    
    # configs : inference_ko_electra_small.yaml file
    retro_reader = RetroReader.load(config_file=args.configs)

    logger.info("Start inference")
    outputs = retro_reader.inference(korquad['validation'], mode='test')
    
    logger.info("Saving final_results")
    with open('outputs/rear_verification/final_result.json', "w") as writer:
        writer.write(json.dumps(outputs, indent=4, ensure_ascii=False) + "\n")

    logger.warning("Inference retrospective reader Done.")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", "-c", type=str, default="configs/inference_ko_electra_small.yaml", help="config file path")
    
    # add for checkpoint resume
    parser.add_argument("--resume", "-r", type=str, default="", help="resuming from checkpoint path")
    
    args = parser.parse_args()
    main(args)
