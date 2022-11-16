import pandas as pd
import json
from tqdm import tqdm

from typing import Union, Any, Dict
from datasets.arrow_dataset import Batch
from datasets import Dataset

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


# GovQuAD에 answer 2개씩 되어있는 거 없음.
# # data augmentation for multiple answers
# def data_aug_for_multiple_answers(example: Batch) -> Union[Dict, Any]:
#     result = {key: [] for key in examples.keys()}
    
#     def update(i, answers=None):
#         for key in result.keys():
#             if key == "answers" and answers is not None:
#                 result[key].append(answers)
#             else:
#                 result[key].append(examples[key][i])
                
#     for i, (answers, unanswerable) in enumerate(
#         zip(examples["answers"], examples["is_impossible"])
#     ):
#         answerable = not unanswerable
#         assert (
#             len(answers["text"]) == len(answers["answer_start"]) or
#             answers["answer_start"][0] == -1
#         )
#         if answerable and len(answers["text"]) > 1:
#             for n_ans in range(len(answers["text"])):
#                 ans = {
#                     "text": [answers["text"][n_ans]],
#                     "answer_start": [answers["answer_start"][n_ans]],
#                 }
#                 update(i, ans)
#         elif not answerable:
#             update(i, {"text": [], "answer_start": []})
#         else:
#             update(i)
            
#     return result


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
    # # Load GovQuAD V2.0 dataset
    file_train = 'train_10_prep2'
    file_valid = 'dev_10_prep2'

    # train_df = load_and_convert('data/' + file_train + '.json')
    # valid_df = load_and_convert('data/' + file_valid + '.json')

    # train_df.to_csv('data/df_' + file_train + '.csv', mode='w')
    # valid_df.to_csv('data/df_' + file_valid + '.csv', mode='w')

    train_df = pd.read_csv('data/df_' + file_train + '.csv')
    valid_df = pd.read_csv('data/df_' + file_valid + '.csv')

    train_dataset = Dataset.from_dict(train_df)
    valid_dataset = Dataset.from_dict(valid_df)

    korquad = datasets.DatasetDict({
        "train": train_dataset,      # num_rows: 29800 -> 49645  (16353)
        "validation": valid_dataset  # num_rows: 5800 -> 6424   (2002)
    })

    korquad = korquad.map(
        schema_integrate, 
        batched=True,
        remove_columns=korquad.column_names["train"],
        features=EXAMPLE_FEATURES,
    )
    # num_rows in train: 29800 -> 49645, num_unanswerable in train:  16353
    # num_rows in valid:  5800 -> 6424, num_unanswerable in valid:  2002
    num_unanswerable_train = sum(korquad["train"]["is_impossible"])
    num_unanswerable_valid = sum(korquad["validation"]["is_impossible"])
    logger.warning(f"Number of unanswerable sample for SQuAD v2.0 train dataset: {num_unanswerable_train}")
    logger.warning(f"Number of unanswerable sample for SQuAD v2.0 validation dataset: {num_unanswerable_valid}")


    # # Train data augmentation for multiple answers
    # # no answer {"text": [], "answer_start": [-1]} -> {"text": [], "answer_start": []}
    # korquad_train = korquad["train"].map(
    #     data_aug_for_multiple_answers,
    #     batched=True,
    #     batch_size=1000,
    #     num_proc=5,
    # )

    # korquad = datasets.DatasetDict({
    #     "train": korquad_train,              # num_rows: 130,319
    #     "validation": korquad["validation"]  # num_rows:  11,873
    # })



    # Load Retro Reader
    # features: parse arguments
    #           make train/eval dataset from examples
    #           load model from 🤗 hub
    #           set sketch/intensive reader and rear verifier
    retro_reader = RetroReader.load(
        train_examples=korquad["train"],
        eval_examples=korquad["validation"],
        config_file=args.configs
    )

    print(retro_reader.sketch_reader.args.num_train_epochs)



    # Train
    # modified!!!!!!!! training only intensive module
    retro_reader.train('intensive')
    logger.warning("Train retrospective reader Done.")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", "-c", type=str, default="configs/train_en_electra_large.yaml", help="config file path")
    
    # add for checkpoint resume
    parser.add_argument("--resume", "-r", type=str, default="", help="resuming from checkpoint path")
    
    args = parser.parse_args()
    print(args) # for checking
    main(args)
