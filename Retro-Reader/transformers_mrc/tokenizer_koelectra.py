# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes."""

from __future__ import absolute_import, division, print_function, unicode_literals
from transformers import BertTokenizer
from transformers.utils import logging

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {'vocab_file': 'vocab.txt'}

PRETRAINED_VOCAB_FILES_MAP = {
    'vocab_file': {
        'monologg/koelectra-base-v3-discriminator': "https://huggingface.co/monologg/koelectra-base-v3-discriminator/raw/main/vocab.txt",
        'monologg/koelectra-small-v3-discriminator': "https://huggingface.co/monologg/koelectra-small-v3-discriminator/raw/main/vocab.txt"
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    'monologg/koelectra-base-v3-discriminator': 512,
    'monologg/koelectra-small-v3-discriminator': 512
}

PRETRAINED_INIT_CONFIGURATION = {
    'monologg/koelectra-base-v3-discriminator': {'do_lower_case': False},
    'monologg/koelectra-small-v3-discriminator': {'do_lower_case': False}
}


class KoElectraTokenizer(BertTokenizer):
    r"""
    Constructs an Electra tokenizer.
    :class:`~transformers.ElectraTokenizer` is identical to :class:`~transformers.BertTokenizer` and runs end-to-end
    tokenization: punctuation splitting + wordpiece.

    Refer to superclass :class:`~transformers.BertTokenizer` for usage examples and documentation concerning
    parameters.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
