# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" ALBERT model configuration """

from __future__ import absolute_import, division, print_function, unicode_literals
from transformers.utils import logging
from transformers import PretrainedConfig

logger = logging.get_logger(__name__)

KOELECTRA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'monologg/koelectra-base-v3-discriminator': "https://huggingface.co/monologg/koelectra-base-v3-discriminator/raw/main/config.json",
    'monologg/koelectra-small-v3-discriminator': "https://huggingface.co/monologg/koelectra-small-v3-discriminator/raw/main/config.json",
}


class KoElectraConfig(PretrainedConfig):

    pretrained_config_archive_map = KOELECTRA_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self,
                 vocab_size=35000,
                 hidden_size=768,  # 256
                 # embedding_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,  # 4
                 intermediate_size=3072,  # 1024
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12,
                 **kwargs):
        super(KoElectraConfig, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        # self.embedding_size = embedding_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
