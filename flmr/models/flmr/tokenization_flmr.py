# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team, The Hugging Face Team.
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
"""Tokenization classes for FLMR."""


from typing import List, Optional, Union

from transformers.utils import TensorType, logging
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers import AutoTokenizer
from .configuration_flmr import FLMRTextConfig

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer_config.json"}

CONTEXT_ENCODER_PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "LinWeizheDragon/PreFLMR_ViT-L": (
            "https://huggingface.co/LinWeizheDragon/PreFLMR_ViT-L/resolve/main/context_tokenizer/vocab.txt"
        ),
        "LinWeizheDragon/FLMR": (
            "https://huggingface.co/LinWeizheDragon/FLMR/resolve/main/context_tokenizer/vocab.txt"
        ),
    },
    "tokenizer_file": {
        "LinWeizheDragon/PreFLMR_ViT-L": (
            "https://huggingface.co/LinWeizheDragon/PreFLMR_ViT-L/resolve/main/context_tokenizer/tokenizer_config.json"
        ),
        "LinWeizheDragon/FLMR": (
            "https://huggingface.co/LinWeizheDragon/FLMR/resolve/main/context_tokenizer/tokenizer_config.json"
        ),
    },
}
QUESTION_ENCODER_PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "LinWeizheDragon/PreFLMR_ViT-L": (
            "https://huggingface.co/LinWeizheDragon/PreFLMR_ViT-L/resolve/main/query_tokenizer/vocab.txt"
        ),
        "LinWeizheDragon/FLMR": ("https://huggingface.co/LinWeizheDragon/FLMR/resolve/main/query_tokenizer/vocab.txt"),
    },
    "tokenizer_file": {
        "LinWeizheDragon/PreFLMR_ViT-L": (
            "https://huggingface.co/LinWeizheDragon/PreFLMR_ViT-L/resolve/main/query_tokenizer/tokenizer_config.json"
        ),
        "LinWeizheDragon/FLMR": (
            "https://huggingface.co/LinWeizheDragon/FLMR/resolve/main/query_tokenizer/tokenizer_config.json"
        ),
    },
}


CONTEXT_ENCODER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "LinWeizheDragon/PreFLMR_ViT-L": 512,
    "LinWeizheDragon/FLMR": 512,
}
QUESTION_ENCODER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "LinWeizheDragon/PreFLMR_ViT-L": 512,
    "LinWeizheDragon/FLMR": 512,
}


CONTEXT_ENCODER_PRETRAINED_INIT_CONFIGURATION = {
    "LinWeizheDragon/PreFLMR_ViT-L": {"do_lower_case": True},
    "LinWeizheDragon/FLMR": {"do_lower_case": True},
}
QUESTION_ENCODER_PRETRAINED_INIT_CONFIGURATION = {
    "LinWeizheDragon/PreFLMR_ViT-L": {"do_lower_case": True},
    "LinWeizheDragon/FLMR": {"do_lower_case": True},
}


# Modified from colbert.modeling.tokenization
class FLMRBertContextEncoderTokenizer(BertTokenizer):
    r"""
    Construct a FLMRContextEncoder tokenizer.

    [`FLMRContextEncoderTokenizer`] is identical to [`BertTokenizer`] and runs end-to-end tokenization: punctuation
    splitting and wordpiece.

    Refer to superclass [`BertTokenizer`] for usage examples and documentation concerning parameters.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = CONTEXT_ENCODER_PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = CONTEXT_ENCODER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = CONTEXT_ENCODER_PRETRAINED_INIT_CONFIGURATION

    def __init__(
        self,
        doc_maxlen: Optional[int] = 512,
        **kwargs,
    ):
        super().__init__(
            doc_maxlen=doc_maxlen,
            **kwargs,
        )

        self.doc_maxlen = doc_maxlen
        self.D_marker_token, self.D_marker_token_id = "[D]", self.convert_tokens_to_ids("[unused1]")

    def __call__(
        self,
        text: List[str],
        padding: Optional[Union[str, bool]] = "max_length",
        truncation: Optional[Union[bool, str]] = "longest_first",
        max_length: Optional[int] = 512,
        return_tensors: Optional[Union[str, TensorType]] = "pt",
        **kwargs,
    ):
        # add placehold for the [D] marker
        text = [". " + x for x in text]

        if max_length > self.doc_maxlen:
            # can not exceed the pre-set length
            max_length = self.doc_maxlen

        encoding = super().__call__(
            text,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
            max_length=max_length,
            **kwargs,
        )

        ids, mask = encoding["input_ids"], encoding["attention_mask"]

        # postprocess for the [D] marker
        ids[:, 1] = self.D_marker_token_id

        # if bsize:
        #     # This bsize function is used in the original ColBERT codebase to split inputs into multiple batches
        #     if image_features is not None:
        #         ids, mask, image_features, reverse_indices = _sort_by_length(ids, mask, bsize, image_features=image_features)
        #         batches = _split_into_batches(ids, mask, bsize, image_features=image_features)
        #     else:
        #         ids, mask, reverse_indices = _sort_by_length(ids, mask, bsize)
        #         batches = _split_into_batches(ids, mask, bsize)

        #     return batches, reverse_indices

        encoding["input_ids"] = ids
        encoding["attention_mask"] = mask

        return encoding


# Modified from colbert.modeling.tokenization
class FLMRBertQueryEncoderTokenizer(BertTokenizer):
    r"""
    Constructs a FLMRQueryEncoder tokenizer.

    [`FLMRQueryEncoder`] is identical to [`BertTokenizer`] and runs end-to-end tokenization: punctuation
    splitting and wordpiece.

    Refer to superclass [`BertTokenizer`] for usage examples and documentation concerning parameters.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = QUESTION_ENCODER_PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = QUESTION_ENCODER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = QUESTION_ENCODER_PRETRAINED_INIT_CONFIGURATION

    def __init__(
        self,
        *args,
        query_maxlen: Optional[int] = 32,
        attend_to_mask_tokens: Optional[bool] = False,
        **kwargs,
    ):
        super().__init__(
            *args,
            query_maxlen=query_maxlen,
            attend_to_mask_tokens=attend_to_mask_tokens,
            **kwargs,
        )

        self.query_maxlen = query_maxlen
        self.background_maxlen = 512 - self.query_maxlen + 1  # FIXME: Make this configurable
        self.attend_to_mask_tokens = attend_to_mask_tokens

        self.Q_marker_token, self.Q_marker_token_id = "[Q]", self.convert_tokens_to_ids("[unused0]")

    def __call__(
        self,
        text: Union[str, List[str]],
        padding: Optional[Union[str, bool]] = "max_length",
        truncation: Optional[Union[bool, str]] = True,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = "pt",
        **kwargs,
    ):
        if isinstance(text, str):
            # convert to list if input is a single string
            text = [text]

        # add placehold for the [Q] marker
        text = [". " + x for x in text]

        if max_length is not None:
            # use user specified max_length
            pass
        else:
            # use default max length
            max_length = self.query_maxlen

        encoding = super().__call__(
            text,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
            max_length=max_length,
            **kwargs,
        )

        ids, mask = encoding["input_ids"], encoding["attention_mask"]

        # postprocess for the [Q] marker and the [MASK] augmentation
        ids[:, 1] = self.Q_marker_token_id
        ids[ids == self.pad_token_id] = self.mask_token_id

        if self.attend_to_mask_tokens:
            # When attend_to_mask_tokens is True, we want to attend to the [MASK] tokens
            mask[ids == self.mask_token_id] = 1
            assert mask.sum().item() == mask.size(0) * mask.size(1), mask

        return {"input_ids": ids, "attention_mask": mask}

class FLMRAutoContextEncoderTokenizer:
    r"""
    Construct a ContextEncoderTokenizer tokenizer with AutoTokenizer.

    [`FLMRAutoContextEncoderTokenizer`] is identical to [`AutoTokenizer`] and runs end-to-end tokenization: punctuation
    splitting and wordpiece.

    Refer to superclass [`AutoTokenizer`] for usage examples and documentation concerning parameters.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = CONTEXT_ENCODER_PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = CONTEXT_ENCODER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = CONTEXT_ENCODER_PRETRAINED_INIT_CONFIGURATION

    def __init__(
        self,
        *args,
        doc_maxlen: Optional[int] = 512,
        **kwargs,
    ):
        self.doc_maxlen = doc_maxlen
        self.tokenizer = AutoTokenizer.from_pretrained(*args, **kwargs)
        self.additional_special_tokens = self.tokenizer.additional_special_tokens


    def __call__(
        self,
        text: List[str],
        padding: Optional[Union[str, bool]] = "max_length",
        truncation: Optional[Union[bool, str]] = "longest_first",
        max_length: Optional[int] = 512,
        return_tensors: Optional[Union[str, TensorType]] = "pt",
        **kwargs,
    ):
        # add placehold for the [D] marker
        text = [". " + x for x in text]

        if max_length > self.doc_maxlen:
            # can not exceed the pre-set length
            max_length = self.doc_maxlen

        encoding = self.tokenizer(
            text,
            padding=padding,
            truncation=True,
            return_tensors=return_tensors,
            max_length=max_length,
            **kwargs,
        )

        ids, mask = encoding["input_ids"], encoding["attention_mask"]

        encoding["input_ids"] = ids
        encoding["attention_mask"] = mask

        return encoding

    def encode(self, text, text_pair=None, add_special_tokens=True, **kwargs):
        return self.tokenizer.encode(text, text_pair, add_special_tokens, **kwargs)

    def add_special_tokens(self, token, **kwargs):
        return self.tokenizer.add_special_tokens(token, **kwargs)

    def save_pretrained(self, path):
        self.tokenizer.save_pretrained(path)


# Modified from colbert.modeling.tokenization
class FLMRAutoQueryEncoderTokenizer:
    r"""
    Constructs a QueryEncoderTokenizer tokenizer with AutoTokenizer.

    [`FLMRAutoQueryEncoderTokenizer`] is identical to [`AutoTokenizer`] and runs end-to-end tokenization: punctuation
    splitting and wordpiece.

    Refer to superclass [`AutoTokenizer`] for usage examples and documentation concerning parameters.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = QUESTION_ENCODER_PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = QUESTION_ENCODER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = QUESTION_ENCODER_PRETRAINED_INIT_CONFIGURATION

    def __init__(
        self,
        *args,
        query_maxlen: Optional[int] = 32,
        attend_to_mask_tokens: Optional[bool] = False,
        **kwargs,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(*args, **kwargs)
        self.additional_special_tokens = self.tokenizer.additional_special_tokens
        self.query_maxlen = query_maxlen
        self.background_maxlen = 512 - self.query_maxlen + 1  # FIXME: Make this configurable
        self.attend_to_mask_tokens = attend_to_mask_tokens

    def __call__(
        self,
        text: Union[str, List[str]],
        padding: Optional[Union[str, bool]] = "max_length",
        truncation: Optional[Union[bool, str]] = True,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = "pt",
        **kwargs,
    ):
        if isinstance(text, str):
            # convert to list if input is a single string
            text = [text]

        # add placehold for the [Q] marker
        text = [". " + x for x in text]

        if max_length is not None:
            # use user specified max_length
            pass
        else:
            # use default max length
            max_length = self.query_maxlen

        encoding = self.tokenizer(
            text,
            padding=padding,
            truncation=True,
            return_tensors=return_tensors,
            max_length=max_length,
            **kwargs,
        )

        ids, mask = encoding["input_ids"], encoding["attention_mask"]

        if self.attend_to_mask_tokens:
            # When attend_to_mask_tokens is True, we want to attend to the [MASK] tokens
            mask[ids == self.mask_token_id] = 1
            assert mask.sum().item() == mask.size(0) * mask.size(1), mask

        return {"input_ids": ids, "attention_mask": mask}

    def encode(self, text, text_pair=None, add_special_tokens=True, **kwargs):
        return self.tokenizer.encode(text, text_pair, add_special_tokens, **kwargs)

    def add_special_tokens(self, token, **kwargs):
        return self.tokenizer.add_special_tokens(token, **kwargs)

    def save_pretrained(self, path):
        self.tokenizer.save_pretrained(path)


class FLMRContextEncoderTokenizer:
    r"""
    Constructs a FLMRContextEncoderTokenizer tokenizer.

    [`FLMRContextEncoderTokenizer`] is identical to [`BertTokenizer`] or [`AutoTokenizer`], depends on whether
    the tokenizer is initialized from bert.
    """
    def __init__(self) -> None:
        pass

    @classmethod
    def from_pretrained(
        cls,
        *args,
        text_config: Optional[FLMRTextConfig] = None,
        **kwargs,
    ):
        if text_config.text_encoder_base_model == "bert-base-uncased":
            return FLMRBertContextEncoderTokenizer.from_pretrained(*args, **kwargs)
        else:
            return FLMRAutoContextEncoderTokenizer(*args, **kwargs)

class FLMRQueryEncoderTokenizer:
    r"""
    Constructs a FLMRContextEncoderTokenizer tokenizer.

    [`FLMRContextEncoderTokenizer`] is identical to [`BertTokenizer`] or [`AutoTokenizer`], depends on whether
    the tokenizer is initialized from bert.
    """
    def __init__(self) -> None:
        pass

    @classmethod
    def from_pretrained(
        cls,
        *args,
        text_config: Optional[FLMRTextConfig] = None,
        **kwargs,
    ):
        if text_config.text_encoder_base_model == "bert-base-uncased":
            return FLMRBertQueryEncoderTokenizer.from_pretrained(*args, **kwargs)
        else:
            return FLMRAutoQueryEncoderTokenizer(*args, query_maxlen=text_config.query_maxlen, **kwargs)