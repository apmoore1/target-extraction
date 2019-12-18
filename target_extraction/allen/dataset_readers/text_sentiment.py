import logging
import json
from typing import Dict, Union

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer
from allennlp.data.dataset_readers import TextClassificationJsonReader
from allennlp.data.instance import Instance
from overrides import overrides

logger = logging.getLogger(__name__)

@DatasetReader.register("text_sentiment")
class TextSentimentReader(TextClassificationJsonReader):
    """
    Subclasses :py:class:`allennlp.data.dataset_readers.TextClassificationJsonReader`
    of which the only differences is the `label_name` construction parameter 
    which is explained in the Parameters section below.

    Reads tokens and their labels from a labeled text classification dataset.
    Expects a "text" field and a "label" field in JSON format.
    The output of ``read`` is a list of ``Instance`` s with the fields:
        tokens: ``TextField`` and
        label: ``LabelField``
    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.
        See :class:`TokenIndexer`.
    tokenizer : ``Tokenizer``, optional (default = ``{"tokens": SpacyTokenizer()}``)
        Tokenizer to use to split the input text into words or other kinds of tokens.
    segment_sentences: ``bool``, optional (default = ``False``)
        If True, we will first segment the text into sentences using SpaCy and then tokenize words.
        Necessary for some models that require pre-segmentation of sentences, like the Hierarchical
        Attention Network (https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf).
    max_sequence_length: ``int``, optional (default = ``None``)
        If specified, will truncate tokens to specified maximum length.
    skip_label_indexing: ``bool``, optional (default = ``False``)
        Whether or not to skip label indexing. You might want to skip label indexing if your
        labels are numbers, so the dataset reader doesn't re-number them starting from 0.
    lazy : ``bool``, optional, (default = ``False``)
        Whether or not instances can be read lazily.
    label_name: ``str``, optional, (default = ``label``)
        The name of the label field in the JSON objects that are read.
    """

    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        tokenizer: Tokenizer = None,
        segment_sentences: bool = False,
        max_sequence_length: int = None,
        skip_label_indexing: bool = False,
        lazy: bool = False, label_name: str = 'label'
    ) -> None:
        super().__init__(token_indexers=token_indexers, tokenizer=tokenizer, 
                         segment_sentences=segment_sentences, 
                         max_sequence_length=max_sequence_length, 
                         skip_label_indexing=skip_label_indexing,
                         lazy=lazy)
        self._label_field = label_name

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            for line in data_file.readlines():
                if not line:
                    continue
                items = json.loads(line)
                text = items["text"]
                label = items.get(self._label_field, None)
                if label is not None:
                    if self._skip_label_indexing:
                        try:
                            label = int(label)
                        except ValueError:
                            raise ValueError(
                                "Labels must be integers if skip_label_indexing is True."
                            )
                    else:
                        label = str(label)
                instance = self.text_to_instance(text=text, label=label)
                if instance is not None:
                    yield instance

    @overrides
    def text_to_instance(self, text: str, label: Union[str, int] = None,
                         **kwargs) -> Instance:
        return super().text_to_instance(text=text, label=label)