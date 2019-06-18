import logging
import json
from typing import Dict, Any, Optional, List

from allennlp.common.file_utils import cached_path
from allennlp.common.checks import ConfigurationError
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.fields import TextField, SequenceLabelField, MetadataField, Field
from overrides import overrides

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("target_extraction")
class TargetExtractionDatasetReader(DatasetReader):
    '''
    Dataset reader designed to read a list of JSON like objects of the 
    following type:

    {`tokenized_text`: [`This`, `Camera`, `lens`, `is`, `great`], 
     `sequence_labels`: [`O`, `B`, `I`, `O`, `O`],
     `pos_tags`: [`DET`, `NOUN`, `NOUN`, `AUX`, `ADJ`]}

    Where the `pos_tags` are optional. This type of JSON can be created 
    from exporting a `target_extraction.data_types.TargetTextCollection` 
    using the `to_json_file` method.

    If the `pos_tags` are given, they can be used as either features or 
    for joint learning.
     
    The only sequence labels that we currently support is BIO or also known as 
    IOB-2.

    :params pos_tags: Whether or not to extract POS tags if avaliable.
    :returns: A ``Dataset`` of ``Instances`` for Target Extraction.
    '''
    def __init__(self, lazy: bool = False,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 pos_tags: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or \
                               {"tokens": SingleIdTokenIndexer()}
        self._pos_tags = pos_tags

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, 'r') as te_file:
            logger.info("Reading Target Extraction instances from jsonl "
                        "dataset at: %s", file_path)
            for line in te_file:
                example = json.loads(line)
                example_instance: Dict[str, Any] = {}

                sequence_labels = example["sequence_labels"]
                tokens_ = example["tokenized_text"]
                # TextField requires ``Token`` objects
                tokens = [Token(token) for token in tokens_]
                if self._pos_tags:
                    if 'pos_tags' not in example:
                        pos_err = (f"The POS tags are within the data: {example}"
                                  "\nPlease add them in manually or automatically"
                                  " to this dataset if you wish to use them.")
                        raise ConfigurationError(pos_err)
                    example_instance['pos_tags'] = example['pos_tags']

                example_instance['sequence_labels'] = sequence_labels
                example_instance['tokens'] = tokens
                yield self.text_to_instance(**example_instance)
    
    def text_to_instance(self, tokens: List[Token], 
                         sequence_labels: Optional[List[str]] = None,
                         pos_tags: Optional[List[str]] = None) -> Instance:
        '''
        The tokens are expected to be pre-tokenised.

        :param tokens: Tokenised text that either has target extraction labels 
                       or is to be tagged.
        :param sequence_labels: The target extraction BIO labels.
        :param pos_tags: POS tags to be used either as features or for joint 
                         learning.
        :returns: An Instance object with all of the above enocded for a
                  PyTorch model.
        '''
        sequence = TextField(tokens, self._token_indexers)
        instance_fields: Dict[str, Field] = {'tokens': sequence}
        instance_fields["metadata"] = MetadataField({"words": [x.text for x in tokens]})

        if sequence_labels is not None:
            instance_fields['labels'] = SequenceLabelField(sequence_labels, sequence, "labels")

        if pos_tags is not None:
            instance_fields['pos_tags'] = SequenceLabelField(pos_tags, sequence, "pos_tags")
        return Instance(instance_fields)