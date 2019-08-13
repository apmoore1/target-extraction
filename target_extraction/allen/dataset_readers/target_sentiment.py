import logging
import json
from typing import Dict, Any, Optional, List, Union

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.fields import TextField, ListField, MetadataField, Field
from allennlp.data.fields import SequenceLabelField
from overrides import overrides

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("target_sentiment")
class TargetSentimentDatasetReader(DatasetReader):
    '''
    Dataset reader designed to read a list of JSON like objects of the 
    following type:

    {`text`: `This Camera lens is great`, 
     `targets`: [`Camera`],
     `target_sentiments`: [`positive`]}

    or

    {`text`: `This Camera lens is great`, 
     `categories`: [`CAMERA`],
     `category_sentiments`: [`positive`]}

    or

    {`text`: `This Camera lens is great`, 
     `targets`: [`Camera`]
     `categories`: [`CAMERA`],
     `target_sentiments`: [`positive`]}

    This type of JSON can be created from exporting a 
    `target_extraction.data_types.TargetTextCollection` using the 
    `to_json_file` method.

    The difference between the three objects depends on the objective of the 
    model being trained:  
    1. Version is for a purely Target based sentiment classifier.
    2. Version is for a purely Aspect or latent based sentiment classifier.
    3. Version is if you want to make use of the relationship between the 
       Target and Aspect in the sentiment classifier.

    :params token_indexers: We use this to define the input representation 
                            for the text. See 
                            :class:`allennlp.data.token_indexers.TokenIndexer`.
    :params tokenizer: Tokenizer to use to split the sentence text as well as 
                       the text of the target.
    :params lazy: Whether or not instances can be read lazily.
    '''
    def __init__(self, lazy: bool = False,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tokenizer: Tokenizer = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or \
                               {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, 'r') as te_file:
            logger.info("Reading Target Sentiment instances from jsonl "
                        "dataset at: %s", file_path)
            for line in te_file:
                example = json.loads(line)
                example_instance: Dict[str, Any] = {}

                example_instance["text"] = example["text"]
                if 'target_sentiments' in example and 'targets' in example:
                    example_instance['targets'] = example['targets']
                    example_instance['target_sentiments'] = example['target_sentiments']
                if 'categories' in example:
                    example_instance['categories'] = example['categories']
                if 'category_sentiments' in example:
                    example_instance['category_sentiments'] = example['category_sentiments']
                yield self.text_to_instance(**example_instance)

    def text_to_instance(self, text: str, 
                         targets: Optional[List[str]] = None,
                         target_sentiments: Optional[List[Union[str, int]]] = None,
                         categories: Optional[List[str]] = None,
                         category_sentiments: Optional[List[Union[str, int]]]= None
                         ) -> Instance:
        '''
        The original text, text tokens as well as the targets and target 
        tokens are stored in the MetadataField.

        :NOTE: At least targets and/or categories must be present.

        :param text: The text that contains the target(s) and/or categories.
        :param targets: The targets that are within the text
        :param target_sentiments: The sentiment of the targets. To be used if 
                                  training the classifier
        :param categories: The categories that are within the text
        :param category_sentiments: The sentiment of the categories
        :returns: An Instance object with all of the above enocded for a
                  PyTorch model.
        '''
        instance_fields: Dict[str, Field] = {}
        
        tokens = self._tokenizer.tokenize(text)
        instance_fields['tokens'] = TextField(tokens, self._token_indexers)

        # Metadata field
        metadata_dict = {}
        metadata_dict['text words'] = [x.text for x in tokens]
        metadata_dict['text'] = text

        if targets is not None:
            all_target_tokens = [self._tokenizer.tokenize(target) 
                                 for target in targets]
            target_fields = [TextField(target_tokens, self._token_indexers)  
                             for target_tokens in all_target_tokens]
            target_fields = ListField(target_fields)
            instance_fields['targets'] = target_fields
            # Target sentiment if it exists
            if target_sentiments is not None:
                target_sentiments_field = SequenceLabelField(target_sentiments, 
                                                             target_fields)
                instance_fields['target_sentiments'] = target_sentiments_field
            # Add the targets and the tokenised targets to the metadata
            metadata_dict['targets'] = [target for target in targets]
            metadata_dict['target words'] = [[x.text for x in target_tokens] 
                                             for target_tokens in all_target_tokens]
        if categories is not None:
            category_fields = TextField([Token(category) for category in categories], 
                                        self._token_indexers)
            instance_fields['categories'] = category_fields
            # Category sentiment if it exists
            if category_sentiments is not None:
                category_sentiments_field = SequenceLabelField(category_sentiments, 
                                                               category_fields)
                instance_fields['category_sentiments'] = category_sentiments_field
            # Add the categories to the metadata
            metadata_dict['categories'] = [category for category in categories]
        
        if targets is None and categories is None:
            raise ValueError('Either targets or categories must be given if you '
                             'want to be predict the sentiment of a target '
                             'or a category')

        instance_fields["metadata"] = MetadataField(metadata_dict)
        
        return Instance(instance_fields)