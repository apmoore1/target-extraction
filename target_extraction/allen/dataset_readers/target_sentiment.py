import logging
import json
from typing import Dict, Any, Optional, List, Union, NamedTuple

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.fields import TextField, ListField, MetadataField, Field
from allennlp.data.fields import SequenceLabelField, ArrayField
from overrides import overrides
import numpy as np

from target_extraction.data_types import TargetText
from target_extraction.data_types_util import Span

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class TargetToken(NamedTuple):
    '''
    :param is_target: The length of the list denotes the number of targets within
                      the text the token came from. The value if `1` denotes 
                      that the token is a Target, `0` not a target. Each index 
                      denotes a different multi word target within the text 
                      the token came from. 
    '''
    is_target: List[int]


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

    or 

    {`text`: `This Camera lens is great`, 
     `targets`: [`Camera`],
     `target_sentiments`: [`positive`],
     `spans`: [[5,11]]}

    This type of JSON can be created from exporting a 
    `target_extraction.data_types.TargetTextCollection` using the 
    `to_json_file` method.

    The difference between the three objects depends on the objective of the 
    model being trained:  
    1. Version is for a purely Target based sentiment classifier.
    2. Version is for a purely Aspect or latent based sentiment classifier.
    3. Version is if you want to make use of the relationship between the 
       Target and Aspect in the sentiment classifier.
    4. If the Target based sentiment classifier requires the knowledge of 
       where the target is.

    :param lazy: Whether or not instances can be read lazily.
    :param token_indexers: We use this to define the input representation 
                            for the text. See 
                            :class:`allennlp.data.token_indexers.TokenIndexer`.
    :param tokenizer: Tokenizer to use to split the sentence text as well as 
                       the text of the target.
    :param left_right_contexts: If True it will return within the 
                                instance for `text_to_instance` the 
                                sentence context left and right of the target.
    :param reverse_right_context: If True this will reverse the text that is 
                                  in the right context. NOTE left_right_context 
                                  has to be True.
    :param incl_target: If left_right_context is True and this also 
                        the left and right contexts will include the target
                        word(s) as well.
    :param use_categories: Whether or not to return the categories in the 
                           instances even if they do occur in the dataset. 
                           This is a temporary solution to the following 
                           `issue <https://github.com/apmoore1/target-extraction/issues/5>`_.
                           The number of categories does not have to match the 
                           number of targets, just there has to be at least one 
                           category per sentence. 
    :param target_sequences: Whether or not to generate `target_sequences` 
                             which are a sequence of masks per target for all 
                             target texts. This will allow the model to know 
                             which tokens in the context relate to the target.
                             Example of this is shown below (for this to work 
                             does require the `span` of each target)
    :param position_embeddings: Whether or not to create distance values 
                                that can be converted to embeddings similar 
                                to the `position_weights` but instead of the 
                                model later on using them as weights it uses 
                                the distances to learn position embeddings.
                                (for this to work does require the `span` of 
                                each target). `A Position-aware Bidirectional 
                                Attention Network for Aspect-level Sentiment 
                                Analysis <https://www.aclweb.org/anthology/C18-1066.pdf>`_
    :param position_weights: In the instances there will be an extra key 
                             `position_weights` which will be an array of 
                             integers representing the linear distance between 
                             each token and it's target e.g. If the text 
                             contains two targets where each token is represented
                             by a number and the 1's target tokens = 
                             [[0,0,0,1], [1,1,0,0]] then the `position_weights`
                             will be [[4,3,2,1], [1,1,2,3]]. (for this to work 
                             does require the `span` of each target). An example 
                             of position weighting is in section 3.3 of 
                             `Modeling Sentiment Dependencies with Graph 
                             Convolutional Networks for Aspect-level Sentiment 
                             Classification <https://arxiv.org/pdf/1906.04501.pdf>`_
    :param max_position_distance: The maximum position distance given to a token 
                                  from the target e.g. [0,0,0,0,0,1,0,0] if the 
                                  each value represents a token and 1's represent
                                  target tokens then the distance array would be 
                                  [6,5,4,3,2,1,2,3] if the `max_position_distance`
                                  is 5 then the distance array will be 
                                  [5,5,4,3,2,1,2,3]. (for this to work either 
                                  `position_embeddings` has to be True or 
                                  `position_weights`)
    :raises ValueError: If the `left_right_contexts` is not True while either the 
                        `incl_targets` or `reverse_right_context` arguments are 
                        True.
    :raises ValueError: If the `left_right_contexts` and `target_sequences` are 
                        True at the same time.
    :raises ValueError: If the `max_position_distance` when set is less than 2.
    :raises ValueError: If `max_position_distance` is set but neither 
                        `position_embeddings` nor `position_weights` are 
                        `True`. 

    :Example of target_sequences: {`text`: `This Camera lens is great but the 
                                            screen is rubbish`, 
                                   `targets`: [`Camera`, `screen`],
                                   `target_sentiments`: [`positive`, `negative`],
                                   `target_sequences`: [[0,1,0,0,0,0,0,0,0,0], 
                                                        [0,0,0,0,0,0,0,1,0,0]],
                                   `spans`: [[5,11], [34:40]]}
    '''
    def __init__(self, lazy: bool = False,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tokenizer: Tokenizer = None,
                 left_right_contexts: bool = False,
                 reverse_right_context: bool = False,
                 incl_target: bool = False,
                 use_categories: bool = False,
                 target_sequences: bool = False,
                 position_embeddings: bool = False,
                 position_weights: bool = False,
                 max_position_distance: Optional[int] = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or \
                               {"tokens": SingleIdTokenIndexer()}
        if incl_target and not left_right_contexts:
            raise ValueError('If `incl_target` is True then `left_right_contexts`'
                             ' argument also has to be True')
        if reverse_right_context and not left_right_contexts:
            raise ValueError('If `reverse_right_context` is True then '
                             '`left_right_contexts` argument also has to be True')
        self._incl_target = incl_target
        self._reverse_right_context = reverse_right_context
        self._left_right_contexts = left_right_contexts
        self._use_categories = use_categories
        self._target_sequences = target_sequences

        if self._left_right_contexts and self._target_sequences:
            raise ValueError('Cannot have both `left_right_contexts` and '
                             '`target_sequences` True at the same time either'
                             ' one or the other or None.')
        if (not position_embeddings and not position_weights and 
            max_position_distance is not None):
            raise ValueError('`max_position_distance` contains a value '
                             f'{max_position_distance} When neither `position'
                             '_embeddings` nor `position_weights` are True')
        self._position_embeddings = position_embeddings
        if position_embeddings:
            self._position_indexers = {"position_tokens": SingleIdTokenIndexer()}
        self._position_weights = position_weights
        self._max_position_distance = max_position_distance

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
                if 'spans' in example:
                    example_instance['spans'] = example['spans']
                yield self.text_to_instance(**example_instance)

    def _add_context_field(self, sentence_contexts: List[str]) -> ListField:
        context_fields = []
        for context in sentence_contexts:
            tokens = self._tokenizer.tokenize(context)
            context_field = TextField(tokens, self._token_indexers)
            context_fields.append(context_field)
        return ListField(context_fields)

    @staticmethod
    def _target_indicators_to_distances(target_indicators: List[List[int]],
                                        max_distance: Optional[int] = None,
                                        as_string: bool = False
                                        ) -> List[List[Union[int,str]]]:
        '''
        :param target_indicators: For a text the outer list represents the number 
                                  of targets in the sentence and the inner list 
                                  are 0's representing no target tokens and 1's 
                                  representing targets for one potential multi
                                  word target in that text. e.g. [[0,0,1,1,0], [1,0,0,0,0]]
                                  this would mean the text has two targets where 
                                  the first is a multi word target and the second 
                                  is a single word target.
        :param max_distance: The maximum distance that can be given.
        :param as_string: Whether the integers should become string value. Required
                          if you want to use these as position embeddings.
        :returns: A list of a list where the outer list represents the number of 
                  targets in the text and the inner represents the distance the 
                  tokens are to those targets e.g. using the example in 
                  `target_indicators` the return would be [[3,2,1,1,2], [1,2,3,4,5]]
        '''
        if max_distance is not None:
            if max_distance < 2:
                distance_error = ('Max distance has to be greater than 1. '
                                  f'Currently max distance is {max_distance}')
                raise ValueError(distance_error)
        target_indicator_distances: List[List[int]] = []
        for target_indicator_list in target_indicators:
            target_indicator_distance: List[int] = []
            first_one = target_indicator_list.index(1)
            # tokens up to the target
            if first_one == 0:
                pass
            else:
                for distance in reversed(range(first_one)):
                    distance = distance + 2
                    if max_distance is not None:
                        if distance > max_distance:
                            distance = max_distance
                    target_indicator_distance.append(distance)
            # https://stackoverflow.com/questions/522372/finding-first-and-last-index-of-some-value-in-a-list-in-python
            last_one = len(target_indicator_list) - 1 - target_indicator_list[::-1].index(1)
            length_of_target = (last_one - first_one) + 1
            # tokens in the target
            for _ in range(length_of_target):
                target_indicator_distance.append(1)
            # tokens after the target
            number_tokens_left = (len(target_indicator_list) - last_one) - 1
            for distance in range(number_tokens_left):
                distance = distance + 2
                if max_distance is not None:
                    if distance > max_distance:
                        distance = max_distance
                target_indicator_distance.append(distance)
            assert len(target_indicator_list) == len(target_indicator_distance)
            # to string 
            if as_string:
                target_indicator_distance = [str(distance) for distance in target_indicator_distance]
            target_indicator_distances.append(target_indicator_distance)
        return target_indicator_distances

    def text_to_instance(self, text: str, 
                         targets: Optional[List[str]] = None,
                         target_sentiments: Optional[List[Union[str, int]]] = None,
                         spans: Optional[List[List[int]]] = None,
                         categories: Optional[List[str]] = None,
                         category_sentiments: Optional[List[Union[str, int]]] = None,
                         **kwargs) -> Instance:
        '''
        The original text, text tokens as well as the targets and target 
        tokens are stored in the MetadataField.

        :NOTE: At least targets and/or categories must be present.
        :NOTE: That the left and right contexts returned in the instance are 
               a List of a List of tokens. A list for each Target.

        :param text: The text that contains the target(s) and/or categories.
        :param targets: The targets that are within the text
        :param target_sentiments: The sentiment of the targets. To be used if 
                                  training the classifier
        :param spans: The spans that represent the character offsets for each 
                      of the targets given in the targets list.
        :param categories: The categories that are within the text
        :param category_sentiments: The sentiment of the categories
        :returns: An Instance object with all of the above encoded for a
                  PyTorch model.
        :raises ValueError: If either targets and categories are both None
        :raises ValueError: If `self._target_sequences` is True and the passed 
                            `spans` argument is None.
        :raises ValueError: If `self._left_right_contexts` is True and the 
                            passed `spans` argument is None.
        '''
        if targets is None and categories is None:
            raise ValueError('Either targets or categories must be given if you '
                             'want to be predict the sentiment of a target '
                             'or a category')

        instance_fields: Dict[str, Field] = {}
        

        # Metadata field
        metadata_dict = {}

        if targets is not None:
            # need to change this so that it takes into account the case where 
            # the positions are True but not the target sequences.
            if self._target_sequences or self._position_embeddings or self._position_weights:
                if spans is None:
                    raise ValueError('To create target sequences requires `spans`')
                spans = [Span(span[0], span[1]) for span in spans]
                target_text_object = TargetText(text=text, spans=spans, 
                                                targets=targets, text_id='anything')
                target_text_object.force_targets()
                text = target_text_object['text']
                allen_tokens = self._tokenizer.tokenize(text)
                tokens = [x.text for x in allen_tokens]
                target_text_object['tokenized_text'] = tokens
                target_text_object.sequence_labels(per_target=True)
                target_sequences = target_text_object['sequence_labels']
                # Need to add the target sequences to the instances
                in_label = {'B', 'I'}
                number_targets = len(targets)
                all_target_tokens: List[List[Token]] = [[] for _ in range(number_targets)]
                target_sequence_fields = []
                target_indicators: List[List[int]] = []
                for target_index in range(number_targets):
                    one_values = []
                    target_ones = [0] * len(allen_tokens)
                    for token_index, token in enumerate(allen_tokens):
                        target_sequence_value = target_sequences[target_index][token_index]
                        in_target = 1 if target_sequence_value in in_label else 0
                        if in_target:
                            all_target_tokens[target_index].append(allen_tokens[token_index])
                            one_value_list = [0] * len(allen_tokens)
                            one_value_list[token_index] = 1
                            one_values.append(one_value_list)
                            target_ones[token_index] = 1
                    one_values = np.array(one_values)
                    target_sequence_fields.append(ArrayField(one_values, dtype=np.int32))
                    target_indicators.append(target_ones)
                if self._position_embeddings:
                    target_distances = self._target_indicators_to_distances(target_indicators, 
                                                                            max_distance=self._max_position_distance, 
                                                                            as_string=True)
                    target_text_distances = []
                    for target_distance in target_distances:
                        token_distances = [Token(distance) for distance in target_distance]
                        token_distances = TextField(token_distances, self._position_indexers)
                        target_text_distances.append(token_distances)
                    instance_fields['position_embeddings'] = ListField(target_text_distances)
                if self._position_weights:
                    target_distances = self._target_indicators_to_distances(target_indicators, 
                                                                            max_distance=self._max_position_distance, 
                                                                            as_string=False)
                    target_distances = np.array(target_distances)
                    instance_fields['position_weights'] = ArrayField(target_distances, 
                                                                     dtype=np.int32)
                if self._target_sequences:
                    instance_fields['target_sequences'] = ListField(target_sequence_fields)
                instance_fields['tokens'] = TextField(allen_tokens, self._token_indexers)
                metadata_dict['text words'] = tokens
                metadata_dict['text'] = text
                # update target variable as the targets could have changed due 
                # to the force_targets function
                targets = target_text_object['targets']
            else:
                all_target_tokens = [self._tokenizer.tokenize(target) 
                                     for target in targets]
            target_fields = [TextField(target_tokens, self._token_indexers)  
                            for target_tokens in all_target_tokens]
            target_fields = ListField(target_fields)
            instance_fields['targets'] = target_fields
            # Add the targets and the tokenised targets to the metadata
            metadata_dict['targets'] = [target for target in targets]
            metadata_dict['target words'] = [[x.text for x in target_tokens] 
                                             for target_tokens in all_target_tokens]

            # Target sentiment if it exists
            if target_sentiments is not None:
                target_sentiments_field = SequenceLabelField(target_sentiments, 
                                                             target_fields,
                                                             label_namespace='target-sentiment-labels')
                instance_fields['target_sentiments'] = target_sentiments_field

        if categories is not None and self._use_categories:
            category_fields = TextField([Token(category) for category in categories], 
                                        self._token_indexers)
            instance_fields['categories'] = category_fields
            # Category sentiment if it exists
            if category_sentiments is not None:
                category_sentiments_field = SequenceLabelField(category_sentiments, 
                                                               category_fields,
                                                               label_namespace='category-sentiment-labels')
                instance_fields['category_sentiments'] = category_sentiments_field
            # Add the categories to the metadata
            metadata_dict['categories'] = [category for category in categories]

        if 'tokens' not in instance_fields:
            tokens = self._tokenizer.tokenize(text)
            instance_fields['tokens'] = TextField(tokens, self._token_indexers)
            metadata_dict['text'] = text
            metadata_dict['text words'] = [x.text for x in tokens]

        # If required processes the left and right contexts
        left_contexts = None
        right_contexts = None
        if self._left_right_contexts:
            if spans is None:
                raise ValueError('To create left, right, target contexts requires'
                                 ' the `spans` of the targets which is None')
            spans = [Span(span[0], span[1]) for span in spans]
            target_text_object = TargetText(text=text, spans=spans, 
                                            targets=targets, text_id='anything')
            # left, right, and target contexts for each target in the 
            # the text
            left_right_targets = target_text_object.left_right_target_contexts(incl_target=self._incl_target)
            left_contexts: List[str] = []
            right_contexts: List[str] = []
            for left_right_target in left_right_targets:
                left, right, _ = left_right_target
                left_contexts.append(left)
                if self._reverse_right_context:
                    right_tokens = self._tokenizer.tokenize(right)
                    reversed_right_tokens = []
                    for token in reversed(right_tokens):
                        reversed_right_tokens.append(token.text)
                    right = ' '.join(reversed_right_tokens)
                right_contexts.append(right)
        
        if left_contexts is not None:
            left_field = self._add_context_field(left_contexts)
            instance_fields["left_contexts"] = left_field
        if right_contexts is not None:
            right_field = self._add_context_field(right_contexts)
            instance_fields["right_contexts"] = right_field

        instance_fields["metadata"] = MetadataField(metadata_dict)
        
        return Instance(instance_fields)