'''
Moudle that contains the two main data types 
`target_extraction.data_types.TargetText` and 
`target_extraction.data_types.TargetTextCollection` where the later is a
container for the former.

classes:

1. `target_extraction.data_types.TargetText`
2. `target_extraction.data_types.TargetTextCollection`
'''
from collections.abc import MutableMapping
from collections import OrderedDict, Counter, defaultdict
import copy
import json
from pathlib import Path
from typing import Optional, List, Tuple, Iterable, NamedTuple, Any, Callable
from typing import Union, Dict, Set

from target_extraction.tokenizers import is_character_preserving, token_index_alignment
from target_extraction.data_types_util import Span

class TargetText(MutableMapping):
    '''
    This is a data structure that inherits from MutableMapping which is 
    essentially a python dictionary.

    The following are the default keys that are in all `TargetText` 
    objects, additional items can be added through __setitem__ but the default 
    items cannot be deleted.

    1. text - The text associated to all of the other items
    2. text_id -- The unique ID associated to this object 
    3. targets -- List of all target words that occur in the text. A special 
                  placeholder of None (python None value) can exist where the 
                  target does not exist but a related Category does this would 
                  mean though that the related span is Span(0, 0), this type of 
                  special placeholder is in place for the SemEval 2016 Restaurant 
                  dataset where they link the categories to the targets but 
                  not all categories have related targets thus None.
    4. spans -- List of Span NamedTuples where each one specifies the start and 
       end of the respective targets within the text.
    5. target_sentiments -- List sepcifying the sentiment of the respective 
       targets within the text.
    6. categories -- List of categories that exist in the data which may or 
       may not link to the targets (this is dataset speicific). NOTE: 
       depending on the dataset and how it is parsed the category can exist 
       but the target does not as the category is a latent variable, in 
       these cases the category and category sentiments will be the same size 
       which would be a different size to the target and target sentiments 
       size. E.g. can happen where the dataset has targets and categories 
       but they do not map to each other in a one to one manner e.g 
       SemEval 2014 restuarant dataset, there are some samples that contain 
       categories but no targets. Another word for category can be aspect.
    7. category_sentiments -- List of the sentiments associated to the 
       categories. If the categories and targets map to each other then 
       this will be empty and you will only use the target_sentiments.

    Methods:
    
    1. to_json -- Returns the object as a dictionary and then encoded using 
       json.dumps
    2. tokenize -- This will add a new key `tokenized_text` to this TargetText 
       instance that will store the tokens of the text that is associated to 
       this TargetText instance.
    3. pos_text -- This will add a new key `pos_tags` to this TargetText 
       instance. This key will store the pos tags of the text that is 
       associated to this Target Text instance.
    4. force_targets -- Does not return anything but modifies the `spans` and 
       `text` values as whitespace is prefixed and suffixed the target unless 
       the prefix or suffix is whitespace. NOTE that this is the only method 
       that currently can change the `spans` and `text` key values after they 
       have been set.
    5. sequence_labels -- Adds the `sequence_labels` key to this TargetText 
       instance which can be used to train a machine learning algorthim to 
       detect targets.
    6. get_sequence_indexs -- The indexs related to the tokens, pos tags etc 
       for each labelled sequence span.
    7. get_sequence_spans -- The span indexs from the sequence labels given 
       assuming that the sequence labels are in BIO format.
    8. get_targets_from_sequence_labels -- Retrives the target words given the 
       sequence labels.
    9. one_sample_per_span -- This returns a similar TargetText instance 
       where the new instance will only contain one target per span.
    10. left_right_target_contexts -- This will return the sentence that is 
        left and right of the target as well as the words in the target for 
        each target in the sentence.
    
    Static Functions:

    1. from_json -- Returns a TargetText object given a json string. For 
       example the json string can be the return of TargetText.to_json.
    2. targets_from_spans -- Given a sequence of spans and the associated text 
       it will return the targets that are within the text based on the spans
    3. target_text_from_prediction -- Creates a TargetText object from data 
       that has come from predictions of a Target Extract tagger
    '''

    def _check_is_list(self, item: List[Any], item_name: str) -> None:
        '''
        This will check that the argument given is a List and if not will raise 
        a TypeError.

        :param item: The argument that is going to be checked to ensure it is a
                     list.
        :param item_name: Name of the item. This is used within the raised 
                          error message, if an error is raised.
        :raises TypeError: If any of the items are not of type List.
        '''
        type_err = f'{item_name} should be a list not {type(item)} {item}'
        if not isinstance(item, list):
            raise TypeError(type_err)

    def sanitize(self) -> None:
        '''
        This performs a check on all of the lists that can be given at 
        object construction time to ensure that the following conditions are 
        met:
        
        1. The target, spans and target_sentiments lists are all of the same 
           size if set.
        2. The categories and the category_sentiments lists are all of the 
           same size if set. 

        Further more it checks the following:

        1. If targets or spans are set then both have to exist.
        2. If targets and spans are set that the spans text match the 
           associated target words e.g. if the target is `barry davies` in 
           `today barry davies went` then the spans should be [[6,18]]
    
        :raises ValueError: If any of the above conditions are not True.
        '''

        def length_mis_match(lists_to_check: List[Any], 
                             text_id_msg: str) -> None:
            length_mismatch_msg = 'The following lists do not match '\
                                  f'{lists_to_check}'
            list_lengths = [len(_list) for _list in lists_to_check 
                            if _list is not None]
            current_list_size = -1
            for list_length in list_lengths:
                if current_list_size == -1:
                    current_list_size = list_length
                else:
                    if current_list_size != list_length:
                        raise ValueError(text_id_msg + length_mismatch_msg)

        targets = self._storage['targets']
        target_sentiments = self._storage['target_sentiments']
        spans = self._storage['spans']
        categories = self._storage['categories']
        category_sentiments = self._storage['category_sentiments']

        text_id = self._storage['text_id']
        text_id_msg = f'Text id that this error refers to {text_id}\n'

        # Checking the length mismatches for the two different lists
        length_mis_match([targets, target_sentiments, spans], text_id_msg)
        length_mis_match([categories, category_sentiments], text_id_msg)
        

        # Checking that if targets are set than so are spans
        if targets is not None and spans is None:
            spans_none_msg = f'If the targets are a list: {targets} then spans'\
                             f' should also be a list and not None: {spans}'
            raise ValueError(text_id_msg + spans_none_msg)
        # Checking that the words Spans reference in the text match the 
        # respective target words. Edge case is the case of None targets which 
        # should have a Span value of (0, 0)
        if targets is not None:
            text = self._storage['text'] 
            for target, span in zip(targets, spans):
                if target is None:
                    target_span_msg = 'As the target value is None the span '\
                                      'it refers to should be of value '\
                                      f'Span(0, 0) and not {span}'
                    if span != Span(0, 0):
                        raise ValueError(text_id_msg + target_span_msg)
                else:
                    start, end = span.start, span.end
                    text_target = text[start:end]
                    target_span_msg = 'The target the spans reference in the '\
                                      f'text: {text_target} does not match '\
                                      f'the target in the targets list: {target}'
                    if text_target != target:
                        raise ValueError(text_id_msg + target_span_msg)

    def __init__(self, text: str, text_id: str,
                 targets: Optional[List[str]] = None, 
                 spans: Optional[List[Span]] = None, 
                 target_sentiments: Optional[List[Union[int, str]]] = None, 
                 categories: Optional[List[str]] = None,
                 category_sentiments: Optional[List[Union[int, str]]] = None,
                 **additional_data):
        '''
        :param additional_data: Any other data that is to be added to the 
                                object at construction.
        '''
        # Ensure that the arguments that should be lists are lists.
        self._list_argument_names = ['targets', 'spans', 'target_sentiments', 
                                     'categories', 'category_sentiments']
        self._list_arguments = [targets, spans, target_sentiments, categories,
                                category_sentiments]
        names_arguments = zip(self._list_argument_names, self._list_arguments)
        for argument_name, list_argument in names_arguments:
            if list_argument is None:
                continue
            self._check_is_list(list_argument, argument_name)

        temp_dict = dict(text=text, text_id=text_id, targets=targets,
                         spans=spans, target_sentiments=target_sentiments, 
                         categories=categories, 
                         category_sentiments=category_sentiments)
        self._protected_keys = set(['text', 'text_id', 'targets', 'spans'])
        self._storage = temp_dict
        self._storage = {**self._storage, **additional_data}
        self.sanitize()

    def __getitem__(self, key: str) -> Any:
        '''
        :returns: One of the values from the self._storage dictionary. e.g. 
                  if the key is `text` it will return the string representing 
                  the text associated to this object.
        '''
        return self._storage[key]

    def __iter__(self) -> Iterable[str]:
        '''
        Returns an interator over the keys in self._storage which are the 
        following Strings by default additional keys can be added:

        1. text
        2. text_id
        3. targets
        4. spans
        5. target_sentiments
        6. categories
        7. category_sentiments

        :returns: The keys in self._storage
        '''
        return iter(self._storage)

    def __len__(self) -> int:
        '''
        :returns: The number of items in self._storage.
        '''
        return len(self._storage)
    
    def __repr__(self) -> str:
        '''
        :returns: String returned is what user see when the instance is 
                  printed or printed within a interpreter.
        '''
        return f'TargetText({self._storage})'

    def __eq__(self, other: 'TargetText') -> bool:
        '''
        Two TargetText instances are equal if they both have the same `text_id`
        value.

        :param other: Another TargetText object that is being compared to this 
                      TargetText object.
        :returns: True if they have the same `text_id` value else False.
        '''

        if not isinstance(other, TargetText):
            return False
        elif self['text_id'] != other['text_id']:
            return False
        return True

    def __delitem__(self, key: str) -> None:
        '''
        Given a key that matches a key within self._storage or self.keys() 
        it will delete that key and value from this object.

        NOTE: Currently  'text', 'text_id', 'spans', and 'targets' are keys 
        that cannot be deleted.

        :param key: Key and its respective value to delete from this object.
        '''
        if key in self._protected_keys:
            raise KeyError('Cannot delete a key that is protected, list of '
                           f' protected keys: {self._protected_keys}')
        del self._storage[key]

    def __setitem__(self, key: str, value: Any) -> None:
        '''
        Given a key and a respected value it will either change that current 
        keys value to the one gien here or create a new key with that value.

        NOTE: Currently  'text', 'text_id', 'spans', and 'targets' are keys 
        that cannot be changed.

        :param key: Key to be added or changed
        :param value: Value associated to the given key.
        '''
        if key in self._protected_keys:
            raise KeyError('Cannot change a key that is protected, list of '
                           f' protected keys: {self._protected_keys}')
        # If the key value should be a list ensure that the new value is a 
        # list as well.
        if key in self._list_argument_names:
            self._check_is_list(value, key)
        self._storage[key] = value
        self.sanitize()

    def to_json(self) -> str:
        '''
        Required as TargetText is not json serlizable due to the 'spans'.

        :returns: The object as a dictionary and then encoded using json.dumps
        '''
        return json.dumps(self._storage)

    def _shift_spans(self, num_shifts: int, target_span: Span) -> None:
        '''
        This only affects the current state of the TargetText attributes. 
        The attributes this affects is the `spans` attribute.

        NOTE: This is only used within self.force_targets method.

        :param num_shifts: The number of whitespaces that the target at 
                            span_index is going to be added. 1 if it is 
                            just prefix or suffix space added, 2 if both or 
                            0 if none.
        :param spans: The current target span indexs that are having extra 
                        whitespace added either prefix or suffix.
        '''
        relevant_span_indexs: List[int] = []
        target_span_end = target_span.end
        for span_index, other_target_span in enumerate(self['spans']):
            if other_target_span == target_span:
                continue
            elif other_target_span.start > target_span_end:
                relevant_span_indexs.append(span_index)

        for relevant_span_index in relevant_span_indexs:
            start, end = self['spans'][relevant_span_index]
            start += num_shifts
            end += num_shifts
            self._storage['spans'][relevant_span_index] = Span(start, end)

    def force_targets(self) -> None:
        '''
        :NOTE: As this affects the following attributes `spans` and `text` it 
        therefore has to modify these through self._storage as both of these 
        attributes are within self._protected_keys.

        Does not return anything but modifies the `spans` and `text` values 
        as whitespace is prefixed and suffixed the target unless the prefix 
        or suffix is whitespace.

        Motivation:
        Ensure that the target tokens are not within another seperate String 
        e.g. target = `priced` but the sentence is `the laptop;priced is high` 
        and the tokenizer is on whitespace it will not have `priced` seperated 
        therefore the BIO tagging is not determinstric thus force will add 
        whitespace around the target word e.g. `the laptop; priced`. This was 
        mainly added for the TargetText.sequence_tags method.
        '''
    
        for span_index in range(len(self['spans'])):
            text = self['text']
            last_token_index = len(text) - 1

            span = self['spans'][span_index]
            prefix = False
            suffix = False

            start, end = span
            if start != 0:
                if text[start - 1] != ' ':
                    prefix = True
            if end < last_token_index:
                if text[end] != ' ':
                    suffix = True

            text_before = text[:start]
            text_after = text[end:]
            target = text[start:end]
            if prefix and suffix:
                self._storage['text'] = f'{text_before} {target} {text_after}'
                self._shift_spans(2, span)
                self._storage['spans'][span_index] = Span(start + 1, end + 1)
            elif prefix:
                self._storage['text'] = f'{text_before} {target}{text_after}'
                self._shift_spans(1, span)
                self._storage['spans'][span_index] = Span(start + 1, end + 1)
            elif suffix:
                self._storage['text'] = f'{text_before}{target} {text_after}'
                self._shift_spans(1, span)

    def tokenize(self, tokenizer: Callable[[str], List[str]],
                 perform_type_checks: bool = False) -> None:
        '''
        This will add a new key `tokenized_text` to this TargetText instance
        that will store the tokens of the text that is associated to this 
        TargetText instance.

        For a set of tokenizers that are definetly comptable see 
        target_extraction.tokenizers module.

        Ensures that the tokenization is character preserving.

        :param tokenizer: The tokenizer to use tokenize the text for each 
                          TargetText instance in the current collection
        :param perform_type_checks: Whether or not to perform type checks 
                                    to ensure the tokenizer returns a List of 
                                    Strings
        :raises TypeError: If the tokenizer given does not return a List of 
                           Strings.
        :raises ValueError: This is raised if the TargetText instance contains
                            empty text.
        :raises ValueError: If the tokenization is not character preserving.
        '''
        text = self['text']
        tokenized_text = tokenizer(text)
        if perform_type_checks:
            if not isinstance(tokenized_text, list):
                raise TypeError('The return type of the tokenizer function ',
                                f'{tokenizer} should be a list and not '
                                f'{type(tokenized_text)}')
            for token in tokenized_text:
                if not isinstance(token, str):
                    raise TypeError('The return type of the tokenizer function ',
                                    f'{tokenizer} should be a list of Strings'
                                    f' and not a list of {type(token)}')

        if len(tokenized_text) == 0:
            raise ValueError('There are no tokens for this TargetText '
                             f'instance {self}')
        if not is_character_preserving(text, tokenized_text):
            raise ValueError('The tokenization method used is not character'
                             f' preserving. Original text `{text}`\n'
                             f'Tokenized text `{tokenized_text}`')
        self['tokenized_text'] = tokenized_text
    
    def pos_text(self, tagger: Callable[[str], Tuple[List[str], List[str]]], 
                 perform_type_checks: bool = False) -> None:
        '''
        This will add a new key `pos_tags` to this TargetText instance.
        This key will store the pos tags of the text that is associated to 
        this Target Text instance. NOTE: It will also replace the current 
        tokens in the `tokenized_text` key with the tokens produced 
        from the pos tagger.

        For a set of pos taggers that are definetly comptable see 
        target_extraction.pos_taggers module. The pos tagger will have to 
        produce both a list of tokens and pos tags.

        :param tagger: POS tagger.
        :param perform_type_checks: Whether or not to perform type checks 
                                    to ensure the POS tagger returns a 
                                    tuple containing two lists both containing 
                                    Strings.
        :raises TypeError: If the POS tagger given does not return a Tuple
        :raises TypeError: If the POS tagger given does not return a List of 
                           Strings for both the tokens and the pos tags.
        :raises TypeError: If the POS tagger tokens or pos tags are not lists
        :raises ValueError: If the POS tagger return is not a tuple of length 
                            2
        :raises ValueError: This is raised if the Target Text text is empty
        :raises ValueError: If the number of pos tags for this instance
                            does not have the same number of tokens that has 
                            been generated by the tokenizer function.
        '''
        text = self['text']
        tokens_pos_tags = tagger(text)

        if perform_type_checks:
            if not isinstance(tokens_pos_tags, tuple):
                raise TypeError('The return type for the pos tagger should be'
                                f' a tuple not {type(tokens_pos_tags)}')
            if len(tokens_pos_tags) != 2:
                raise ValueError('The return of the POS tagger should be a '
                                 f'tuple of length 2 not {len(tokens_pos_tags)}')
            if not isinstance(tokens_pos_tags[0], list):
                raise TypeError('The return type of the tagger function ',
                                f'{tagger} should be a list and not '
                                f'{type(tokens_pos_tags[0])} for the tokens')
            if not isinstance(tokens_pos_tags[1], list):
                raise TypeError('The return type of the tagger function ',
                                f'{tagger} should be a list and not '
                                f'{type(tokens_pos_tags[1])} for the POS tags')
            for name, tags in [('tokens', tokens_pos_tags[0]),
                               ('pos_tags', tokens_pos_tags[1])]:
                for tag in tags:
                    if not isinstance(tag, str):
                        raise TypeError('The return type of the tagger function ',
                                        f'{tagger} should be a list of Strings'
                                        f' and not a list of {type(tag)} for '
                                        f'the {name}')
        tokens, pos_tags = tokens_pos_tags
        num_pos_tags = len(pos_tags)
        if len(pos_tags) == 0:
            raise ValueError('There are no tags for this TargetText '
                             f'instance {self}')
        num_tokens = len(tokens)
        if num_tokens != num_pos_tags:
            raise ValueError(f'Number of POS tags {pos_tags} should be the '
                             f'same as the number of tokens {tokens}')

        self['pos_tags'] = pos_tags
        self['tokenized_text'] = tokens

    def sequence_labels(self) -> None:
        '''
        Adds the `sequence_labels` key to this TargetText instance which can 
        be used to train a machine learning algorthim to detect targets.

        The `force_targets` method might come in useful here for training 
        and validation data to ensure that more of the targets are not 
        affected by tokenization error as only tokens that are fully within 
        the target span are labelled with `B` or `I` tags.

        Currently the only sequence labels supported is IOB-2 labels for the 
        targets only. Future plans look into different sequence label order
        e.g. IOB see link below for more details of the difference between the 
        two sequence, of which there are more sequence again.
        https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)

        :raises KeyError: If the current TargetText has not been tokenized.
        :raises ValueError: If two targets overlap the same token(s) e.g 
                            `Laptop cover was great` if `Laptop` and 
                            `Laptop cover` are two seperate targets this should 
                            riase a ValueError as a token should only be 
                            associated to one target.
        '''
        text = self['text']
        if 'tokenized_text' not in self:
            raise KeyError(f'Expect the current TargetText {self} to have '
                           'been tokenized using the self.tokenize method.')
        self.sanitize()
        tokens = self['tokenized_text']
        sequence_labels = ['O' for _ in range(len(tokens))]
        # This is the case where there are no targets thus all sequence labels 
        # are `O`
        if self['spans'] is None or self['targets'] is None:
            self['sequence_labels'] = sequence_labels
            return
        
        target_spans: List[Span] = self['spans']
        tokens_index = token_index_alignment(text, tokens)

        for target_span in target_spans:
            target_span_range = list(range(*target_span))
            same_target = False
            for sequence_index, token_index in enumerate(tokens_index):
                token_start, token_end = token_index
                token_end = token_end - 1
                if (token_start in target_span_range and
                        token_end in target_span_range):
                    if sequence_labels[sequence_index] != 'O':
                        err_msg = ('Cannot have two sequence labels for one '
                                    f'token, text {text}\ntokens {tokens}\n'
                                    f'token indexs {tokens_index}\nTarget '
                                    f'spans {target_spans}')
                        raise ValueError(err_msg)
                    if same_target:
                        sequence_labels[sequence_index] = 'I'
                    else:
                        sequence_labels[sequence_index] = 'B'
                    same_target = True
        self['sequence_labels'] = sequence_labels

    def _key_error(self, key: str) -> None:
        '''
        :param key: The key to check for within this TargetText instance.
        :raises KeyError: If the key given does not exist within this 
                          TargetText instance.
        '''
        if f'{key}' not in self:
            raise KeyError(f'Requires that this TargetText instance {self}'
                           f'contians the key `{key}`')

    def get_sequence_indexs(self, sequence_key: str) -> List[List[int]]:
        '''
        The following sequence label tags are supported: IOB-2. These are the 
        tags that are currently generated by `sequence_labels`. 

        :param sequence_key: Key to sequence labels such as a BIO sequence 
                             labels. Example key name would be `sequence_labels`
                             after `sequence_labels` function has been called 
                             or more appropiately `predicted_sequence_labels` 
                             when you have predicted sequence labels.
        :returns: A list of a list of intergers where each list of integers 
                  represent the token/pos tag/sequence label index of each 
                  sequence label span.
                  :Example: These sequence labels [`O`, `B`, `I`, `O`, `B`] 
                            would return the following integers list [[1, 2], [4]]
        :raises ValueError: If the sequence labels that are contained in the 
                            sequence key value contain values other than 
                            `B`, `I`, or `O`.
        :raises ValueError: If then number of tokens in the current TargetText 
                            object is not the same as the number of sequence 
                            labels.
        '''
        # number of tokens and sequence labels, should 
        # all be the same, it is if the `sequence_labels` function is used
        tokens = self['tokenized_text']
        sequence_labels = self[sequence_key]
        if len(tokens) != len(sequence_labels):
            raise ValueError(f'The number of tokens in the TargetText object {self}'
                             f' is not the same as the number of sequence labels')

        same_target = False
        start_index = 0
        end_index = 0
        sequence_indexs: List[List[int]] = []
        for label_index, sequence_label in enumerate(sequence_labels):
            if sequence_label == 'B':
                if same_target == True:
                    sequence_index = list(range(start_index, end_index))
                    sequence_indexs.append(sequence_index)
                    same_target = False
                    start_index = 0
                    end_index = 0

                same_target = True
                start_index = label_index
                end_index = label_index + 1
            elif sequence_label == 'I':
                end_index = label_index + 1
            elif sequence_label == 'O':
                if same_target:
                    sequence_index = list(range(start_index, end_index))
                    sequence_indexs.append(sequence_index)
                    same_target = False
                    start_index = 0
                    end_index = 0
            else:
                raise ValueError('Sequence labels should be `B` `I` or `O` '
                                 f'and not {sequence_label}. Sequence label '
                                 f'key used {sequence_key}\nTargetText {self}')
        if end_index != 0:
            sequence_index = list(range(start_index, end_index))
            sequence_indexs.append(sequence_index)
        return sequence_indexs

    def get_sequence_spans(self, sequence_key: str,
                           confidence: Optional[float] = None) -> List[Span]:
        '''
        The following sequence label tags are supported: IOB-2. These are the 
        tags that are currently generated by `sequence_labels`

        :param sequence_key: Key to sequence labels such as a BIO sequence 
                             labels. Example key name would be `sequence_labels`
                             after `sequence_labels` function has been called 
                             or more appropiately `predicted_sequence_labels` 
                             when you have predicted sequence labels.
        :param confidence: Optional argument that will return only spans 
                           that have been predicted with a confidence 
                           higher than this. 
                           :NOTE: As it is BIO labelling in the case where 
                                  all but one of the B and I's is greater than 
                                  the threshold that span would not be 
                                  returned, as one of the words in the multi 
                                  word target word is less than the threshold.
        :returns: The span indexs from the sequence labels given assuming that 
                  the sequence labels are in BIO format.
        :raises KeyError: If no `confidence` key are found. However `confidence` 
                          is only required if the confidence argument is set.
        :raises ValueError: If the sequence labels that are contained in the 
                            sequence key value contain values other than 
                            `B`, `I`, or `O`.
        :raises ValueError: If the confidence value is not between 0 and 1
        '''
        # number of tokens, sequence labels, and token text indexs should 
        # all be the same, it is if the `sequence_labels` function is used
        if confidence is not None:
            self._key_error('confidence')
            if confidence > 1.0 or confidence < 0.0:
                raise ValueError('Confidence value has to be bounded between '
                                 f'1 and 0 and not {confidence}')

        sequence_indexs: List[List[int]] = self.get_sequence_indexs(sequence_key)
        if not sequence_indexs:
            return []
        tokens = self['tokenized_text']
        token_text_indexs = token_index_alignment(self['text'], tokens)
        sequence_spans: List[Span] = []

        confidences = None
        if confidence is not None:
            confidences = self['confidence']
        for span_sequence_index in sequence_indexs:
            # Test that each sequence label was predicted with enough confidence 
            if confidence is not None:
                next_span = False
                for index in span_sequence_index:
                    if confidences[index] <= confidence:
                        next_span = True
                if next_span:
                    continue
            start_index = span_sequence_index[0]
            start_span = token_text_indexs[start_index][0]
            
            end_index = span_sequence_index[-1]
            end_span = token_text_indexs[end_index][1]
            sequence_spans.append(Span(start_span, end_span))
        return sequence_spans

    def get_targets_from_sequence_labels(self, sequence_key: str, 
                                         confidence: Optional[float] = None
                                         ) -> List[str]:
        '''
        This function mains use is when the sequence labels have been 
        predicted on a piece of text that has no gold annotations.

        :param sequence_key: Key to sequence labels such as a BIO sequence 
                             labels. Example key name would be `sequence_labels`
                             after `sequence_labels` function has been called 
                             or more appropiately `predicted_sequence_labels` 
                             when you have predicted sequence labels.
        :param confidence: Optional argument that will return only target 
                           texts that have been predicted with a confidence 
                           higher than this. 
                           :NOTE: As it is BIO labelling in the case where 
                                  all but one of the B and I's is greater than 
                                  the threshold that target word would not be 
                                  returned as one of the words in the multi 
                                  word target word is less than the threshold.
        :returns: The target text's that the sequence labels have predcited.
        :raises KeyError: If no `tokenized_text` or `confidence` key are found.
                          However `confidence` is only required if the 
                          confidence argument is set.
        :raises ValueError: If the confidence value is not between 0 and 1
        '''
        if confidence is not None:
            self._key_error('confidence')
            if confidence > 1.0 or confidence < 0.0:
                raise ValueError('Confidence value has to be bounded between '
                                 f'1 and 0 and not {confidence}')
        self._key_error('tokenized_text')
        sequence_indexs: List[List[int]] = self.get_sequence_indexs(sequence_key)
        # No targets to extract
        if not sequence_indexs:
            return []
        tokens = self['tokenized_text']
        confidences = None
        if confidence is not None:
            confidences = self['confidence']
        targets = []
        for span_sequence_index in sequence_indexs:
            start_index = span_sequence_index[0]
            end_index = span_sequence_index[-1] + 1
            target_tokens = tokens[start_index: end_index]
            # Test that each token in target tokens was predicted with a 
            # great enough confidence
            if confidence is not None:
                next_span = False
                for index in span_sequence_index:
                    if confidences[index] <= confidence:
                        next_span = True
                if next_span:
                    continue
            print(start_index)
            print(end_index)
            target = ' '.join(target_tokens)
            targets.append(target)
        return targets

    def one_sample_per_span(self, remove_empty: bool = False) -> 'TargetText':
        '''
        This returns a similar TargetText instance where the new instance 
        will only contain one target per span. 
        
        This is for the cases where you can have a target e.g. `food` that has 
        a different related category attached to it e.g.
        TargetText(text=`$8 and there is much nicer, food, all of it great and 
                  continually refilled.`, text_id=`1`, 
                  targets=[`food`, `food`, `food`], 
                  categories=[`style`, `quality`, `price`], 
                  target_sentiments=[`pos`,`pos`,`pos`], 
                  spans=[Span(27, 31),Span(27, 31),Span(27, 31)])
        As we can see the targets and the categories are linked, this is only 
        really the case in SemEval 2016 datasets from what I know currently. 
        In the example case above it will transform it to the following:
        TargetText(text=`$8 and there is much nicer, food, all of it great and 
                   continually refilled.`, text_id=`1`, 
                   targets=[`food`],spans=[Span(27,31)])
        This type of pre-processing is perfect for the Target Extraction 
        task.

        :param remove_empty: If the TargetText instance contains any None 
                             targets then these will be removed along with 
                             their respective Spans.
        :returns: This returns a similar TargetText instance where the new 
                  instance will only contain one target per span.
        '''
        text = self['text']
        text_id = self['text_id']
        targets: List[str] = []
        spans: List[Span] = []

        if self['spans'] is None:
            return TargetText(text=text, text_id=text_id)

        current_spans = self['spans']
        unique_spans = set(current_spans)
        spans = sorted(unique_spans, key=lambda x: x[0])
        temp_spans: List[Span] = []
        for span in spans:
            targets_text = text[span.start: span.end]
            if span.start == 0 and span.end == 0 and remove_empty:
                continue
            else:
                temp_spans.append(span)
                targets.append(targets_text)
        spans = temp_spans
        return TargetText(text=text, text_id=text_id, 
                          targets=targets, spans=spans)

    def left_right_target_contexts(self, incl_target: bool
                                   ) -> List[Tuple[List[str], List[str], List[str]]]:
        '''
        :param incl_target: Whether or not the left and right sentences should 
                            also include the target word.
        :returns: The sentence that is left and right of the target as well as 
                  the words in the target for each target in the sentence.
        '''
        left_right_target_list = []
        text = self['text']
        if self['spans'] is not None:
            for span in self['spans']:
                span: Span
                span_start = span.start
                span_end = span.end
                if incl_target:
                    left_context = text[:span_end]
                    right_context = text[span_start:]
                else:
                    left_context = text[:span_start]
                    right_context = text[span_end:]
                target_context = text[span_start:span_end]
                contexts = (left_context, right_context, target_context)
                left_right_target_list.append(contexts)
        return left_right_target_list  



    @staticmethod
    def from_json(json_text: str) -> 'TargetText':
        '''
        This is required as the 'spans' are Span objects which are not json 
        serlizable and are required for TargetText therefore this handles 
        that special case.

        This function is also required as we have had to avoid using the 
        __set__ function and add objects via the _storage dictionary 
        underneath so that we could add values to this object that are not 
        within the constructor like `tokenized_text`. To ensure that it is 
        compatable with the TargetText concept we call `TargetText.sanitize`
        method at the end.

        :param json_text: JSON representation of TargetText 
                          (can be from TargetText.to_json)
        :returns: A TargetText object
        :raises KeyError: If within the JSON representation there is no 
                          `text` or `text_id` key.
        '''
        json_target_text = json.loads(json_text)
        if not 'text' in json_target_text or not 'text_id' in json_target_text:
            raise KeyError('The JSON text given does not contain a `text`'
                           f' or `text_id` field: {json_target_text}')
        target_text = TargetText(text=json_target_text['text'], 
                                 text_id=json_target_text['text_id'])
        for key, value in json_target_text.items():
            if key == 'text' or key == 'text_id':
                continue
            if key == 'spans':
                if value == None:
                    target_text._storage[key] = None
                else:
                    all_spans = []
                    for span in value:
                        all_spans.append(Span(*span))
                    target_text._storage[key] = all_spans
            else:
                target_text._storage[key] = value
        target_text.sanitize()
        return target_text

    @staticmethod
    def targets_from_spans(text:str, spans: List[Span]) -> List[str]:
        '''
        :param text: The text that the spans are associated too.
        :param spans: A list of Span values that represent the character index 
                      of the target words to be returned.
        :returns: The target words that are associated to the spans and text 
                  given.
        '''
        targets = []
        if not spans:
            return targets
        for span in spans:
            target = text[span.start: span.end]
            targets.append(target)
        return targets

    @staticmethod
    def target_text_from_prediction(text: str, text_id: str, 
                                    sequence_labels: List[str], 
                                    tokenized_text: List[str],
                                    confidence: Optional[float] = None,
                                    confidences: Optional[List[float]] = None,
                                    **additional_data) -> 'TargetText':
        '''
        Creates a TargetText object from data that has come from predictions
        of a Target Extract tagger e.g. the dictionaries that are returned 
        from :meth:`target_extraction.allen.allennlp_model.predict_sequences`

        :param text: Text to give to the TargetText object
        :param text_id: Text ID to give to the TargetText object
        :param sequence_labels: The predicted sequence labels
        :param tokenized_text: The tokens that were used to produce the 
                               predicted sequence labels (should be returned 
                               by the Target Extract tagger predictor).
        :param confidence: The level of confidence from the tagger that is 
                           required for a target to be a target e.g. 0.9
        :param confidences: The list of confidence values produced 
                            by the Target Extract tagger predictor to be used 
                            with the confidence argument. The list of confidence 
                            values should be the same size as the sequence labels 
                            list and tokenized text.
        :param additional_data: Any other keyword arguments to provide to the 
                                TargetText object
        :returns: A TargetText object with spans and targets values
        :raises ValueError: If sequence labels, tokenized text and confidecnes 
                            are not of the same length
        :raises ValueError: If the following keys are in the additional data;
                            1. confidence, 2. text, 3. text_id, 4. tokenized_text
                            5. sequence_labels, 6. targets, 7. spans. As these 
                            keys will be populated by within the TargetText 
                            object automatically.
        '''
        if len(sequence_labels) != len(tokenized_text):
            raise ValueError('Sequence labels and tokenized texts are not of '
                             f'the same length:\nSequence labels {sequence_labels}'
                             f'\nTokenized text: {tokenized_text}')
        if confidence is not None and len(sequence_labels) != len(confidences):
            raise ValueError('Sequence labels and confidences are not of '
                             f'the same length:\nSequence labels {sequence_labels}'
                             f'\nconfidences: {confidences}')
        not_allowed_additional_keys = {'confidence', 'text', 'text_id', 
                                       'tokenized_text', 'sequence_labels',
                                       'targets', 'spans'}
        for key in additional_data:
            if key in not_allowed_additional_keys:
                raise ValueError("The following keys are not allowd in the "
                                 f"additional data:\n{not_allowed_additional_keys}")
        temp_target_text = TargetText(text_id=text_id, text=text, 
                                      tokenized_text=tokenized_text,
                                      sequence_labels=sequence_labels,
                                      confidence=confidences)
        target_spans = temp_target_text.get_sequence_spans('sequence_labels',
                                                           confidence=confidence)
        targets = TargetText.targets_from_spans(text, target_spans)
        return TargetText(text_id=text_id, text=text, confidence=confidences,
                          tokenized_text=tokenized_text, targets=targets,
                          spans=target_spans, sequence_labels=sequence_labels,
                          **additional_data)



class TargetTextCollection(MutableMapping):
    '''
    This is a data structure that inherits from MutableMapping which is 
    essentially a python dictionary, however the underlying storage is a 
    OrderedDict therefore if you iterate over it, the iteration will always be 
    in the same order.

    This structure only contains TargetText instances.

    Methods:
    
    1. to_json -- Writes each TargetText instances as a dictionary using it's 
       own to_json function on a new line within the returned String. The 
       returned String is not json comptable but if split by new line it is and 
       is also comptable with the from_json method of TargetText.
    2. add -- Wrapper around __setitem__. Given as an argument a TargetText 
       instance it will be added to the collection.
    3. to_json_file -- Saves the current TargetTextCollection to a json file 
       which won't be strictly json but each line in the file will be and each 
       line in the file can be loaded in from String via TargetText.from_json. 
       Also the file can be reloaded into a TargetTextCollection using 
       TargetTextCollection.load_json.
    4. tokenize -- This applies the TargetText.tokenize method across all 
       of the TargetText instances within the collection.
    5. pos_text -- This applies the TargetText.pos_text method across all of 
        the TargetText instances within the collection.
    6. sequence_labels -- This applies the TargetText.sequence_labels 
       method across all of the TargetText instances within the collection.
    7. force_targets -- This applies the TargetText.force_targets method 
       across all of the TargetText instances within the collection.
    8. exact_match_score -- Recall, Precision, and F1 score in a Tuple. 
       All of these measures are based on exact span matching rather than the 
       matching of the sequence label tags, this is due to the annotation spans 
       not always matching tokenization therefore this removes the tokenization 
       error that can come from the sequence label measures.
    9. samples_with_targets -- Returns all of the samples that have target 
                               spans as a TargetTextCollection. 
    10. target_count -- A dictionary of target text as key and values as the  
        number of times the target text occurs in this TargetTextCollection
    11. one_sample_per_span -- This applies the TargetText.one_sample_per_span 
        method across all of the TargetText instances within the collection to 
        create a new collection with those new TargetText instances within it.
    12. sanitize -- This applies the TargetText.sanitize function to all of 
        the TargetText instances within this collection, affectively ensures 
        that all of the instances follow the specified rules that TargetText 
        instances should follow.
    13. number_targets -- Returns the total number of targets.
    14. target_sentiments -- A dictionary where the keys are target texts and 
        the values are a List of sentiment values that have been associated to 
        that target.
    
    Static Functions:

    1. from_json -- Returns a TargetTextCollection object given the json like 
       String from to_json. For example the json string can be the return of 
       TargetTextCollection.to_json.
    2. load_json -- Returns a TargetTextCollection based on each new line in 
       the given json file.
    3. combine -- Returns a TargetTextCollection that is the combination of all 
       of those given.
    '''
    def __init__(self, target_texts: Optional[List['TargetText']] = None,
                 name: Optional[str] = None) -> None:
        '''
        :param target_texts: A list of TargetText instances to add to the 
                             collection.
        :param name: Name to call the collection.
        '''
        self._storage = OrderedDict()

        if target_texts is not None:
            for target_text in target_texts:
                self.add(target_text)
        
        if name is None:
            name = ''
        self.name = name

    def add(self, value: 'TargetText') -> None:
        '''
        Wrapper around set item. Instead of having to add the value the 
        usual way of finding the instances 'text_id' and setting this containers
        key to this value, it does this for you.

        e.g. performs self[value['text_id']] = value

        :param value: The TargetText instance to store in the collection
        '''

        self[value['text_id']] = value

    def to_json(self) -> str:
        '''
        Required as TargetTextCollection is not json serlizable due to the 
        'spans' in the TargetText instances.

        :returns: The object as a list of dictionarys where each the TargetText
                  instances are dictionaries.
        '''
        json_text = ''
        for index, target_text_instance in enumerate(self.values()):
            if index != 0:
                json_text += '\n'
            target_text_instance: TargetText
            json_text += target_text_instance.to_json()
        return json_text

    @staticmethod
    def from_json(json_text: str, **target_text_collection_kwargs
                  ) -> 'TargetTextCollection':
        '''
        Required as the json text is expected to be the return from the 
        self.to_json method. This string is not passable by a standard json 
        decoder.

        :param json_text: This is expected to be a dictionary like object for 
                          each new line in this text
        :param target_text_collection_kwargs: Key word arguments to give to 
                                              the TargetTextCollection 
                                              constructor.
        :returns: A TargetTextCollection based on each new line in the given 
                  text to be passable by TargetText.from_json method.
        '''
        if json_text.strip() == '':
            return TargetTextCollection(**target_text_collection_kwargs)

        target_text_instances = []
        for line in json_text.split('\n'):
            target_text_instances.append(TargetText.from_json(line))
        return TargetTextCollection(target_text_instances, 
                                    **target_text_collection_kwargs)

    @staticmethod
    def load_json(json_fp: Path, **target_text_collection_kwargs
                  ) -> 'TargetTextCollection':
        '''
        Allows loading a dataset from json. Where the json file is expected to 
        be output from TargetTextCollection.to_json_file as the file will be 
        a json String on each line generated from TargetText.to_json.

        :param json_fp: File that contains json strings generated from 
                        TargetTextCollection.to_json_file
        :param target_text_collection_kwargs: Key word arguments to give to 
                                              the TargetTextCollection 
                                              constructor.
        :returns: A TargetTextCollection based on each new line in the given 
                  json file.
        '''
        target_text_instances = []
        with json_fp.open('r') as json_file:
            for line in json_file:
                if line.strip():
                    target_text_instance = TargetText.from_json(line)
                    target_text_instances.append(target_text_instance)
        return TargetTextCollection(target_text_instances, 
                                    **target_text_collection_kwargs)

    def to_json_file(self, json_fp: Path) -> None:
        '''
        Saves the current TargetTextCollection to a json file which won't be 
        strictly json but each line in the file will be and each line in the 
        file can be loaded in from String via TargetText.from_json. Also the 
        file can be reloaded into a TargetTextCollection using 
        TargetTextCollection.load_json.

        :param json_fp: File path to the json file to save the current data to.
        '''
        with json_fp.open('w+') as json_file:
            for index, target_text_instance in enumerate(self.values()):
                target_text_instance: TargetText
                target_text_string = target_text_instance.to_json()
                if index != 0:
                    target_text_string = f'\n{target_text_string}'
                json_file.write(target_text_string)

    def tokenize(self, tokenizer: Callable[[str], List[str]]) -> None:
        '''
        This applies the TargetText.tokenize method across all of 
        the TargetText instances within the collection.

        For a set of tokenizers that are definetly comptable see 
        target_extraction.tokenizers module.

        Ensures that the tokenization is character preserving.

        :param tokenizer: The tokenizer to use tokenize the text for each 
                          TargetText instance in the current collection
        :raises TypeError: If the tokenizer given does not return a List of 
                           Strings.
        :raises ValueError: This is raised if any of the TargetText instances 
                            in the collection contain an empty string.
        :raises ValueError: If the tokenization is not character preserving.
        '''

        for index, target_text_instance in enumerate(self.values()):
            if index == 0:
                target_text_instance.tokenize(tokenizer, True)
            else:
                target_text_instance.tokenize(tokenizer, False)
    
    def pos_text(self, tagger: Callable[[str], List[str]]) -> None:
        '''
        This applies the TargetText.pos_text method across all of 
        the TargetText instances within the collection.

        For a set of pos taggers that are definetly comptable see 
        target_extraction.pos_taggers module.

        :param tagger: POS tagger.
        :raises TypeError: If the POS tagger given does not return a List of 
                           Strings.
        :raises ValueError: This is raised if any of the TargetText instances 
                            in the collection contain an empty string.
        :raises ValueError: If the Target Text instance has not been tokenized.
        :raises ValueError: If the number of pos tags for a Target Text instance
                            does not have the same number of tokens that has 
                            been generated by the tokenizer function.
        '''

        for index, target_text_instance in enumerate(self.values()):
            if index == 0:
                target_text_instance.pos_text(tagger, True)
            else:
                target_text_instance.pos_text(tagger, False)

    def force_targets(self) -> None:
        '''
        This applies the TargetText.force_targets method across all of the 
        TargetText instances within the collection.
        '''
        for target_text_instance in self.values():
            target_text_instance.force_targets()

    def sequence_labels(self, return_errors: bool = False
                        ) -> List['TargetText']:
        '''
        This applies the TargetText.sequence_labels method across all of 
        the TargetText instances within the collection.

        :param return_errors: Returns TargetText objects that have caused 
                              the ValueError to be raised.
        :returns: A list of TargetText objects that have caused the ValueError 
                  to be raised if `return_errors` is True else an empty list 
                  will be returned. 
        :raises KeyError: If the current TargetText has not been tokenized.
        :raises ValueError: If two targets overlap the same token(s) e.g 
                            `Laptop cover was great` if `Laptop` and 
                            `Laptop cover` are two seperate targets this should 
                            riase a ValueError as a token should only be 
                            associated to one target.
        '''

        errored_targets = []
        for target_text_instance in self.values():
            if return_errors:
                try:
                    target_text_instance.sequence_labels()
                except ValueError:
                    errored_targets.append(target_text_instance)
            else:
                target_text_instance.sequence_labels()
        return errored_targets


    def exact_match_score(self, 
                          predicted_sequence_key: str = 'predicted_sequence_labels'
                          ) -> Tuple[float, float, float, 
                                     Dict[str, List[Tuple[str, Span]]]]:
        '''        
        Just for clarification we use the sequence label tags to find the 
        predicted spans. However even if you have a perfect sequence label 
        score does not mean you will have a perfect extact span score 
        as the tokenizer used for the sequence labelling might not align 
        perfectly with the annotated spans.

        The False Positive mistakes, False Negative mistakes, and correct
        True Positive Dictionary keys are those names with the values neing a 
        List of Tuples where the Tuple is made up of the TargetText instance ID 
        and the Span that was incorrect (FP) or not tagged (FN) or correct (TP).
        Example of this is as follows:
        {`FP`: [('1', Span(0, 4))], 'FN': [], 'TP': []}

        :param predicted_sequence_key: Key of the predicted sequence labels 
                                       within this TargetText instance.
        :returns: Recall, Precision, and F1 score, False Positive mistakes, 
                  False Negative mistakes, and correct True Positives in a 
                  Dict. All of these measures are based on exact span matching 
                  rather than the matching of the sequence label tags, 
                  this is due to the annotation spans not always matching 
                  tokenization therefore this removes the tokenization 
                  error that can come from the sequence label measures.
        :raises KeyError: If there are no predicted sequence label key 
                          within this TargetText.
        :raises ValueError: If the predicted or true spans contain multiple 
                            spans that have the same span e.g. 
                            [Span(4, 15), Span(4, 15)]
        '''
        # tp = True Positive count
        tp = 0.0
        num_pred_true = 0.0
        num_actually_true = 0.0
        fp_mistakes: List[Tuple[str, Span]] = []
        fn_mistakes: List[Tuple[str, Span]] = []
        correct_tp: List[Tuple[str, Span]] = []

        for target_text_index, target_text_instance in enumerate(self.values()):
            if target_text_index == 0:
                keys_to_check = ['spans', 
                                f'{predicted_sequence_key}']
                for key in keys_to_check:
                    target_text_instance._key_error(key)
            predicted_spans = target_text_instance.get_sequence_spans(predicted_sequence_key)
            # Add to the number of predicted true and actually true
            predicted_spans: List[Span]
            num_pred_true += len(predicted_spans)

            true_spans: List[Span] = target_text_instance['spans']
            if true_spans is None:
                true_spans = []
            num_actually_true += len(true_spans)
            
            # This should be impossible to get to
            if len(predicted_spans) != len(set(predicted_spans)):
                raise ValueError(f'Predicted spans {predicted_spans} contain'
                                 f' multiple of the same predicted span. '
                                 f'TargetText: {target_text_instance}')
            # This is possible
            if len(true_spans) != len(set(true_spans)):
                raise ValueError(f'True spans {true_spans} contain'
                                 f' multiple of the same true span. '
                                 f'TargetText: {target_text_instance}')
            
            text_id = target_text_instance['text_id']
            true_spans = set(true_spans)
            for predicted_span in predicted_spans:
                if predicted_span in true_spans:
                    tp += 1
                    correct_tp.append((text_id, predicted_span))
                else:
                    fp_mistakes.append((text_id, predicted_span))
            for true_span in true_spans:
                if true_span not in predicted_spans:
                    fn_mistakes.append((text_id, true_span))
        
        error_analysis_dict = {'FP': fp_mistakes, 'FN': fn_mistakes, 
                               'TP': correct_tp}
        if tp == 0.0:
            return 0.0, 0.0, 0.0, error_analysis_dict
        recall = tp / num_actually_true
        precision = tp / num_pred_true
        f1 = (2 * precision * recall) / (precision + recall)
        return recall, precision, f1, error_analysis_dict 

    def samples_with_targets(self) -> 'TargetTextCollection':
        '''
        :returns: All of the samples that have targets as a 
                  TargetTextCollection for this TargetTextCollection.
        :raises KeyError: If either `spans` or `targets` does not exist in 
                          one or more of the TargetText instances within this 
                          collection. These key's are protected keys thus they
                          should always exist but this is just a warning if 
                          you have got around the protected keys.
        '''
        sub_collection = TargetTextCollection()
        for target_text in self.values():
            if target_text['spans'] and target_text['targets']:
                sub_collection.add(target_text)
        return sub_collection

    def target_count(self, lower: bool = False) -> Dict[str, int]:
        '''
        :Note: The target can not exist e.g. be a `None` target as the target 
               can be combined with the category like in the SemEval 2016 
               Restaurant dataset. In these case we do not include these 
               in the target_count.
        :param lower: Whether or not to lower the target text.
        :returns: A dictionary of target text as key and values as the number 
                  of times the target text occurs in this TargetTextCollection
        '''
        target_count: Dict[str, int] = Counter()
        for target_dict in self.values():
            if target_dict['targets']:
                for target in target_dict['targets']:
                    if target is None:
                        continue
                    if lower:
                        target = target.lower()
                    target_count.update([target])
        return dict(target_count)

    def target_sentiments(self, lower: bool = False, 
                          unique_sentiment: bool = False
                          ) -> Dict[str, Union[List[str], Set[str]]]:
        '''
        :Note: The target can not exist e.g. be a `None` target as the target 
               can be combined with the category like in the SemEval 2016 
               Restaurant dataset. In these case we do not include these 
               in the target_count.
        :param lower: Whether or not to lower the target text.
        :param unique_sentiment: Whether or not the return is a dictionary  
                                 whose values are a List of Strings or if 
                                 True a Set of Strings.
        :returns: A dictionary where the keys are target texts and the values 
                  are a List of sentiment values that have been associated to 
                  that target. The sentiment value can occur more than once 
                  indicating the number of times that target has been associated 
                  with that sentiment unless unique_sentiment is True then 
                  instead of a List of sentiment values a Set is used instead.
        :Explanation: If the target `camera` has occured with the sentiment 
                      `positive` twice and `negative` once then it will return 
                      {`camera`: [`positive`, `positive`, `negative`]}. However
                      if `unique_sentiment` is True then it will return:
                      {`camera`: {`positive`, `negative`}}.

        '''
        target_sentiment_values: Dict[str, List[str]] = defaultdict(list)
        if unique_sentiment:
            target_sentiment_values: Dict[str, Set[str]] = defaultdict(set)
        for target_dict in self.values():
            if target_dict['targets'] and target_dict['target_sentiments']:
                for target, sentiment in zip(target_dict['targets'], 
                                             target_dict['target_sentiments']):
                    if target is None:
                        continue
                    if lower:
                        target = target.lower()
                    if unique_sentiment:
                        target_sentiment_values[target].add(sentiment)
                    else:
                        target_sentiment_values[target].append(sentiment)
        return dict(target_sentiment_values)

    def number_targets(self, incl_none_targets: bool = False) -> int:
        '''
        :param incl_none_targets: Whether to include targets that are `None`
                                  and are therefore associated to the categories 
                                  in the count.
        :returns: The total number of targets in the collection. 
        '''
        target_count = 0
        for target_dict in self.values():
            if target_dict['targets']:
                for target in target_dict['targets']:
                    if not incl_none_targets and target is None:
                        continue
                    target_count += 1
        return target_count

    def one_sample_per_span(self, remove_empty: bool = False
                            ) -> 'TargetTextCollection':
        '''
        This applies the TargetText.one_sample_per_span method across all of the 
        TargetText instances within the collection to create a new collection 
        with those new TargetText instances within it.
        
        :param remove_empty: If the TargetText instance contains any None 
                             targets then these will be removed along with 
                             their respective Spans.
        :returns: A new TargetTextCollection that has samples that come 
                  from this collection but has had the 
                  TargetText.one_sample_per_span method applied to it.
        '''
        
        new_collection = TargetTextCollection()
        for target_text in self.values():
            new_collection.add(target_text.one_sample_per_span(remove_empty=remove_empty))
        return new_collection

    def sanitize(self) -> None:
        '''
        This applies the TargetText.sanitize function to all of 
        the TargetText instances within this collection, affectively ensures 
        that all of the instances follow the specified rules that TargetText 
        instances should follow.
        '''

        for target_text in self.values():
            target_text.sanitize()

    @staticmethod
    def combine(*collections) -> 'TargetTextCollection':
        '''
        :param collections: An iterator containing one or more 
                            TargetTextCollections
        :returns: A TargetTextCollection that is the combination of all of 
                  those given.
        '''
        target_objects: 'TargetText' = []
        for collection in collections:
            for target in collection.values():
                target_objects.append(target)
        return TargetTextCollection(target_objects)


    def __setitem__(self, key: str, value: 'TargetText') -> None:
        '''
        Will add the TargetText instance to the collection where the key 
        should be the same as the TargetText instance 'text_id'.

        :param key: Key to be added or changed
        :param value: TargetText instance associated to this key. Where the 
                      key should be the same value as the TargetText instance 
                      'text_id' value.
        '''
        if not isinstance(value, TargetText):
            raise TypeError('The value should be of type TargetText and not '
                            f'{type(value)}')
        text_id = value['text_id']
        if text_id != key:
            raise ValueError(f'The value `text_id`: {text_id} should be the '
                             f'same value as the key: {key}')
        # We copy it to stop any mutable objects from changing outside of the 
        # collection
        value_copy = copy.deepcopy(value)
        self._storage[key] = value_copy

    def __delitem__(self, key: str) -> None:
        '''
        Given a key that matches a key within self._storage or self.keys() 
        it will delete that key and value from this object.

        :param key: Key and its respective value to delete from this object.
        '''
        del self._storage[key]

    def __eq__(self, other: 'TargetTextCollection') -> bool:
        '''
        Two TargetTextCollection instances are equal if they both have 
        the same TargetText instances within it.

        :param other: Another TargetTextCollection object that is being  
                      compared to this TargetTextCollection object.
        :returns: True if they have the same TargetText instances within it.
        '''

        if not isinstance(other, TargetTextCollection):
            return False

        if len(self) != len(other):
            return False

        for key in self.keys():
            if key not in other:
                return False
        return True

    def __repr__(self) -> str:
        '''
        :returns: String returned is what user see when the instance is 
                  printed or printed within a interpreter.
        '''
        rep_text = 'TargetTextCollection('
        for key, value in self.items():
            rep_text += f'key: {key}, value: {value}'
            break
        if len(self) > 1:
            rep_text += '...)'
        else:
            rep_text += ')'
        return rep_text

    def __len__(self) -> int:
        '''
        :returns: The number of TargetText instances in the collection.
        '''
        return len(self._storage)

    def __iter__(self) -> Iterable[str]:
        '''
        Returns as interator over the TargetText instances 'text_id''s that 
        are stored in this collection. This is an ordered iterator as the 
        underlying dictionary used to store the TargetText instances is an 
        OrderedDict in self._storage.

        :returns: TargetText instances 'text_id''s that are stored in this 
                  collection
        '''
        return iter(self._storage)

    def __getitem__(self, key: str) -> 'TargetText':
        '''
        :returns: A TargetText instance that is stored within this collection.
        '''
        return self._storage[key]
