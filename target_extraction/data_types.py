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
from collections import OrderedDict, Counter, defaultdict, deque
import copy
import json
import functools
from pathlib import Path
from typing import Optional, List, Tuple, Iterable, NamedTuple, Any, Callable
from typing import Union, Dict, Set
import traceback
import random

from target_extraction.tokenizers import is_character_preserving, token_index_alignment
from target_extraction.data_types_util import (Span, OverLappingTargetsError,
                                               AnonymisedError, OverwriteError)

def check_anonymised(func):
    '''
    Assumes the first argument in the given function is a TargetText object 
    defined by self.

    :raises AnonymisedError: If the TargetText object given to `func` 
                             `anonymised` attribute is True.
    '''
    @functools.wraps(func)
    def wrapper_func(*args, **kwargs):
        target_text_object = args[0]
        if target_text_object.anonymised:
            anonymised_err = (f'Cannot perform this function as the Target '
                              f'{target_text_object} has been anonymised '
                              'and therefore has no `text`')
            raise AnonymisedError(anonymised_err)
        return func(*args, **kwargs)
    return wrapper_func


class TargetText(MutableMapping):
    '''
    This is a data structure that inherits from MutableMapping which is 
    essentially a python dictionary.

    The following are the default keys that are in all `TargetText` 
    objects, additional items can be added through __setitem__

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

    Attributes:

    1. anonymised -- If True then the data within the TargetText object has 
       no text but the rest of the metadata should exist.

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
    11. replace_target -- Given an index and a new target word it will replace 
        the target at the index with the new target word and return a new 
        TargetText object with everything the same apart from this new target.
    12. de_anonymise -- This will set the `anonymised` attribute to False 
        from True and set the `text` key value to the value in the `text` 
        key within the `text_dict` argument. 
    13. in_order -- True if all the `targets` within this TargetText 
        are in sequential left to right order within the text.
    14. re_order -- Re-Orders the TargetText object targets so that they are in 
        a left to right order within the text, this will then re-order all 
        values within this object that are in a list format into this order. 
        Once the TargetText has been re-ordered it will return True when 
        :py:meth`target_extraction.data_types.TargetText.in_order` is called.
    15. add_unique_key -- Given a key e.g. `targets` it will create a new value 
        in the TargetText object that is a list of strings which are unique IDs
        based on the `text_id` and the index the `targets` occur in e.g. 
        if the `targets` contain [`food`, `service`] and the `text_id` is 
        `12a5` then the `target_id` created will contain `[`12a5$$0`,`12a5$$1`]`  
    
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
        3. If anonymised esures that the `text` key does not exist.

        The 2nd check is not performed if `self.anonymised` is False.
    
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
            for target, span in zip(targets, spans):
                if target is None:
                    target_span_msg = 'As the target value is None the span '\
                                      'it refers to should be of value '\
                                      f'Span(0, 0) and not {span}'
                    if span != Span(0, 0):
                        raise ValueError(text_id_msg + target_span_msg)
                else:
                    if span == Span(0, 0) and target != '':
                        target_span_msg = (f'The Span is {Span(0, 0)} and the '
                                           f'target is {target} therefore the '
                                           'span must be in-correct for this'
                                           f' target {self}.')
                        raise ValueError(target_span_msg)
                    # Cannot check the text value when the data has been anonymised
                    if self.anonymised:
                        continue
                    text = self._storage['text'] 
                    start, end = span.start, span.end
                    text_target = text[start:end]
                    target_span_msg = 'The target the spans reference in the '\
                                      f'text: {text_target} does not match '\
                                      f'the target in the targets list: {target}'
                    if text_target != target:
                        raise ValueError(text_id_msg + target_span_msg)
        if self.anonymised and 'text' in self._storage:
            raise ValueError('The TargetText object is anonymised and therefore'
                             f' should not contain a `text` key. {self}')

    def __init__(self, text: Union[str, None], text_id: str,
                 targets: Optional[List[str]] = None, 
                 spans: Optional[List[Span]] = None, 
                 target_sentiments: Optional[List[Union[int, str]]] = None, 
                 categories: Optional[List[str]] = None,
                 category_sentiments: Optional[List[Union[int, str]]] = None,
                 anonymised: bool = False,
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
        # anonymised data will have no text
        temp_dict = dict(text=text, text_id=text_id, targets=targets,
                         spans=spans, target_sentiments=target_sentiments, 
                         categories=categories, 
                         category_sentiments=category_sentiments)
        if anonymised:
            del temp_dict['text']
            self._protected_keys = set(['text_id', 'targets', 'spans'])
        else:
            self._protected_keys = set(['text', 'text_id', 'targets', 'spans'])
        self._storage = temp_dict
        self._storage = {**self._storage, **additional_data}
        self._anonymised = anonymised
        self.sanitize()

    @property
    def anonymised(self) -> bool:
        '''
        :returns: True if the data within the TargetText has been anonymised.
                  Anonymised data means that there is no text associated with
                  the TargetText object but all of the metadata is there.
        '''
        return self._anonymised

    @anonymised.setter
    def anonymised(self, value: bool) -> None:
        '''
        Sets whether or not `anonymised` attribute is True or False. Either 
        which way when set it performs the `sanitize` check to ensure that 
        the attribute can be set to this value else it is reverted.

        :param value: If True then the `text` key will be deleted. In all 
                      cases the TargetText object is subjected to the 
                      :py:meth:`sanitize` to ensure that the anonymised 
                      process is correct.
        :raises AnonymisedError: If the TargetText object cannot be set to the 
                                 `anonymised` value given. If this Error occurs 
                                 then the object will have kept the original 
                                 `anonymised` value.
        '''
        # If want to anonymise all the 
        if not self.anonymised and value:
            del self._storage['text']
        
        self._anonymised = value
        try:
            self.sanitize()
        except:
            self._anonymised = not value
            sanitize_err = traceback.format_exc()
            raise AnonymisedError('Cannot de-anonymise this TargetText '
                                    f'{self} as it cannot pass the `sanitize`'
                                    ' check of which the following is the '
                                    f'error from said check {sanitize_err}')

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

    def _shift_spans(self, target_span: Span, prefix: bool, 
                     suffix: bool) -> None:
        '''
        This only affects the current state of the TargetText attributes. 
        The attributes this affects is the `spans` attribute.

        NOTE: This is only used within self.force_targets method.

        :param prefix: Whether it affects the prefix of the target_span
        :param suffix: Whether it affects the suffix of the target_span
        :param spans: The current target span indexs that are having extra 
                        whitespace added either prefix or suffix.
        '''
        target_span_start = target_span.start
        target_span_end = target_span.end
        for span_index, other_target_span in enumerate(self['spans']):
            if other_target_span == target_span:
                continue
            start, end = self['spans'][span_index]
            if prefix:
                if other_target_span.start >= target_span_start:
                    start += 1
                if other_target_span.end >= target_span_start:
                    end += 1
            if suffix:
                if other_target_span.start >= target_span_end:
                    start += 1
                if other_target_span.end >= target_span_end:
                    end += 1
            self._storage['spans'][span_index] = Span(start, end)

    @check_anonymised
    def force_targets(self) -> None:
        '''
        :NOTE: As this affects the following attributes `spans` and `text` it 
               therefore has to modify these through self._storage as both of  
               these attributes are within self._protected_keys.

        Does not return anything but modifies the `spans` and `text` values 
        as whitespace is prefixed and suffixed the target unless the prefix 
        or suffix is whitespace.

        Motivation:
        Ensure that the target tokens are not within another separate String 
        e.g. target = `priced` but the sentence is `the laptop;priced is high` 
        and the tokenizer is on whitespace it will not have `priced` seperated 
        therefore the BIO tagging is not deterministic thus force will add 
        whitespace around the target word e.g. `the laptop; priced`. This was 
        mainly added for the TargetText.sequence_tags method.

        :raises AnonymisedError: If the object has been anonymised then this 
                                 method cannot be used.
        '''
        for span_index in range(len(self['spans'])):
            text = self._storage['text']
            last_token_index = len(text)

            span = self._storage['spans'][span_index]
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
                self._shift_spans(span, prefix=True, suffix=True)
                self._storage['spans'][span_index] = Span(start + 1, end + 1)
            elif prefix:
                self._storage['text'] = f'{text_before} {target}{text_after}'
                self._shift_spans(span, prefix=True, suffix=False)
                self._storage['spans'][span_index] = Span(start + 1, end + 1)
            elif suffix:
                self._storage['text'] = f'{text_before}{target} {text_after}'
                self._shift_spans(span, prefix=False, suffix=True)
        # Get the targets from the re-aligned spans
        updated_targets = []
        text = self._storage['text']
        for span in self._storage['spans']:
            target = text[span.start: span.end]
            updated_targets.append(target)
        self._storage['targets'] = updated_targets

    @check_anonymised
    def tokenize(self, tokenizer: Callable[[str], List[str]],
                 perform_type_checks: bool = False) -> None:
        '''
        This will add a new key `tokenized_text` to this TargetText instance
        that will store the tokens of the text that is associated to this 
        TargetText instance.

        For a set of tokenizers that are definitely comptable see 
        target_extraction.tokenizers module.

        Ensures that the tokenization is character preserving.

        :param tokenizer: The tokenizer to use tokenize the text for each 
                          TargetText instance in the current collection
        :param perform_type_checks: Whether or not to perform type checks 
                                    to ensure the tokenizer returns a List of 
                                    Strings
        :raises AnonymisedError: If the object has been anonymised then this 
                                 method cannot be used.
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

    @check_anonymised
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
        :raises AnonymisedError: If the object has been anonymised then this 
                                 method cannot be used.
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

    @check_anonymised
    def sequence_labels(self, per_target: bool = False) -> None:
        '''
        Adds the `sequence_labels` key to this TargetText instance which can 
        be used to train a machine learning algorthim to detect targets. The 
        value associated to the `sequence_labels` key will be a list of 
        `B`, `I`, or `O` labels, where each label is associated to a token.

        The `force_targets` method might come in useful here for training 
        and validation data to ensure that more of the targets are not 
        affected by tokenization error as only tokens that are fully within 
        the target span are labelled with `B` or `I` tags.

        Currently the only sequence labels supported is IOB-2 labels for the 
        targets only. Future plans look into different sequence label order
        e.g. IOB see link below for more details of the difference between the 
        two sequence, of which there are more sequence again.
        https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)

        :param per_target: Whether the the value of associated to the 
                           `sequence_labels` key should be one list for all 
                           of the targets False. Or if True should be a list 
                           of a labels per target where the labels will only 
                           be associated to the represented target.
        :raises AnonymisedError: If the object has been anonymised then this 
                                 method cannot be used.
        :raises KeyError: If the current TargetText has not been tokenized.
        :raises ValueError: If two targets overlap the same token(s) e.g 
                            `Laptop cover was great` if `Laptop` and 
                            `Laptop cover` are two separate targets this should 
                            raise a ValueError as a token should only be 
                            associated to one target.
        '''
        text = self['text']
        if 'tokenized_text' not in self:
            raise KeyError(f'Expect the current TargetText {self} to have '
                           'been tokenized using the self.tokenize method.')
        self.sanitize()
        tokens = self['tokenized_text']
        sequence_labels = ['O' for _ in range(len(tokens))]
        if per_target:
            sequence_labels = [sequence_labels]
        # This is the case where there are no targets thus all sequence labels 
        # are `O`
        if self['spans'] is None or self['targets'] is None:
            self['sequence_labels'] = sequence_labels
            return

        if per_target:
            sequence_labels = []
            for _ in self['targets']:
                sequence_labels.append(['O' for _ in range(len(tokens))])
        
        target_spans: List[Span] = self['spans']
        tokens_index = token_index_alignment(text, tokens)

        for target_index, target_span in enumerate(target_spans):
            target_span_range = list(range(*target_span))
            same_target = False
            current_sequence_labels = sequence_labels
            if per_target:
                current_sequence_labels = sequence_labels[target_index]
            for sequence_index, token_index in enumerate(tokens_index):
                token_start, token_end = token_index
                token_end = token_end - 1
                if (token_start in target_span_range and
                        token_end in target_span_range):
                    if current_sequence_labels[sequence_index] != 'O':
                        err_msg = ('Cannot have two sequence labels for one '
                                    f'token, text {text}\ntokens {tokens}\n'
                                    f'token indexs {tokens_index}\nTarget '
                                    f'spans {target_spans}')
                        raise ValueError(err_msg)
                    if same_target:
                        current_sequence_labels[sequence_index] = 'I'
                    else:
                        current_sequence_labels[sequence_index] = 'B'
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

    @check_anonymised
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
        :raises AnonymisedError: If the object has been anonymised then this 
                                 method cannot be used.
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

    @check_anonymised
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
        :raises AnonymisedError: If the object has been anonymised then this 
                                 method cannot be used.
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

    @check_anonymised
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
        :raises AnonymisedError: If the object has been anonymised then this 
                                 method cannot be used.
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
            target = ' '.join(target_tokens)
            targets.append(target)
        return targets

    @check_anonymised
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
        :raises AnonymisedError: If the object has been anonymised then this 
                                 method cannot be used.
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

    @check_anonymised
    def left_right_target_contexts(self, incl_target: bool
                                   ) -> List[Tuple[List[str], List[str], List[str]]]:
        '''
        :param incl_target: Whether or not the left and right sentences should 
                            also include the target word.
        :returns: The sentence that is left and right of the target as well as 
                  the words in the target for each target in the sentence.
        :raises AnonymisedError: If the object has been anonymised then this 
                                 method cannot be used.
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

    @check_anonymised
    def replace_target(self, target_index: int, replacement_target_word: str
                       ) -> 'TargetText':
        '''
        :params target_index: The target index of the target word to replace
        :param replacement_target_word: The target word to replace the target 
                                        word at the given index
        :returns: Given the target index and replacement target word it will 
                  replace the target at the index with the new target word and 
                  return a new TargetText object with everything the same apart 
                  from this new target.
        :raises ValueError: If the target_index is less than 0 or an index 
                            number that does not exist.
        :raises OverLappingTargetsError: If the target to replace is contained 
                                         within another target e.g. 
                                         `what a great day` if this has two 
                                         targets `great` and `great day` then 
                                         it will raise this error if you 
                                         replace either word as each is 
                                         within the other.
        :raises AnonymisedError: If the object has been anonymised then this 
                                 method cannot be used.
        :Example: Given the following TargetText Object 
        '''
        self_dict = copy.deepcopy(dict(self))

        number_targets = len(self_dict['targets'])
        if target_index < 0 or target_index >= number_targets:
            raise ValueError('Not a valid target_index number. Number of targets'
                             f'in the current object {number_targets}')
        # Change the target word
        targets = self_dict['targets']
        target_to_be_replaced = targets[target_index]
        targets[target_index] = replacement_target_word

        # Change the target spans
        spans = self_dict['spans']
        span_to_change = spans[target_index]
        spans_to_change: List[int] = []
        for span_index, span in enumerate(spans):
            if span_index == target_index:
                continue
            span: Span
            
            # Check that there are no overlapping targets
            raise_in_target_error = False
            if span.start >= span_to_change.start and span.start < span_to_change.end:
                raise_in_target_error = True
            elif span.end > span_to_change.start and span.end <= span_to_change.end:
                raise_in_target_error = True
            if raise_in_target_error:
                raise OverLappingTargetsError('There are targets that share '
                                              f'the same context {self}')
            
            if span.start >= span_to_change.end:
                spans_to_change.append(span_index)
        
        difference_in_length = len(replacement_target_word) - len(target_to_be_replaced) 
        # Change all of the spans
        for span_index in spans_to_change:
            span = spans[span_index]
            new_start = span.start + difference_in_length
            new_end = span.end + difference_in_length
            spans[span_index] = Span(new_start, new_end)
        # Change the target that is being replaced span by only the end
        new_end = span_to_change.end + difference_in_length
        spans[target_index] = Span(span_to_change.start, new_end)
        # Change the text
        text = self_dict['text']
        span_to_change_start = span_to_change.start
        span_to_change_end = span_to_change.end
        start_text = text[:span_to_change_start]
        end_text = text[span_to_change_end:]
        text = f'{start_text}{replacement_target_word}{end_text}'
        
        self_dict['targets'] = targets
        self_dict['spans'] = spans
        self_dict['text'] = text
        return TargetText(**self_dict)

    def de_anonymise(self, text_dict: Dict[str, str]) -> None:
        '''
        This will set the `anonymised` attribute to False from True and 
        set the `text` key value to the value in the `text` key within the 
        `text_dict` argument.

        :param text_dict: A dictionary that contain the following two keys: 
                          1. `text` and 2. `text_id` where the `text_id` has 
                          to match the current TargetText object `text_id` and 
                          the `text` value will become the new value in the 
                          `text` key for this TargetText object.
        :raises ValueError: If the TargetText object `text_id` does not match 
                            the `text_id` within `text_dict` argument.
        :raises AnonymisedError: If the `text` given does not pass the 
                                 :py:meth:`sanitize` test.
        '''
        current_text_id = self['text_id']
        other_text_id = text_dict['text_id']
        if current_text_id != other_text_id:
            raise ValueError(f"The current `text_id` {current_text_id} "
                             "does not match that of the argument's `text_id`"
                             f" {other_text_id}. For TargetText {self}")
        text = text_dict['text']
        self._storage['text'] = text
        try:
            self.anonymised = False
        except AnonymisedError:
            del self._storage['text']
            sanitize_err = traceback.format_exc()
            raise AnonymisedError('Cannot de-anonymise this TargetText '
                                  f'{self} as it cannot pass the `sanitize`'
                                  ' check of which the following is the '
                                  f'error from said check {sanitize_err}')

    def in_order(self) -> bool:
        '''
        :returns: True if all the `targets` within this TargetText 
                  are in sequential left to right order within the text.
        '''
        spans = self['spans']
        ordered_spans = sorted(spans)
        if ordered_spans != spans:
            return False
        return True

    def re_order(self, keys_not_to_order: Optional[List[str]] = None) -> None:
        '''
        Re-orders the TargetText object so that the targets are in a left to 
        right order within the text, this will then re-order all values within 
        this object that are in a list format into this order. Once the 
        TargetText has been re-ordered it will return True when 
        :py:meth`target_extraction.data_types.TargetText.in_order` is called.

        :param keys_not_to_order: Any key values not to re-order using this 
                                  function e.g. `pos_tags`, `tokenized_text`, 
                                  etc
        :raises AssertionError: If running :py:meth`target_extraction.data_types.TargetText.in_order`
                                after being re-ordered does not return True.
        '''
        def sorting_by_index(index_order: List[int], 
                             value_to_sort: List[Any]) -> List[Any]:
            sorted_value = []
            for index in index_order:
                sorted_value.append(value_to_sort[index])
            return sorted_value

        if keys_not_to_order is None:
            keys_not_to_order = []

        spans: List[Span] = self['spans']
        index_order = sorted(range(len(spans)), key=lambda k: spans[k], 
                             reverse=False)
        new_key_values = {}
        for key, value in self._storage.items():
            try:
                if isinstance(value, list) and key not in keys_not_to_order:
                    # Edge case where the list can be just an empty list
                    if not value:
                        continue
                    # Need to check if the first instance of the value is a 
                    # list and if so then that needs to be sorted and not the 
                    # outer list
                    sorted_value = []
                    if isinstance(value[0], list):
                        for inner_value in value:
                            sorted_inner_value = sorting_by_index(index_order, 
                                                                  inner_value)
                            sorted_value.append(sorted_inner_value)
                    else:
                        sorted_value = sorting_by_index(index_order, value)
                    assert sorted_value
                    new_key_values[key] = sorted_value
            except:
                real_err = traceback.format_exc()
                err_msg = (f'The following error {real_err} has occured on the '
                           f'following key {key} and value {value} for this '
                           f'TargetText {self}')
                raise Exception(err_msg)
        # Covers the rollback problem
        for key, value in new_key_values.items():
            self._storage[key] = value
        self.sanitize()
        assert self.in_order(), print(f'After re-ordering the object is '
                                      f'still not in order:{self}')
    
    def add_unique_key(self, id_key: str, id_key_name: str, 
                       id_delimiter: str = '::') -> None:
        '''
        :param id_key: The name of the key within this TargetText that requires 
                       unique ids that will be stored in `id_key_name`.
        :param id_key_name: The name of the key to associate to these new 
                            unique ids.
        :param id_delimiter: The delimiter to seperate the `text_id` and the 
                             index of the `id_key` that is being represented 
                             by this unique id.
        :raises KeyError: If the `id_key_name` already exists within the 
                          TargetText.
        :raises TypeError: If the value of `id_key` is not of type List.
        :Example: self.add_unique_key(`targets`, `targets_id`) where 
                  `targets`=[`food`, `service`] and `text_id`=`12a5` the 
                  following key will be added to self `targets_id` with the 
                  following value = `[`12a5::0`, `12a5::1`]`
        '''
        self._key_error(id_key)
        text_id = self['text_id']
        if id_key_name in self:
            raise KeyError(f'The new id_key_name {id_key_name} '
                           f'already exists within {self}')
        if not isinstance(self[id_key], list):
            raise TypeError(f'The value of `id_key` {self[id_key]} in {self} '
                            f'has to be of type List and not {type(self[id_key])}')
        new_ids = []
        for index in range(len(self[id_key])):
            new_ids.append(f'{text_id}{id_delimiter}{index}')
        self[id_key_name] = new_ids

    @staticmethod
    def from_json(json_text: str, anonymised: bool = False) -> 'TargetText':
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
        :param anonymised: Whether or not the TargetText object being loaded 
                           is an anonymised version.
        :returns: A TargetText object
        :raises KeyError: If within the JSON representation there is no 
                          `text_id` key. Or if anonymised is False raises a
                          KeyError if there is no `text` key in the JSON 
                          representation.
        '''
        json_target_text = json.loads(json_text)
        text = None
        if not 'text_id' in json_target_text:
            raise KeyError('The JSON text given does not contain a '
                           f'`text_id` field: {json_target_text}')
        if not anonymised:
            if not 'text' in json_target_text:
                raise KeyError('The JSON text given does not contain a `text`'
                               f'field: {json_target_text}')
            text = json_target_text['text']
    
        target_text = TargetText(text=text, anonymised=anonymised,
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

    Attributes:
    
    1. name -- Name associated to the TargetTextCollection.
    2. metadata -- Any metadata to associate to the object e.g. domain of the 
       dataset, all metadata is stored in a dictionary. By default the 
       metadata will always have the `name` attribute within 
       the metadata under the key `name`. If `anonymised` is also True then 
       this will also be in the metadata under the key `anonymised`
    3. anonymised -- If True then the data within the TargetText objects have 
       no text but the rest of the metadata should exist.

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
    12. number_targets -- Returns the total number of targets.
    13. number_categories -- Returns the total number of categories.
    14. category_count -- Returns a dictionary of categories as keys and 
        values as the number of times the category occurs.
    15. target_sentiments -- A dictionary where the keys are target texts and 
        the values are a List of sentiment values that have been associated to 
        that target.
    16. dict_iter -- Returns an interator of all of the TargetText objects 
        within the collection as dictionaries.
    17. unique_distinct_sentiments -- A set of the distinct sentiments within 
        the collection. The length of the set represents the number of distinct 
        sentiments within the collection.
    18. de_anonymise -- This will set the `anonymised` attribute to False 
        from True and set the `text` key value to the value in the `text` 
        key within the `text_dict` argument for each of the TargetTexts in 
        the collection. If any Error is raised this collection will revert back
        fully to being anonymised.
    19. sanitize -- This applies the TargetText.sanitize function to all of 
        the TargetText instances within this collection, affectively ensures 
        that all of the instances follow the specified rules that TargetText 
        instances should follow.
    20. in_order -- This returns True if all TargetText objects within the 
        collection contains a list of targets that are in order of appearance 
        within the text from left to right e.g. if the only TargetText in the 
        collection contains two targets where the first target in the `targets`
        list is the first (left most) target in the text then this method would 
        return True.
    21. re_order -- This will apply :py:meth:`target_extraction.data_types.TargetText.re_order`
        to each TargetText within the collection.
    22. add_unique_key -- Applies the following 
        :py:meth:`target_extraction.data_types.TargetText.add_unique_key` 
        to each TargetText within this collection
    23. key_difference -- Given this collection and another it will return all
        of the keys that the other collection contains which this does not.
    24. combine_data_on_id -- Given this collection and another it will add all
        of the data from the other collection into this collection based on the 
        unique key given. 
    25. one_sentiment_text -- Adds the `text_sentiment_key` to each TargetText 
        within the collection where the value will represent the sentiment value 
        for the text based on the `sentiment_key` values and `average_sentiment` 
        determining how to handle multiple sentiments. This will allow text level 
        classifiers to be trained on target/aspect/category data.

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
                 name: Optional[str] = None, 
                 metadata: Optional[Dict[str, Any]] = None,
                 anonymised: bool = False) -> None:
        '''
        :param target_texts: A list of TargetText instances to add to the 
                             collection.
        :param name: Name to call the collection, this is added to the metadata 
                     automatically and overrides the name key value in the 
                     metadata if exists.
        :param metadata: Any data that you would like to associate to this 
                         TargetTextCollection.
        :param anonymised: Wether or not the TargetText objects should be loaded 
                           in and anonymised, as well as stating whether or not 
                           the whole collection should be anonymised when 
                           loading in new TargetText objects.
        '''
        self._storage = OrderedDict()

        self._anonymised = anonymised
        if target_texts is not None:
            for target_text in target_texts:
                target_text.sanitize()
                self.add(target_text)

        self.metadata = None
        if metadata is not None:
            self.metadata = metadata
        
        if anonymised:
            self.metadata = {} if metadata is None else metadata
            self.metadata['anonymised'] = anonymised

        if name is not None:
            self.name = name
            self.metadata = {} if metadata is None else metadata
            self.metadata['name'] = name
        else:
            self.name = ''

    @property
    def name(self) -> str:
        '''
        :returns: The name attribute.
        '''

        return self._name

    @name.setter
    def name(self, name_string: str) -> None:
        '''
        Sets the value of the name attribute, and also updates the `name` key 
        value in the `metadata` attribute.

        :param name_string: New name to give to the name attribute.
        '''
        self._name = name_string
        self.metadata = {} if self.metadata is None else self.metadata
        self.metadata['name'] = self._name

    @property
    def anonymised(self) -> bool:
        '''
        :returns: True if the data within the TargetTextCollection has been 
                  anonymised. Anonymised data means that there is no text 
                  associated with any of the TargetText objects within the 
                  collection, but all of the metadata is there.
        '''
        return self._anonymised

    @anonymised.setter
    def anonymised(self, value: bool) -> None:
        '''
        Sets whether or not `anonymised` attribute is True or False. This in 
        effect performs the 
        :py:meth:`target_extraction.data_types.TargetText.anonymised`
        on each TargetText object within the collection if True. When you 
        want to set this to False you need to perform 
        :py:meth:`target_extraction.data_types.TargetTextCollection.de_anonymise`.

        :param value: True for anonymised, else False. If True this will 
                      enforce that all the TargetText objects do not have a
                      `text` key/value and the attribute `anonymised` is True.
        :raises AnonymisedError: If the TargetText object within the collection 
                                 cannot be set to the 
                                 `anonymised` value given. If this Error occurs 
                                 then the object will have kept the original 
                                 `anonymised` value.
        '''
        for target_text in self.values():
            target_text.anonymised = value
        self.metadata = {} if self.metadata is None else self.metadata
        self.metadata['anonymised'] = value
        self._anonymised = value 

    def add(self, value: 'TargetText') -> None:
        '''
        Wrapper around set item. Instead of having to add the value the 
        usual way of finding the instances 'text_id' and setting this containers
        key to this value, it does this for you.

        e.g. performs self[value['text_id']] = value

        :param value: The TargetText instance to store in the collection. Will 
                      anonymise the TargetText object if the collection's 
                      anonymised attribute is True.
        '''
        value.anonymised = self.anonymised
        self[value['text_id']] = value

    def to_json(self) -> str:
        '''
        Required as TargetTextCollection is not json serlizable due to the 
        'spans' in the TargetText instances.

        :returns: The object as a list of dictionaries where each the TargetText
                  instances are dictionaries. It will also JSON serialize any 
                  meta data as well.
        '''
        json_text = ''
        for index, target_text_instance in enumerate(self.values()):
            if index != 0:
                json_text += '\n'
            target_text_instance: TargetText
            json_text += target_text_instance.to_json()
        if self.metadata is not None:
            if json_text != '':
                json_text += '\n'
            json_text += json.dumps({'metadata': self.metadata})
        return json_text

    @staticmethod
    def _get_metadata(json_iterable: Iterable[str]) -> Tuple[Union[Dict[str, Any], None],
                                                             Union[str, None], bool]:
        '''
        :param json_iterable: An interable that generates a JSON string, of 
                              which the last string contains the metadata if 
                              it exists. 
        :returns: The metadata for the collection being loaded, as a Tuple of 
                  length 3 where the 3 items are: 1. The metadata, 
                  2. The name of the collection, and 3. Whether it has been 
                  anonymised. The first 2 by default are None and the 3 is 
                  False by default. 
        '''
        metadata = None
        name = None
        anonymised = False
        for line in deque(json_iterable, 1):
            if line.strip():
                json_line = json.loads(line)
                if 'metadata' in json_line:
                    metadata = json_line['metadata']
                    if 'name' in metadata:
                        name = metadata['name']
                    if 'anonymised' in metadata:
                        anonymised = metadata['anonymised']  
        return metadata, name, anonymised

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
        :raises AnonymisedError: If the `TargetText` object that it is loading 
                                 is anonymied but the `target_text_collection_kwargs`
                                 argument contains `anonymised` False, as 
                                 you cannot de-anonymised without performing 
                                 the 
                                 :py:meth:`target_extraction.data_types.TargetTextCollection.de_anonymised`.
        '''
        
       
        if json_text.strip() == '':
            return TargetTextCollection(**target_text_collection_kwargs)

        target_text_instances = []
        metadata, name, anonymised = TargetTextCollection._get_metadata(json_text.split('\n'))
        for line in json_text.split('\n'):
            json_line = json.loads(line)
            if not 'metadata' in json_line:
                target_text_instance = TargetText.from_json(line, anonymised=anonymised)
                target_text_instances.append(target_text_instance)
        # Key word arguments over riding meta data
        if 'name' in target_text_collection_kwargs:
            name = target_text_collection_kwargs['name']
        if 'metadata' in target_text_collection_kwargs:
            metadata = target_text_collection_kwargs['metadata']
        if 'anonymised' in target_text_collection_kwargs:
            anonymised = target_text_collection_kwargs['anonymised']
        return TargetTextCollection(target_text_instances, name=name, 
                                    metadata=metadata, anonymised=anonymised)

    @staticmethod
    def load_json(json_fp: Path, **target_text_collection_kwargs
                  ) -> 'TargetTextCollection':
        '''
        Allows loading a dataset from json. Where the json file is expected to 
        be output from TargetTextCollection.to_json_file as the file will be 
        a json String on each line generated from TargetText.to_json. This 
        will also load any meta data that was stored within the TargetTextCollection.

        :param json_fp: File that contains json strings generated from 
                        TargetTextCollection.to_json_file
        :param target_text_collection_kwargs: Key word arguments to give to 
                                              the TargetTextCollection 
                                              constructor. If there was
                                              any meta data stored within the 
                                              loaded json then these key word 
                                              arguments would over ride the 
                                              meta data stored.
        :returns: A TargetTextCollection based on each new line in the given 
                  json file, and the optional meta data on the last line.
        '''
        target_text_instances = []
        with json_fp.open('r') as json_file:
            metadata, name, anonymised = TargetTextCollection._get_metadata(json_file)

        with json_fp.open('r') as json_file:
            for line in json_file:
                if line.strip():
                    json_line = json.loads(line)
                    if 'metadata' not in json_line:
                        target_text_instance = TargetText.from_json(line, anonymised)
                        target_text_instances.append(target_text_instance)
                        
        # Key word arguments over riding meta data
        if 'name' in target_text_collection_kwargs:
            name = target_text_collection_kwargs['name']
        if 'metadata' in target_text_collection_kwargs:
            metadata = target_text_collection_kwargs['metadata']
        if 'anonymised' in target_text_collection_kwargs:
            anonymised = target_text_collection_kwargs['anonymised']
        return TargetTextCollection(target_text_instances, name=name, 
                                    metadata=metadata, anonymised=anonymised)

    def to_json_file(self, json_fp: Path, 
                     include_metadata: bool = False) -> None:
        '''
        Saves the current TargetTextCollection to a json file which won't be 
        strictly json but each line in the file will be and each line in the 
        file can be loaded in from String via TargetText.from_json. Also the 
        file can be reloaded into a TargetTextCollection using 
        TargetTextCollection.load_json.

        :param json_fp: File path to the json file to save the current data to.
        :param include_metadata: Whether or not to include the metadata when 
                                 writing to file.
        '''
        with json_fp.open('w+') as json_file:
            for index, target_text_instance in enumerate(self.values()):
                target_text_instance: TargetText
                target_text_string = target_text_instance.to_json()
                if index != 0:
                    target_text_string = f'\n{target_text_string}'
                json_file.write(target_text_string)
            if self.metadata is not None and include_metadata:
                metadata_to_write = {'metadata': self.metadata}
                json_file.write(f'\n{json.dumps(metadata_to_write)}')

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

    def number_categories(self) -> int:
        '''
        :returns: The total number of categories in the collection
        :raises ValueError: If one of the category values in the list is of 
                            value None
        '''
        return sum(self.category_count().values())

    def category_count(self) -> Dict[str, int]:
        '''
        :returns: A dictionary of categories as keys and values as the number 
                  of times the category occurs in this TargetTextCollection
        :raises ValueError: If any category has the value of None.
        '''
        categories_count = Counter()
        for target_dict in self.values():
            if target_dict['categories']:
                for category in target_dict['categories']:
                    if category is None:
                        raise ValueError('One of the category value is None, '
                                         f'within {target_dict}')
                    categories_count.update([category])
        return dict(categories_count)

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

    def dict_iterator(self) -> Iterable[Dict[str, Any]]:
        '''
        :returns: An interator of all of the TargetText objects 
                  within the collection as dictionaries.
        '''
        for target_text in self.values():
            target_text: TargetText
            yield dict(target_text)
    
    def unique_distinct_sentiments(self, 
                                   sentiment_key: str = 'target_sentiments'
                                   ) -> Set[int]:
        '''
        :param sentiment_key: The key that represents the sentiment value 
                              for each TargetText object 
        :returns: A set of the distinct sentiments within the collection. 
                  The length of the set represents the number of distinct 
                  sentiments within the collection.
        :raises TypeError: If the value in the sentiment_key is not of type list
        '''
        unique_ds = set()
        for target_object in self.values():
            sentiment_value = target_object[sentiment_key]
            if not isinstance(sentiment_value, list):
                raise TypeError(f'The sentiment key {sentiment_key} contains a'
                                f' value that is not of type List: '
                                f'{sentiment_value}. TargetText object: '
                                f'{target_object}')
            unique_ds.add(len(set(sentiment_value)))
        # Need to remove 0's which come about because an empty list is of 
        # length 0
        if 0 in unique_ds:
            unique_ds.remove(0)
        return unique_ds

    def de_anonymise(self, text_dicts: Iterable[Dict[str, str]]) -> None:
        '''
        This will set the `anonymised` attribute to False 
        from True and set the `text` key value to the value in the `text` 
        key within the `text_dict` argument for each of the TargetTexts in 
        the collection. If any Error is raised this collection will revert back
        fully to being anonymised.

        :param text_dicts: An iterable of dictionaries that contain the following 
                           two keys: 1. `text` and 2. `text_id` where 
                           the `text_id` has to be a key within the current 
                           collection. The `text` associated to that id will 
                           become that TargetText object's text value.
        :raises ValueError: If the length of the `text_dicts` does not match 
                            that of the collection.
        :raises KeyError: If any of the `text_id`s in the `text_dicts` do not 
                          match those within this collection.
        '''
        try:
            self_len = len(self)
            text_dict_len = {}
            for text_dict in text_dicts:
                text_dict_id = text_dict['text_id']
                text_dict_len[text_dict_id] = 1
                if text_dict_id not in self:
                    raise KeyError(f"The key {text_dict_id} from `text_dicts`"
                                   f" is not in this collection.")
                self[text_dict_id].de_anonymise(text_dict)
            text_dict_len = len(text_dict_len)
            if self_len != text_dict_len:
                raise ValueError(f'The length of collection {self_len} is not '
                                 'equal to the length of the `text_dicts` '
                                 f'{text_dict_len}.')
        except Exception as e:
            # Cleans up after the exception as we have to preserve the case 
            # that it is still anonymised
            for target_text in self.values():
                if not target_text.anonymised:
                    target_text.anonymised = True
            raise e
        self.anonymised = False

    def sanitize(self) -> None:
        '''
        This applies the TargetText.sanitize function to all of 
        the TargetText instances within this collection, affectively ensures 
        that all of the instances follow the specified rules that TargetText 
        instances should follow.
        '''

        for target_text in self.values():
            target_text.sanitize()

    def in_order(self) -> bool:
        '''
        This returns True if all TargetText objects within the 
        collection contains a list of targets that are in order of appearance 
        within the text from left to right e.g. if the only TargetText in the 
        collection contains two targets where the first target in the `targets`
        list is the first (left most) target in the text then this method would 
        return True.

        :returns: True if all the `targets` within all the TargetText objects 
                  in this collection are in sequential left to right order 
                  within the text.
        '''
        for target_text in self.values():
            if not target_text.in_order():
                return False
        return True
    
    def re_order(self, keys_not_to_order: Optional[List[str]] = None) -> None:
        '''
        This will apply :py:meth:`target_extraction.data_types.TargetText.re_order`
        to each TargetText within the collection.

        :param keys_not_to_order: Any keys within the TargetTexts that do not 
                                  need re-ordering
        '''
        # This takes into account the rollback problem where an error occurs 
        # halfway through performing the function and half the collection has 
        # been re-ordered where as the other half has not. This will bring it 
        # back into a stable state.
        self_copy = copy.deepcopy(self._storage)
        try:
            for target_text in self.values():
                target_text.re_order(keys_not_to_order)
        except Exception as e:
            self._storage = self_copy
            raise e

    def add_unique_key(self, id_key: str, id_key_name: str, 
                       id_delimiter: str = '::') -> None:
        '''
        Applies the following 
        :py:meth:`target_extraction.data_types.TargetText.add_unique_key` 
        to each TargetText within this collection
        
        :param id_key: The name of the key within this TargetText that requires 
                       unique ids that will be stored in `id_key_name`.
        :param id_key_name: The name of the key to associate to these new 
                            unique ids.
        :param id_delimiter: The delimiter to seperate the `text_id` and the 
                             index of the `id_key` that is being represented 
                             by this unique id.
        '''
        for value in self.values():
            value.add_unique_key(id_key, id_key_name, id_delimiter=id_delimiter)

    def key_difference(self, other_collection: 'TargetTextCollection'
                       ) -> List[str]:
        '''
        :param other_collection: The collection that is being compared to this.
        :returns: A list of keys that represent all of the keys that are in the 
                  other (compared) collection and not in this collection.
        '''
        this_keys = {key for value in self.values() for key in value.keys()}
        other_keys = {key for value in other_collection.values() for key in value.keys()}
        return list(other_keys.difference(this_keys))

    def combine_data_on_id(self, other_collection: 'TargetTextCollection', 
                           id_key: str, data_keys: List[str], 
                           raise_on_overwrite: bool = True,
                           check_same_ids: bool = True) -> None:
        '''
        :param other_collection: The collection that contains the data 
                                 that is to be copied to this collection.
        :param id_key: The key that indicates in each TargetText within 
                       this and the `other_collection` how the values are 
                       to be copied from the `other_collection` to this 
                       collection.
        :param data_keys: The keys of the values in each TargetText within the 
                          `other_collection` that is be copied to the relevant 
                          TargetTexts within this collection. It assumes that if
                          any of key/values are a list of lists that the inner 
                          lists relate to the targets and the outer list is 
                          not related to the targets. 
        :param raise_on_overwrite: If True will raise the 
                                   :py:class:`target_extraction.data_types_util.OverwriteError` 
                                   if any of the `data_keys` exist in any 
                                   of the TargetTexts within this collection.
        :param check_same_ids: If True will ensure that this collection and the 
                               other collection are of same length and check 
                               if each have the same unique ids
        :raises AssertionError: If the number of IDs from the `id_key` does not 
                                match the number of data to be added to a data key
        :raises ValueError: If `check_same_ids` is True and the two collections 
                            are either of not the same length or have  
                            different unique ids according to `id_key` within 
                            the TargetText objects.
        :raises OverwriteError: If `raise_on_overwrite` is True and the any of 
                                the `data_keys` exist in any of the TargetTexts
                                within this collection.
        '''
        def sort_data_by_key(key: str, self_target_text: TargetText, 
                             other_target_text: TargetText, 
                             data_to_sort: List[Any]) -> List[Any]:
            '''
            :param key: A key that appear in both `self_target_text` and 
                        `other_target_text`, where the key for both represents 
                        values that appear in both and are unique.
            :param self_target_text: A TargetText object where the values in 
                                     `key` will determine the sorting performed
                                     to `data_to_sort`. 
            :param other_target_text: The TargetText that represents the `data_to_sort`
                                      and is in this TargetText's sort order 
                                      based on values in `key`
            :param data_to_sort: Data that has come from `other_target_text` that 
                                 is to be sorted based on `key` values from 
                                 `self_target_text`
            :returns: The `data_to_sort` ordered by the values in `self_target_text`
                      key `key`
            :raises AssertionError: If the number of IDs from the `key` does not 
                                    match the number of data_to_sort
            '''
            self_data_values = []
            num_ids = len(other_target_text[key])
            num_data = len(data_to_sort)
            assert_err = (f'The ID key {key} contains {num_ids}, however the '
                          'number of values/data to be added from the other '
                          f'TargetText is {num_data} which is {data_to_sort} '
                          f'OtherTargetText {other_target_text}\n'
                          f'SelfTargetText {self_target_text}')
            assert num_ids == num_data, assert_err

            for self_id_value in self_target_text[key]:
                index_other_id_value = other_target_text[key].index(self_id_value)
                self_data_values.append(data_to_sort[index_other_id_value])
            
            return self_data_values

        if check_same_ids:
            len_self = len(self)
            len_other = len(other_collection)
            if len_self != len_other:
                raise ValueError('The two collections are not the same length. '
                                 f'This length {len_self} other {len_other}')
            self_ids = {_id for value in self.values() for _id in value[id_key]}
            other_ids = {_id for value in other_collection.values() 
                             for _id in value[id_key]}
            self_differences = self_ids.difference(other_ids)
            other_differences = other_ids.difference(self_ids)
            all_differences = self_differences.union(other_differences)
            if len(all_differences):
                raise ValueError(f'The two collections do not contain the same'
                                 f' ids. The difference between this and the '
                                 f'other are the following ids {self_differences}'
                                 f'\nThe difference between the other and this '
                                 f'is the following {other_differences}')
        # If an error occurs would be good to have a roll back poilcy that 
        # will return this collection back to it's original self
        self_copy = copy.deepcopy(self._storage)
        try:
            for text_id, self_target_text in self.items():
                other_target_text = other_collection[text_id]
                # Cannot assume that the unique ids will be in the same order.
                for data_key in data_keys:
                    if data_key in self_target_text and raise_on_overwrite:
                        raise OverwriteError(f'The following data key {data_key}'
                                             ' exists in the following TargetText'
                                             f' {self_target_text} within this collection. '
                                             'The other TargetText that contains '
                                             'this data key to copy the data from '
                                             f'is {other_target_text}')
                    self_data_values = []
                    other_data_values = other_target_text[data_key]
                    # If the other_data_values is a list of a list, need to 
                    # take into account the sorting of the targets should only 
                    # be applied to the inner list.
                    is_inner_list = False
                    if isinstance(other_data_values, list):
                        if other_data_values:
                            if isinstance(other_data_values[0], list):
                                is_inner_list = True
                    if is_inner_list:
                        for other_inner_list_data in other_data_values:
                            self_inner_list_data = sort_data_by_key(id_key, self_target_text, 
                                                                    other_target_text, 
                                                                    other_inner_list_data)
                            self_data_values.append(self_inner_list_data)
                    else:
                        self_data_values = sort_data_by_key(id_key, self_target_text, 
                                                            other_target_text, 
                                                            other_data_values)
                    self_target_text[data_key] = self_data_values
        except Exception as e:
            self._storage = self_copy
            raise e

    def one_sentiment_text(self, sentiment_key: str,
                           average_sentiment: bool = False, 
                           text_sentiment_key: str = 'text_sentiment'
                           ) -> None:
        '''
        Adds the `text_sentiment_key` to each TargetText within the collection 
        where the value will represent the sentiment value for the text based 
        on the `sentiment_key` values and `average_sentiment` determining how 
        to handle multiple sentiments. This will allow text level classifiers 
        to be trained on target/aspect/category data.

        :param sentiment_key: The key in the TargetTexts that represent the 
                              sentiment for the TargetTexts sentence. 
        :param average_sentiment: If False it will only add the `text_sentiment_key` 
                                  to TargetTexts that have one sentiment in the 
                                  `sentiment_key`. If True it will choose the 
                                  most frequent sentiment , ties are decided 
                                  by random choice. If the there are no 
                                  values in `sentiment_key` then 
                                  `text_sentiment_key` will not be added to 
                                  the TargetText.
        :param text_sentiment_key: The key to add the text level sentiment value 
                                   to.
        '''
        for target_text in self.values():
            target_text: TargetText
            target_text._key_error(sentiment_key)

            sentiments = target_text[sentiment_key]
            if average_sentiment:
                if len(sentiments) == 1:
                    target_text[text_sentiment_key] = sentiments[0]
                elif len(sentiments) == 0:
                    continue
                else:
                    sentiment_counts = Counter(sentiments)
                    sorted_counts = sorted(sentiment_counts.items(), 
                                           key=lambda x: x[1], reverse=True)
                    highest_count = sorted_counts[0][1]
                    highest_sentiment_values = []
                    for sentiment_value, count in sorted_counts:
                        if count == highest_count:
                            highest_sentiment_values.append(sentiment_value)
                    assert highest_sentiment_values
                    random_sentiment_value = random.choice(highest_sentiment_values)
                    target_text[text_sentiment_key] = random_sentiment_value
            else:
                if len(sentiments) == 1:
                    target_text[text_sentiment_key] = sentiments[0]

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
                      'text_id' value. Furthermore if the TargetTextCollection's
                      `anonymised` attribute is True then the TargetText object 
                      being added will also be anonymised.
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
        value_copy.anonymised = self.anonymised
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
