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
from collections import OrderedDict
import copy
import json
from pathlib import Path
from typing import Optional, List, Tuple, Iterable, NamedTuple, Any, Callable

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
    3. targets -- List of all target words that occur in the text. Can be None.
    4. spans -- List of Span NamedTuples where each one specifies the start and 
       end of the respective targets within the text.
    5. sentiments -- List of integers sepcifying the sentiment of the 
       respective targets within the text.
    6. categories -- List of categories that represent the targets. NOTE: 
       depending on the dataset and how it is parsed the category can exist 
       but the target does not as the category is a latent variable, in these 
       cases the target will be `NULL` and if there are sentiment values the 
       sentiment will be with respect to both the target and the category 
       value.

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
    6. get_sequence_spans -- The span indexs from the sequence labels given 
       assuming that the sequence labels are in BIO format.
    
    Static Functions:

    1. from_json -- Returns a TargetText object given a json string. For 
       example the json string can be the return of TargetText.to_json.
    '''

    def _check_is_list(self, item: List[Any], item_name: str) -> None:
        '''
        This will check that the argument given is a List and if not will raise 
        a TypeError.

        :param item: The argument that is going to be checked to ensure it is a
                     list.
        :param item_name: Name of the item. This is used within the raised 
                          error message, if an error is raised.
        :raises: TypeError
        '''
        type_err = f'{item_name} should be a list not {type(item)}'
        if not isinstance(item, list):
            raise TypeError(type_err)

    def check_list_sizes(self) -> None:
        '''
        This will check that all of the core lists:
        
        1. targets
        2. spans
        3. sentiments
        4. categories

        Are all of the same length if they are not None. These arguments are 
        found through the self._storage.
        
        Furthermore it checks that if targets are set then spans are also set 
        as well as checking that the spans correspond to the target in the 
        text e.g. if the target is `barry davies` in `today barry davies 
        went` then the spans should be [[6,18]] and if the spans, target or 
        text is any different it will raise a ValueError.

        :returns: Nothing, just raises ValueError if the arguments given in 
                  either the __init__ or through __setitem__ do not conform 
                  to the documentation stated above.
        :raises: ValueError
        '''

        targets = self._storage['targets']
        spans = self._storage['spans']
        categories = self._storage['categories']
        sentiments = self._storage['sentiments']

        text_id = self._storage['text_id']
        text_id_msg = f'Text id that this error refers to {text_id}\n'

        # Checking the length mismatches
        lists_to_check = [targets, spans, categories, sentiments]
        list_lengths = [len(_list) for _list in lists_to_check 
                        if _list is not None]

        current_list_size = -1
        raise_error = False
        for list_length in list_lengths:
            if current_list_size == -1:
                current_list_size = list_length
            else:
                if current_list_size != list_length:
                    raise_error = True
                    break
        if raise_error:
            length_mismatch_msg = 'The targets, spans, categories, and  '\
                                  'sentiments lists given if given are of '\
                                  'different sizes excluding those that are '\
                                  f'None; targets: {targets}, spans: '\
                                  f'{spans}, categories: {categories}'\
                                  f', and sentiments: {sentiments}'
            raise ValueError(text_id_msg + length_mismatch_msg)
        # Checking that if targets are set than so are spans
        if targets is not None and spans is None:
            spans_none_msg = f'If the targets are a list: {targets} then spans'\
                             f' should also be a list and not None: {spans}'
            raise ValueError(text_id_msg + spans_none_msg)
        # Checking that the words Spans reference in the text match the 
        # respective target words
        if targets is not None:
            text = self._storage['text'] 
            for target, span in zip(targets, spans):
                start, end = span.start, span.end
                text_target = text[start:end]
                if text_target != target:
                    target_span_msg = 'The target the spans reference in the '\
                                      f'text: {text_target} does not match '\
                                      f'the target in the targets list: {target}'
                    raise ValueError(text_id_msg + target_span_msg)
        
        

    def __init__(self, text: str, text_id: str,
                 targets: Optional[List[str]] = None, 
                 spans: Optional[List[Span]] = None, 
                 sentiments: Optional[List[int]] = None, 
                 categories: Optional[List[str]] = None):
        # Ensure that the arguments that should be lists are lists.
        self._list_argument_names = ['targets', 'spans', 'sentiments', 
                                     'categories']
        self._list_arguments = [targets, spans, sentiments, categories]
        names_arguments = zip(self._list_argument_names, self._list_arguments)
        for argument_name, list_argument in names_arguments:
            if list_argument is None:
                continue
            self._check_is_list(list_argument, argument_name)

        temp_dict = dict(text=text, text_id=text_id, targets=targets,
                         spans=spans, sentiments=sentiments, 
                         categories=categories)
        self._protected_keys = set(['text', 'text_id', 'targets', 'spans'])
        self._storage = temp_dict
        self.check_list_sizes()

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
        5. sentiments
        6. categories

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
        self.check_list_sizes()

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
    
    def pos_text(self, tagger: Callable[[str], List[str]], 
                 perform_type_checks: bool = False) -> None:
        '''
        This will add a new key `pos_tags` to this TargetText instance.
        This key will store the pos tags of the text that is associated to 
        this Target Text instance.

        For a set of pos taggers that are definetly comptable see 
        target_extraction.pos_taggers module.

        :param tagger: POS tagger.
        :param perform_type_checks: Whether or not to perform type checks 
                                    to ensure the POS tagger returns a List of 
                                    Strings
        :raises TypeError: If the POS tagger given does not return a List of 
                           Strings.
        :raises ValueError: This is raised if the Target Text text is empty
        :raises ValueError: If the this instance has not been tokenized.
        :raises ValueError: If the number of pos tags for this instance
                            does not have the same number of tokens that has 
                            been generated by the tokenizer function.
        '''
        if 'tokenized_text' not in self:
            raise ValueError(f'The Target Text instance {self} has not been '
                             'tokenized.')
        tokens = self['tokenized_text']
        text = self['text']
        pos_tags = tagger(text)

        if perform_type_checks:
            if not isinstance(pos_tags, list):
                raise TypeError('The return type of the tagger function ',
                                f'{tagger} should be a list and not '
                                f'{type(pos_tags)}')
            for tag in pos_tags:
                if not isinstance(tag, str):
                    raise TypeError('The return type of the tagger function ',
                                    f'{tagger} should be a list of Strings'
                                    f' and not a list of {type(tag)}')
        num_pos_tags = len(pos_tags)
        if len(pos_tags) == 0:
            raise ValueError('There are no tags for this TargetText '
                             f'instance {self}')
        num_tokens = len(self['tokenized_text'])
        if num_tokens != num_pos_tags:
            raise ValueError(f'Number of POS tags {pos_tags} should be the '
                             f'same as the number of tokens {tokens}')

        self['pos_tags'] = pos_tags

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
        if self['spans'] is None or self['targets'] is None:
            raise KeyError(f'Expect to have `spans` and `targets` to not be '
                            'None')
        tokens = self['tokenized_text']
        target_spans: List[Span] = self['spans']
        tokens_index = token_index_alignment(text, tokens)

        sequence_labels = ['O' for _ in range(len(tokens))]

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

    def get_sequence_spans(self, sequence_key: str) -> List[Span]:
        '''
        The following sequence label tags are supported: IOB-2. These are the 
        tags that are currently generated by `sequence_labels`

        :param sequence_key: Key to sequence labels such as a BIO sequence 
                             labels. Example key name would be `sequence_labels`
                             after `sequence_labels` function has been called 
                             or more appropiately `predicted_sequence_labels` 
                             when you have predicted sequence labels.
        :returns: The span indexs from the sequence labels given assuming that 
                  the sequence labels are in BIO format.
        :raises ValueError: If the sequence labels that are contained in the 
                            sequence key value contain values other than 
                            `B`, `I`, or `O`.
        '''
        def found_target(start_index: int, end_index: int) -> Span:
            if end_index == 0:
                raise ValueError(f'The end index {end_index} of '
                                 f'a span cannot be 0. sequence label key used '
                                 f'{sequence_key}\nTargetText {self}')
            return Span(start_span_index, end_span_index)


        # number of tokens, sequence labels, and token text indexs should 
        # all be the same, it is if the `sequence_labels` function is used
        tokens = self['tokenized_text']
        token_text_indexs = token_index_alignment(self['text'], tokens)
        sequence_labels = self[sequence_key]
        
        same_target = False
        start_span_index = 0
        end_span_index = 0
        sequence_spans: List[Span] = []
        for text_index, sequence_label in zip(token_text_indexs, 
                                              sequence_labels):
            if sequence_label == 'B':
                if same_target == True:
                    sequence_span = found_target(start_span_index, end_span_index)
                    sequence_spans.append(sequence_span)
                    same_target = False
                    start_span_index = 0
                    end_span_index = 0

                same_target = True
                start_span_index = text_index[0]
                end_span_index = text_index[1]
            elif sequence_label == 'I':
                end_span_index = text_index[1]
            elif sequence_label == 'O':
                if same_target:
                    sequence_span = found_target(start_span_index, end_span_index)
                    sequence_spans.append(sequence_span)
                    same_target = False
                    start_span_index = 0
                    end_span_index = 0
            else:
                raise ValueError('Sequence labels should be `B` `I` or `O` '
                                 f'and not {sequence_label}. Sequence label '
                                 f'key used {sequence_key}\nTargetText {self}')
        if end_span_index != 0:
            sequence_spans.append(Span(start_span_index, end_span_index))
        return sequence_spans

    @staticmethod
    def from_json(json_text: str) -> 'TargetText':
        '''
        This is required as the 'spans' are Span objects which are not json 
        serlizable and are required for TargetText therefore this handles 
        that special case.

        :param json_text: JSON representation of TargetText 
                          (can be from TargetText.to_json)
        :returns: A TargetText object
        '''
        json_target_text = json.loads(json_text)
        for key, value in json_target_text.items():
            if key == 'spans':
                all_spans = []
                for span in value:
                    all_spans.append(Span(*span))
                json_target_text[key] = all_spans
        return TargetText(**json_target_text)


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
    
    Static Functions:

    1. from_json -- Returns a TargetTextCollection object given the json like 
       String from to_json. For example the json string can be the return of 
       TargetTextCollection.to_json.
    2. load_json -- Returns a TargetTextCollection based on each new line in 
       the given json file.
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
        if target_text_instances:
            return TargetTextCollection(target_text_instances, 
                                        **target_text_collection_kwargs)
        return TargetTextCollection(**target_text_collection_kwargs)

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

    def sequence_labels(self) -> None:
        '''
        This applies the TargetText.sequence_labels method across all of 
        the TargetText instances within the collection.

        :raises KeyError: If the current TargetText has not been tokenized.
        :raises ValueError: If two targets overlap the same token(s) e.g 
                            `Laptop cover was great` if `Laptop` and 
                            `Laptop cover` are two seperate targets this should 
                            riase a ValueError as a token should only be 
                            associated to one target.
        '''

        for target_text_instance in self.values():
            target_text_instance.sequence_labels()

    def exact_match_score(self, 
                          predicted_sequence_key: str = 'predicted_sequence_labels'
                          ) -> Tuple[float, float, float]:
        '''        
        Just for clarification we use the sequence label tags to find the 
        predicted spans. However even if you have a perfect sequence label 
        score does not mean you will have a perfect extact span score 
        as the tokenizer used for the sequence labelling might not align 
        perfectly with the annotated spans.

        :param predicted_sequence_key: Key of the predicted sequence labels 
                                       within this TargetText instance.
        :returns: Recall, Precision, and F1 score in a Tuple. All of these 
                  measures are based on exact span matching rather than the 
                  matching of the sequence label tags, this is due to the 
                  annotation spans not always matching tokenization therefore 
                  this removes the tokenization error that can come from the 
                  sequence label measures.
        :raises KeyError: If there are no `sequence_labels` or `spans` or 
                          predicted sequence label key within this TargetText.
        '''
        # tp = True Positive count
        tp = 0.0
        num_pred_true = 0.0
        num_actually_true = 0.0

        for target_text_index, target_text_instance in enumerate(self.values()):
            if target_text_index == 0:
                keys_to_check = ['spans', 'sequence_labels', 
                                f'{predicted_sequence_key}']
                for key in keys_to_check:
                    target_text_instance._key_error(key)
            predicted_spans = target_text_instance.get_sequence_spans(predicted_sequence_key)
            # Add to the number of predicted true and actually true
            predicted_spans: List[Span]
            num_pred_true += len(predicted_spans)

            true_spans: List[Span] = target_text_instance['spans']
            num_actually_true += len(true_spans)
            
            # Find the True Positives
            if len(predicted_spans) != len(set(predicted_spans)):
                raise ValueError(f'Predicted spans {predicted_spans} contain'
                                 f' multiple of the same predicted span. '
                                 f'TargetText: {target_text_instance}')
            if len(true_spans) != len(set(true_spans)):
                raise ValueError(f'True spans {true_spans} contain'
                                 f' multiple of the same true span. '
                                 f'TargetText: {target_text_instance}')
            true_spans = set(true_spans)
            for predicted_span in predicted_spans:
                if predicted_span in true_spans:
                    tp += 1
        
        recall = tp / num_actually_true
        precision = tp / num_pred_true
        f1 = (2 * precision * recall) / (precision + recall)

        return recall, precision, f1


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
