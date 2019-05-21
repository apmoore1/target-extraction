from collections.abc import MutableMapping
from collections import OrderedDict
import copy
import json
from pathlib import Path
from typing import Optional, List, Tuple, Iterable, NamedTuple, Any, Callable

class Span(NamedTuple):
    '''
    Span is a named tuple. It has two fields:

    1. start -- An integer that specifies the start of a target word within a 
       text.
    2. end -- An integer that specifies the end of a target word within a text.
    '''
    start: int
    end: int

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
                 spans: Optional[List['Span']] = None, 
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
    4. tokenize_text -- This will add a new key `tokenized_text` to each of the 
       TargetText instances within the collection. This key will store the 
       tokens of the text that is associated to that Target Text instance.
    5. pos_text -- This will add a new key `pos_tags` to each of the TargetText 
       instances within the collection. This key will store the pos tags of the 
       text that is associated to that Target Text instance.
    
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

    def tokenize_text(self, tokenizer: Callable[[str], List[str]]) -> None:
        '''
        This will add a new key `tokenized_text` to each of the TargetText 
        instances within the collection. This key will store the tokens of the 
        text that is associated to that Target Text instance.

        For a set of tokenizers that are definetly comptable see 
        target_extraction.tokenizers module.

        Ensures that the tokenization is character preserving.

        :param tokenizer: The tokenizer to use tokenize the text for each 
                          TargetText instance in the current collection
        :raises ValueError: This is raised if any of the TargetText instances 
                            in the collection contain an empty string.
        :raises ValueError: If the tokenization is not character preserving.
        '''
        def _is_character_preserving(original_text: str, text_tokens: List[str]
                                     ) -> bool:
            '''
            :param original_text: Text that has been tokenized
            :param text_tokens: List of tokens after the text has been tokenized
            :returns: True if the tokenized text when all characters are joined 
                      together is equal to the original text with all it's 
                      characters joined together.
            '''
            tokens_text = ''.join(text_tokens)
            original_text = ''.join(original_text.split())
            if tokens_text == original_text:
                return True
            else:
                return False

        for index, target_text_instance in enumerate(self.values()):
            text = target_text_instance['text']
            tokenized_text = tokenizer(text)
            if index == 0:
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
                raise ValueError('There is no tokens for this TargetText '
                                 f'instance {target_text_instance}')
            if not _is_character_preserving(text, tokenized_text):
                raise ValueError('The tokenization method used is not character'
                                 f' preserving. Original text `{text}`\n'
                                 f'Tokenized text `{tokenized_text}`')
            target_text_instance['tokenized_text'] = tokenized_text
    
    def pos_text(self, tagger: Callable[[str], List[str]]) -> None:
        '''
        This will add a new key `pos_tags` to each of the TargetText 
        instances within the collection. This key will store the pos tags of the 
        text that is associated to that Target Text instance.

        For a set of pos taggers that are definetly comptable see 
        target_extraction.pos_taggers module.

        :param tagger: POS tagger.
        :raises ValueError: This is raised if any of the TargetText instances 
                            in the collection contain an empty string.
        :raises ValueError: If the Target Text instance has not been tokenized.
        :raises ValueError: If the number of pos tags for a Target Text instance
                            does not have the same number of tokens that has 
                            been generated by the tokenizer function.
        '''

        for index, target_text_instance in enumerate(self.values()):
            if 'tokenized_text' not in target_text_instance:
                raise ValueError(f'The Target Text instance {target_text_instance}'
                                 ' has not been tokenized.') 
            tokens = target_text_instance['tokenized_text']
            text = target_text_instance['text']
            pos_tags = tagger(text)
            if index == 0:
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
                                 f'instance {target_text_instance}')
            num_tokens = len(target_text_instance['tokenized_text'])
            if num_tokens != num_pos_tags:
                raise ValueError(f'Number of POS tags {pos_tags} should be the '
                                 f'same as the number of tokens {tokens}')
            
            target_text_instance['pos_tags'] = pos_tags



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
