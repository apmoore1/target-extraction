from collections.abc import MutableMapping
from typing import Optional, List, Tuple, Iterable, NamedTuple, Any

class Span(NamedTuple):
    '''
    Span is a named tuple. It has two fields:

    1. start -- An integer that specifies the start of a target word within a 
       text.
    2. end -- An integer that specifies the end of a target word within a text.
    '''
    start: int
    end: int

class Target_Text(MutableMapping):
    '''
    This is a data structure that inherits from MutableMapping which is 
    essentially a python dictionary.

    The following are the default keys that are in all `Target_Text` 
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
    '''

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
        name_argument = [('targets', targets), ('spans', spans),
                         ('sentiments', sentiments), ('categories', categories)]
        for argument_name, list_argument in name_argument:
            if list_argument is None:
                continue
            assert_err = f'{argument_name} should be a list not '\
                         f'{type(list_argument)}'
            assert isinstance(list_argument, list), assert_err

        temp_dict = dict(text=text, text_id=text_id, targets=targets,
                         spans=spans, sentiments=sentiments, 
                         categories=categories)
        self._protected_keys = set(['text', 'text_id'])
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
        return f'Target_Text({self._storage})'

    def __eq__(self, other: 'Target_Text') -> bool:
        '''
        Two Target_Text instances are equal if they both have the same `text_id`
        value.

        :param other: Another Target_Text object that is being compared to this 
                      Target_Text object.
        :returns: True if they have the same `text_id` value else False.
        '''

        if not isinstance(other, Target_Text):
            return False
        elif self['text_id'] != other['text_id']:
            return False
        return True

    def __delitem__(self, key: str) -> None:
        '''
        Given a key that matches a key within self._storage or self.keys() 
        it will delete that key and value from this object.

        NOTE: Currently  'text' and 'text_id' are keys that cannot be deleted.

        :param key: Key and its respective value to delete from this object.
        '''
        if key in self._protected_keys:
            raise KeyError('Cannot delete a key that is protected, list of '
                           f' protected keys: {self._protected_keys}')
        del self._storage[key]
        self.check_list_sizes()

    def __setitem__(self, key: str, value: Any) -> None:
        '''
        Given a key and a respected value it will either change that current 
        keys value to the one gien here or create a new key with that value.

        NOTE: Currently  'text' and 'text_id' are keys that cannot be changed.

        :param key: Key to be added or changed
        :param value: Value associated to the given key.
        '''
        if key in self._protected_keys:
            raise KeyError('Cannot change a key that is protected, list of '
                           f' protected keys: {self._protected_keys}')
        self._storage[key] = value
        self.check_list_sizes()
