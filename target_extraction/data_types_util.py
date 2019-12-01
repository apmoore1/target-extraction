'''
Module that contains helpful classes and methods for 
`target_extraction.data_types`

classes:

1. Span
2. OverLappingTargetsError
3. AnonymisedError
'''
from typing import NamedTuple

class Span(NamedTuple):
    '''
    Span is a named tuple. It has two fields:

    1. start -- An integer that specifies the start of a target word within a 
       text.
    2. end -- An integer that specifies the end of a target word within a text.
    '''
    start: int = 0
    end: int = 0

class OverLappingTargetsError(Exception):
    '''
    If two targets within the same sentence overlap with each other when they 
    shouldn't.
    '''
    pass

class AnonymisedError(Exception):
   '''
   If the something cannot be performed because the 
   :py:class:`target_extraction.data_types.TargetText` 
   or :py:class:`target_extraction.data_types.TargetTextCollection` 
   object has been anonymised.
   '''
   def __init__(self, error_string: str) -> None:
        '''
        :param error_string: The error string to attach to this exception.
        '''
        super().__init__(error_string)