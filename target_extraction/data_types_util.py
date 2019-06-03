'''
Module that contains helpful classes and methods for 
`target_extraction.data_types`

classes:

1. Span
'''
from typing import NamedTuple

class Span(NamedTuple):
    '''
    Span is a named tuple. It has two fields:

    1. start -- An integer that specifies the start of a target word within a 
       text.
    2. end -- An integer that specifies the end of a target word within a text.
    '''
    start: int
    end: int