'''
This module contains all the functions that will parse a particular dataset
into a `target_extraction.data_types.TargetTextCollection` object.

Functions:

1. semeval_2014
'''
from pathlib import Path

from target_extraction.data_types import TargetTextCollection

def semeval_2014(data_fp: Path, conflict: bool) -> TargetTextCollection:
    return TargetTextCollection()