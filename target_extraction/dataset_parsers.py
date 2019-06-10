'''
This module contains all the functions that will parse a particular dataset
into a `target_extraction.data_types.TargetTextCollection` object.

Functions:

1. semeval_2014
'''
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element
from xml.etree.ElementTree import ParseError
from typing import List, Union

from target_extraction.data_types import TargetTextCollection, TargetText
from target_extraction.data_types_util import Span

def _semeval_extract_data(sentence_tree: Element, conflict: bool
                          ) -> TargetTextCollection:
    '''
    :param sentence_tree: The root element of the XML tree that has come 
                          from a SemEval XML formatted XML File.
    :param conflict: Whether or not to include targets or categories that 
                     have the `conflict` sentiment value. True is to include 
                     conflict targets and categories.
    :returns: The SemEval data formatted into a 
              `target_extraction.data_types.TargetTextCollection` object.
    '''
    target_text_collection = TargetTextCollection()
    for sentence in sentence_tree:
        text_id = sentence.attrib['id']
        
        targets: List[str] = []
        target_sentiments: List[Union[str, int]] = []
        spans: List[Span] = []

        category_sentiments: List[Union[str, int]] = []
        categories: List[str] = [] 

        for data in sentence:
            if data.tag == 'text':
                text = data.text
            elif data.tag == 'aspectTerms':
                for target in data:
                    # If it is a conflict sentiment and conflict argument True 
                    # skip this target
                    target_sentiment = target.attrib['polarity']
                    if conflict and target_sentiment == 'conflict':
                        continue
                    targets.append(target.attrib['term'])
                    target_sentiments.append(target_sentiment)
                    span_from = int(target.attrib['from'])
                    span_to = int(target.attrib['to'])
                    spans.append(Span(span_from, span_to))
            elif data.tag == 'aspectCategories':
                for category in data:
                    # If it is a conflict sentiment and conflict argument True 
                    # skip this category
                    category_sentiment = category.attrib['polarity']
                    if conflict and category_sentiment == 'conflict':
                        continue
                    categories.append(category.attrib['category'])
                    category_sentiments.append(category.attrib['polarity'])
        target_text_kwargs = {'targets': targets, 'spans': spans, 'text_id': text_id,
                              'target_sentiments': target_sentiments,
                              'categories': categories, 'text': text, 
                              'category_sentiments': category_sentiments}
        for key in target_text_kwargs:
            if not target_text_kwargs[key]:
                target_text_kwargs[key] = None
        target_text = TargetText(**target_text_kwargs)
        target_text_collection.add(target_text)
    return target_text_collection

def semeval_2014(data_fp: Path, conflict: bool) -> TargetTextCollection:
    '''

    The sentiment labels are the following: 1. negative, 2. neutral, 
    3. positive, and 4. conflict. conflict will not appear if the argument 
    `conflict` is False.

    :param data_fp: Path to the SemEval 2014 formatted file.
    :param conflict: Whether or not to include targets or categories that 
                     have the `conflict` sentiment value. True is to include 
                     conflict targets and categories.
    :returns: The SemEval 2014 data formatted into a 
              `target_extraction.data_types.TargetTextCollection` object.
    :raises SyntaxError: If the File passed is detected as not a SemEval 
                         formatted file. 
    :raises `xml.etree.ElementTree.ParseError`: If the File passed is 
                                                not formatted correctly e.g. 
                                                mismatched tags
    '''

    tree = ET.parse(data_fp)
    sentences = tree.getroot()
    if sentences.tag != 'sentences':
        raise SyntaxError('The root of all semeval xml files should '
                          f'be sentences and not {sentences.tag}')
    return _semeval_extract_data(sentences, conflict)

def semeval_2016(data_fp: Path, conflict: bool) -> TargetTextCollection:
    '''
    This is only for subtask 1 files where the review is broken down into 
    sentences. Furthermore if the data contains targets and not just categories 
    the targets and category sentiments are linked and are all stored in the 
    `targets_sentiments` further as some of the datasets only contain category 
    information to make it the same across domains the sentiment values here 
    will always be in the targets_sentiments field.

    The sentiment labels are the following: 1. negative, 2. neutral, 
    3. positive, and 4. conflict. conflict will not appear if the argument 
    `conflict` is False.

    :param data_fp: Path to the SemEval 2016 formatted file.
    :param conflict: Whether or not to include targets and categories that 
                     have the `conflict` sentiment value. True is to include 
                     conflict targets and categories.
    :returns: The SemEval 2016 data formatted into a 
              `target_extraction.data_types.TargetTextCollection` object.
    :raises SyntaxError: If the File passed is detected as not a SemEval 
                         formatted file. 
    :raises `xml.etree.ElementTree.ParseError`: If the File passed is 
                                                not formatted correctly e.g. 
                                                mismatched tags
    '''

    tree = ET.parse(data_fp)
    reviews = tree.getroot()
    if reviews.tag != 'Reviews':
        raise SyntaxError('The root of all semeval xml files should '
                          f'be Reviews and not {sentences.tag}')
    return _semeval_extract_data(sentences, conflict)
