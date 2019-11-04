'''
This module contains all the functions that will parse a particular dataset
into a `target_extraction.data_types.TargetTextCollection` object.

Functions:

1. semeval_2014
'''
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element
from xml.etree.ElementTree import ParseError
from typing import List, Union, Dict, Any, Optional
import tempfile
import zipfile
import tarfile

import requests

from target_extraction.data_types import TargetTextCollection, TargetText
from target_extraction.data_types_util import Span

CACHE_DIRECTORY = Path(Path.home(), '.bella_tdsa')

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
                text = text.replace(u'\xa0', u' ')
            elif data.tag == 'aspectTerms':
                for target in data:
                    # If it is a conflict sentiment and conflict argument True 
                    # skip this target
                    target_sentiment = target.attrib['polarity']
                    if not conflict and target_sentiment == 'conflict':
                        continue
                    targets.append(target.attrib['term'].replace(u'\xa0', u' '))
                    target_sentiments.append(target_sentiment)
                    span_from = int(target.attrib['from'])
                    span_to = int(target.attrib['to'])
                    spans.append(Span(span_from, span_to))
            elif data.tag == 'aspectCategories':
                for category in data:
                    # If it is a conflict sentiment and conflict argument True 
                    # skip this category
                    category_sentiment = category.attrib['polarity']
                    if not conflict and category_sentiment == 'conflict':
                        continue
                    categories.append(category.attrib['category'])
                    category_sentiments.append(category.attrib['polarity'])
            elif data.tag == 'Opinions':
                for opinion in data:
                    category_target_sentiment = opinion.attrib['polarity']
                    if not conflict and category_target_sentiment == 'conflict':
                        continue
                    # Handle the case where some of the SemEval 16 files do 
                    # not contain targets and are only category sentiment files
                    if 'target' in opinion.attrib:
                        # Handle the case where there is a category but no 
                        # target
                        target_text = opinion.attrib['target'].replace(u'\xa0', u' ')
                        span_from = int(opinion.attrib['from'])
                        span_to = int(opinion.attrib['to'])
                        # Special cases for poor annotation in SemEval 2016
                        # task 5 subtask 1 Restaurant dataset
                        if text_id == 'DBG#2:15' and target_text == 'NULL':
                            span_from = 0
                            span_to = 0
                        if text_id == "en_Patsy'sPizzeria_478231878:2"\
                           and target_text == 'NULL':
                            span_to = 0
                        if text_id == "en_MercedesRestaurant_478010602:1" \
                           and target_text == 'NULL':
                            span_to = 0
                        if text_id == "en_MiopostoCaffe_479702043:9" \
                           and target_text == 'NULL':
                           span_to = 0
                        if text_id == "en_MercedesRestaurant_478010600:1" \
                           and target_text == 'NULL':
                           span_from = 0
                           span_to = 0
                        if target_text == 'NULL':
                            target_text = None
                            # Special cases for poor annotation in SemEval 2016
                            # task 5 subtask 1 Restaurant dataset
                            if text_id == '1490757:0':
                                target_text = 'restaurant'
                            if text_id == 'TR#1:0' and span_from == 27:
                                target_text = 'spot'
                            if text_id == 'TFS#5:26':
                                target_text = "environment"
                            if text_id == 'en_SchoonerOrLater_477965850:10':
                                target_text = 'Schooner or Later'
                        targets.append(target_text)
                        spans.append(Span(span_from, span_to))
                    categories.append(opinion.attrib['category'])
                    target_sentiments.append(category_target_sentiment)
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
                          f'be Reviews and not {reviews.tag}')
    all_target_texts: List[TargetText] = []
    for review in reviews:
        if len(review) != 1:
            raise SyntaxError('The number of `sentences` tags under the '
                                '`review` tag should be just 1 and not '
                                f'{len(review)}')
        sentences = review[0]
        review_target_texts = list(_semeval_extract_data(sentences, 
                                                         conflict).values())
        all_target_texts.extend(review_target_texts)
    return TargetTextCollection(all_target_texts)

def download_election_folder(cache_dir: Optional[Path] = None) -> Path:
    '''
    Downloads the data for the Election Twitter dataset by 
    `Wang et al, 2017 <https://www.aclweb.org/anthology/E17-1046>` that can be found 
    `here <https://figshare.com/articles/EACL_2017_-_Multi-target_UK_election_Twitter_sentiment_corpus/4479563/1>`_

    This is then further used in the following functions
    :func:`target_extraction.dataset_parsers.wang_2017_election_twitter_train`
    and 
    :func:`target_extraction.dataset_parsers.wang_2017_election_twitter_test`
    as a way to get the data.

    :param cache_dir: The directory where all of the data is stored for 
                      this code base. If None then the cache directory is
                      `dataset_parsers.CACHE_DIRECTORY`
    :returns: The Path to the `Wang 2017 Election Twitter` folder within the 
              `cache_dir`.
    :raises FileNotFoundError: If not all of files where downloaded the first 
                               time. Will require the user to delete either 
                               the cache directory or the 
                               `Wang 2017 Election Twitter` folder within the 
                               cache directory.
    '''
    def untar_folder(tar_file: Path, folder_to_extract_to: Path) -> None:
        with tarfile.open(tar_file) as _tar_file:
            _tar_file.extractall(folder_to_extract_to)
    if cache_dir is None:
        cache_dir = CACHE_DIRECTORY

    dataset_folder_fp = Path(cache_dir, 'Wang 2017 Election Twitter')
    annotation_fp = Path(dataset_folder_fp, 'annotations')
    tweets_fp = Path(dataset_folder_fp, 'tweets')
    test_id_fp = Path(dataset_folder_fp, 'test_id.txt')
    train_id_fp = Path(dataset_folder_fp, 'train_id.txt')
    if dataset_folder_fp.exists():
        # The following Paths must exist in the folder for it to be the correct
        # downloaded directory else raises FileExistsError
        path_to_exist = [annotation_fp, tweets_fp, test_id_fp, train_id_fp]
        for _path in path_to_exist:
            if not _path.exists():
                file_not_err = (f'The following file is not found {_path} '
                                'and should exist as currently in the '
                                'corresponding data directory to resolve this'
                                ' problem please either delete this whole '
                                f'directory {dataset_folder_fp} or use a '
                                'different cache directory other '
                                f'than {cache_dir}')        
                raise FileNotFoundError(file_not_err)
        # As the folder exists and contains all of the data return as we should 
        # not download something for the sake of downloading it
        return dataset_folder_fp
    
    dataset_folder_fp.mkdir(parents=True, exist_ok=True)
    download_url = 'http://ndownloader.figshare.com/articles/4479563/versions/1'
    response = requests.get(download_url, stream=True)
    with tempfile.NamedTemporaryFile('wb+') as download_file:
        for chunk in response.iter_content(chunk_size=128):
            download_file.write(chunk)
        with zipfile.ZipFile(download_file) as download_zip:
            download_zip.extractall(dataset_folder_fp)
    # Need to un tar annotations and tweet folders
    annotation_tar_file = Path(dataset_folder_fp, 'annotations.tar.gz')
    tweet_tar_file = Path(dataset_folder_fp, 'tweets.tar.gz')
    untar_folder(annotation_tar_file, annotation_fp)
    untar_folder(tweet_tar_file, tweets_fp)
    return dataset_folder_fp
    
def _wang_2017_election_parser(train: bool, cache_dir: Optional[Path] = None
                               ) -> TargetTextCollection:
    '''
    Parser for the Election Twitter dataset by 
    `Wang et al, 2017 <https://www.aclweb.org/anthology/E17-1046>` that can be found 
    `here <https://figshare.com/articles/EACL_2017_-_Multi-target_UK_election_Twitter_sentiment_corpus/4479563/1>`_
    
    :param train: Whether to return the Train data. If False returns the 
                  test data.
    :param cache_dir: The directory where all of the data is stored for 
                      this code base. If None then the cache directory is
                      `dataset_parsers.CACHE_DIRECTORY`
    :returns: Either the training or test dataset of the Election Twitter 
              dataset.
    :raises FileNotFoundError: If not all of files where downloaded the first 
                               time. Will require the user to delete either 
                               the cache directory or the 
                               `Wang 2017 Election Twitter` folder within the 
                               cache directory.
    '''
    def get_tweet_data(tweet_folder: Path) -> Dict[str, Dict[str, Any]]:
        '''
        :param tweet_folder: Directory containing files where each one represents 
                             a Tweet and some target dependent sentiment data.
        :returns: A dictionary of tweet IDs as keys and JSON data representing 
                  the tweet target dependent sentiment data as values
        '''

        data = {}
        for file_path in tweet_folder.iterdir():
            file_name = file_path.stem
            tweet_id = file_name.lstrip('5')
            with open(file_path, 'r') as tweet_data:
                data[tweet_id] = json.load(tweet_data)
        return data

    def parse_tweet(tweet_data: Dict[str, Any], annotation_data: Dict[str, Any], 
                    tweet_id: str) -> TargetText:
        '''
        :params tweet_data: Data containing the Tweet information
        :params annotation_data: Data containing the annotation data on the 
                                 Tweet
        :params tweet_id: ID of the Tweet
        :returns: The Tweet data in 
                  :class:`target_extraction.data_types.TargetText` format
        :raises ValueError: If the Target offset cannot be found.
        '''

        def get_offsets(from_offset: int, tweet_text: str, target: str) -> Span:
            offset_shifts = [0, -1, 1]
            for offset_shift in offset_shifts:
                from_offset_shift = from_offset + offset_shift
                to_offset = from_offset_shift + len(target)
                offsets = Span(from_offset_shift, to_offset)
                offset_text = tweet_text[from_offset_shift : to_offset].lower()
                if offset_text == target.lower():
                    return offsets
            raise ValueError(f'Offset {from_offset} does not match target text'
                             f' {target}. Full text {tweet_text}\nid {tweet_id}')

        
        target_id = str(tweet_id)
        target_text = tweet_data['content']
        target_categories = None
        target_category_sentiments = None
        targets = []
        target_spans = []
        target_sentiments = []
        for entity in tweet_data['entities']:
            target_sentiment = annotation_data['items'][str(entity['id'])]
            if target_sentiment == 'doesnotapply':
                continue
    
            target = entity['entity']
            target_span = get_offsets(entity['offset'], target_text, target)
            # Take the target from the text as sometimes the original label 
            # is lower cased when it should not be according to the text.
            target = target_text[target_span.start: target_span.end]
            
            targets.append(target)
            target_spans.append(target_span)
            target_sentiments.append(target_sentiment)
        return TargetText(target_text, target_id, targets, target_spans,
                          target_sentiments, target_categories, 
                          target_category_sentiments)

    def get_data(tweet_id_file: Path, all_tweet_data: Dict[str, Dict[str, Any]], 
                 all_annotation_data: Dict[str, Dict[str, Any]]
                 ) -> TargetTextCollection:
        '''
        :params tweet_id_file: File Path containing a Tweet id on each new line
        :params all_tweet_data: Dictionary containing data about the Tweet where 
                                the keys are Tweet ID's and values a Dict of 
                                information about the Tweet.
        :param all_annotation_data: Dictionary containing annotation data about  
                                    the Tweet where the keys are Tweet ID's  
                                    and values are the annotation data about 
                                    the Tweet in a form of a Dict.
        :returns: The Twitter data into a 
                  :class:`target_extraction.data_types.TargetTextCollection` 
                  object.
        '''
        targets = []
        with tweet_id_file.open('r') as tweet_id_data:
            for tweet_id in tweet_id_data:
                tweet_id = tweet_id.strip()
                tweet_data = all_tweet_data[tweet_id]
                anno_data = all_annotation_data[tweet_id]
                targets.append(parse_tweet(tweet_data, anno_data, tweet_id))
        return TargetTextCollection(targets)

    data_fp = download_election_folder(cache_dir)
    
    tweets_folder = Path(data_fp, 'tweets', 'tweets')
    annotations_folder = Path(data_fp, 'annotations', 'annotations')

    tweet_data = get_tweet_data(tweets_folder)
    annotation_data = get_tweet_data(annotations_folder)

    ids_file = Path(data_fp, 'train_id.txt')
    if not train:
        ids_file = Path(data_fp, 'test_id.txt')
    return get_data(ids_file, tweet_data, annotation_data)
    


def wang_2017_election_twitter_train(cache_dir: Optional[Path] = None
                                     ) -> TargetTextCollection:
    '''
    The data for this function when downloaded is stored within: 
    `Path(cache_dir, 'Wang 2017 Election Twitter')`
    
    :param cache_dir: The directory where all of the data is stored for 
                      this code base. If None then the cache directory is
                      `dataset_parsers.CACHE_DIRECTORY`
    :returns: The Training dataset of the Election Twitter dataset by 
              `Wang et al, 2017 <https://www.aclweb.org/anthology/E17-1046>` 
              that can be found 
              `here <https://figshare.com/articles/EACL_2017_-_Multi-target_UK_election_Twitter_sentiment_corpus/4479563/1>`_
    :raises FileNotFoundError: If not all of files where downloaded the first 
                               time. Will require the user to delete either 
                               the cache directory or the 
                               `Wang 2017 Election Twitter` folder within the 
                               cache directory.
    '''
    return _wang_2017_election_parser(train=True, cache_dir=cache_dir)

def wang_2017_election_twitter_test(cache_dir: Optional[Path] = None
                                    ) -> TargetTextCollection:
    '''
    The data for this function when downloaded is stored within: 
    `Path(cache_dir, 'Wang 2017 Election Twitter')`

    :param cache_dir: The directory where all of the data is stored for 
                      this code base. If None then the cache directory is
                      `dataset_parsers.CACHE_DIRECTORY`
    :returns: The Test dataset of the Election Twitter dataset by 
              `Wang et al, 2017 <https://www.aclweb.org/anthology/E17-1046>` 
              that can be found 
              `here <https://figshare.com/articles/EACL_2017_-_Multi-target_UK_election_Twitter_sentiment_corpus/4479563/1>`_
    :raises FileNotFoundError: If not all of files where downloaded the first 
                               time. Will require the user to delete either 
                               the cache directory or the 
                               `Wang 2017 Election Twitter` folder within the 
                               cache directory.
    '''
    return _wang_2017_election_parser(train=False, cache_dir=cache_dir)