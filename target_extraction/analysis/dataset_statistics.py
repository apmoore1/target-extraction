'''
This module allows TargetTextCollection objects to be analysed and report 
overall statistics.
'''
from collections import defaultdict
from typing import Dict, Any, List, Union, Callable

import pandas as pd

from target_extraction.data_types import TargetTextCollection
from target_extraction.tokenizers import spacy_tokenizer

def get_sentiment_counts(collection: TargetTextCollection,
                         sentiment_key: str,
                         normalised: bool = True) -> Dict[str, float]:
    '''
    :param collection: The collection containing the sentiment data
    :param sentiment_key: The key in each TargetText within the collection that 
                          contains the True sentiment value.
    :param normalised: Whether to normalise the values in the dictionary 
                       by the number of targets in the collection.
    :returns: A dictionary where keys are sentiment values and the keys 
              are the number of times they occur in the collection.
    '''
    sentiment_count = defaultdict(lambda: 0)
    for target_text in collection.values():
        if target_text[sentiment_key] is not None:
            for sentiment_value in target_text[sentiment_key]:
                sentiment_count[sentiment_value] += 1
    number_targets = collection.number_targets()
    assert number_targets == sum(sentiment_count.values())
    if normalised:
        for sentiment, count in sentiment_count.items():
            sentiment_count[sentiment] = float(count) / float(number_targets)
    return dict(sentiment_count)

def average_target_per_sentences(collection: TargetTextCollection, 
                                 sentence_must_contain_targets: bool) -> float:
    '''
    :param collection: Collection to calculate average target per sentence (ATS) 
                       on.
    :param sentence_must_contain_targets: Whether or not the sentences within the 
                                          collection must contains at least one 
                                          target. This filtering would affect 
                                          the value of the dominator stated in 
                                          the returns.  
    :returns: The ATS for the given collection. Which is: 
              Number of targets / number of sentences
    '''
    number_targets = float(collection.number_targets())
    if sentence_must_contain_targets:
        number_sentences = len(collection.samples_with_targets())
    else:
        number_sentences = len(collection)
    return number_targets / float(number_sentences)

def tokens_per_target(collection: TargetTextCollection,
                      target_key: str, 
                      tokeniser: Callable[[str], List[str]],
                      normalise: bool = False,
                      cumulative_percentage: bool = False) -> Dict[int, int]:
    '''
    :param collection: collection to analyse
    :param target_key: The key within each sample in the collection that contains 
                       the list of targets to be analysed. This can also be the 
                       predicted target key, which might be useful for error 
                       analysis.
    :param tokenizer: The tokenizer to use to split the target(s) into tokens. See 
                      for a module of comptabile tokenisers 
                      :py:mod:`target_extraction.tokenizers`
    :param normalise: The values are normalised based on the total number of 
                      targets. (This does not change the return if 
                      `cumulative_percentage` is True)
    :param cumulative_percentage: If the return should not be frequency counts of 
                                  the number of tokens in each target but rather 
                                  the cumulative percentage of targets with 
                                  that number of tokens.
    :returns: The dictionary where keys are the target length based on the number 
              of tokens in the target and the values are the number of targets 
              in the dataset that contain that number of tokens (same target can 
              be counted more than once if it exists in the dataset more then 
              once). **This is a defaultdict where the value will be 0 if the key 
              does not exist.**
    '''
    lengths = defaultdict(lambda: 0)
    target_count = collection.target_count(lower=False, target_key=target_key)
    total_target_count = sum(target_count.values())
    for target, count in target_count.items():
        length = len(tokeniser(target))
        if normalise:
            count = count / total_target_count
        lengths[length] += count
    if cumulative_percentage:
        lengths = sorted(lengths.items(), key=lambda x: x[0])
        temp_lengths = {}
        current_percentage = 0.0
        for length, count in lengths:
            percentage = (count/total_target_count) * 100
            temp_lengths[length] = current_percentage + percentage
            current_percentage += percentage
        lengths = temp_lengths
    return lengths

def _statistics_to_dataframe(collection_statistics: List[Dict[str, Union[str,int,float]]]
                             ) -> pd.DataFrame:
    '''
    :param collection_statistics: The dictionaries to be converted into 
                                  a single dataframe.
    :returns: The collection statistics given into a dataframe where all columns 
              are the key names and the values are the associated values in the 
              keys from the list of dictionaries.
    '''
    pd_dict = defaultdict(list)
    for collection_statistic in collection_statistics:
        for stat_key, stat_value in collection_statistic.items():
            pd_dict[stat_key].append(stat_value)
    return pd.DataFrame(pd_dict)

def dataset_target_extraction_statistics(collections: List[TargetTextCollection],
                                         lower_target: bool = True,
                                         target_key: str = 'targets',
                                         tokeniser: Callable[[str], List[str]]=spacy_tokenizer(),
                                         dataframe_format: bool = False
                                         ) -> List[Dict[str, Union[str,int,float]]]:
    '''
    :param collections: A list of collections
    :param lower_target: Whether to lower case the targets before counting them
    :param target_key: The key within each sample in each collection that contains 
                       the list of targets to be analysed. This can also be the 
                       predicted target key, which might be useful for error 
                       analysis.
    :param tokenizer: The tokenizer to use to split the target(s) into tokens. See 
                      for a module of comptabile tokenisers 
                      :py:mod:`target_extraction.tokenizers`. This is required 
                      to give statistics on target length.
    :param dataframe_format: If True instead of a list of dictionaries the 
                             return will be a pandas dataframe
    :returns: A list of dictionaries each containing the statistics for the 
              associated collection. Each dictionary will have the following 
              keys:
              1. Name -- this comes from the collection's name attribute
              2. No. Sentences -- number of sentences in the collection
              3. No. Sentences(t) -- number of sentence that contain 
                 targets.
              4. No. Targets -- number of targets
              5. No. Uniq Targets -- number of unique targets
              6. ATS -- Average Target per Sentence (ATS)
              7. ATS(t) -- ATS but where all sentences in the collection must 
                 contain at least one target.
              8. TL (1) -- Percentage of targets that are length 1 based on the 
                 number of tokens.
              9. TL (2) -- Percentage of targets that are length 2 based on the 
                 number of tokens.
              10. TL (3+) -- Percentage of targets that are length 3+ based on the 
                  number of tokens.
    '''
    dataset_stats: List[Dict[str, Union[str,int,float]]] = []
    for collection in collections:
        collection_stats = {}
        collection_stats['Name'] = collection.name
        collection_stats['No. Sentences'] = len(collection)
        collection_stats['No. Sentences(t)'] = len(collection.samples_with_targets())
        collection_stats['No. Targets'] = collection.number_targets()
        collection_stats['No. Uniq Targets'] = len(collection.target_count(lower=lower_target))
        collection_stats['ATS'] = average_target_per_sentences(collection, False)
        collection_stats['ATS(t)'] = average_target_per_sentences(collection, True)
        
        target_lengths = tokens_per_target(collection, target_key, tokeniser, normalise=True)
        collection_stats['TL 1 %'] = round(target_lengths[1] * 100, 2)
        collection_stats['TL 2 %'] = round(target_lengths[2] * 100, 2)
        three_plus = sum([fraction for token_length, fraction in target_lengths.items() 
                          if token_length > 2])
        collection_stats['TL 3+ %'] = round(three_plus * 100, 2)
        dataset_stats.append(collection_stats)
    if dataframe_format:
        return _statistics_to_dataframe(dataset_stats)
    return dataset_stats

def dataset_target_sentiment_statistics(collections: List[TargetTextCollection],
                                        lower_target: bool = True,
                                        target_key: str = 'targets',
                                        tokeniser: Callable[[str], List[str]]=spacy_tokenizer(),
                                        sentiment_key: str = 'target_sentiments',
                                        dataframe_format: bool = False
                                        ) -> Union[List[Dict[str, Union[str,int,float]]], 
                                                   pd.DataFrame]:
    '''
    :param collections: A list of collections
    :param lower_target: Whether to lower case the targets before counting them
    :param sentiment_key: The key in each TargetText within each collection that 
                          contains the True sentiment value.
    :param dataframe_format: If True instead of a list of dictionaries the 
                             return will be a pandas dataframe
    :returns: A list of dictionaries each containing the statistics for the 
              associated collection. Each dictionary will have the keys from 
              :py:func:`dataset_target_extraction_statistics` and the following 
              in addition:
              1. Pos % -- Percentage of positive targets
              2. Neu % -- Percentage of neutral targets
              3. Neg % -- Percentage of Negative targets
    '''
    initial_dataset_stats = dataset_target_extraction_statistics(collections, 
                                                                 lower_target=lower_target, 
                                                                 target_key=target_key, 
                                                                 tokeniser=tokeniser,
                                                                 dataframe_format=False)
    dataset_stats = []
    for collection, collection_stats in zip(collections, initial_dataset_stats):
        sentiment_count = get_sentiment_counts(collection, normalised=True, 
                                               sentiment_key=sentiment_key)
        collection_stats['POS %'] = round(sentiment_count['positive'] * 100, 2)
        collection_stats['NEU %'] = round(sentiment_count['neutral'] * 100, 2)
        collection_stats['NEG %'] = round(sentiment_count['negative'] * 100, 2)
        dataset_stats.append(collection_stats)
    if dataframe_format:
        return _statistics_to_dataframe(dataset_stats)
    return dataset_stats




