'''
This module allows TargetTextCollection objects to be analysed and report 
overall statistics.
'''
from collections import defaultdict
from typing import Dict, Any, List, Union

from target_extraction.data_types import TargetTextCollection

def get_sentiment_counts(collection: TargetTextCollection,
                         normalised: bool = True) -> Dict[str, float]:
    '''
    :param collection: The collection containing the sentiment data
    :param normalised: Whether to normalise the values in the dictionary 
                       by the number of targets in the collection.
    :returns: A dictionary where keys are sentiment values and the keys 
              are the number of times they occur in the collection.
    '''
    sentiment_count = defaultdict(lambda: 0)
    for target_text in collection.values():
        if target_text['target_sentiments'] is not None:
            for sentiment_value in target_text['target_sentiments']:
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

def dataset_target_statistics(collections: List[TargetTextCollection],
                              lower_target: bool = True
                              ) -> List[Dict[str, Union[str,int,float]]]:
    '''
    :param collections: A list of collections
    :param lower_target: Whether to lower case the targets before counting them
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
              8. Pos % -- Percentage of positive targets
              9. Neu % -- Percentage of neutral targets
              10. Neg % -- Percentage of Negative targets
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
        sentiment_count = get_sentiment_counts(collection, normalised=True)
        collection_stats['POS %'] = sentiment_count['positive'] * 100
        collection_stats['NEU %'] = sentiment_count['neutral'] * 100
        collection_stats['NEG %'] = sentiment_count['negative'] * 100
        dataset_stats.append(collection_stats)
    return dataset_stats


