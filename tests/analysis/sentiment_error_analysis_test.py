from typing import List
from pathlib import Path
import math

import pytest

from target_extraction.data_types_util import Span
from target_extraction.data_types import TargetTextCollection, TargetText
from target_extraction.analysis.sentiment_error_analysis import same_one_sentiment
from target_extraction.analysis.sentiment_error_analysis import same_multi_sentiment
from target_extraction.analysis.sentiment_error_analysis import similar_sentiment
from target_extraction.analysis.sentiment_error_analysis import different_sentiment
from target_extraction.analysis.sentiment_error_analysis import unknown_targets
from target_extraction.analysis.sentiment_error_analysis import known_sentiment_known_target
from target_extraction.analysis.sentiment_error_analysis import unknown_sentiment_known_target
from target_extraction.analysis.sentiment_error_analysis import distinct_sentiment
from target_extraction.analysis.sentiment_error_analysis import count_error_key_occurrence
from target_extraction.analysis.sentiment_error_analysis import n_shot_targets, n_shot_subsets
from target_extraction.analysis.sentiment_error_analysis import reduce_collection_by_key_occurrence
from target_extraction.analysis.sentiment_error_analysis import swap_list_dimensions
from target_extraction.analysis.sentiment_error_analysis import num_targets_subset
from target_extraction.analysis.sentiment_error_analysis import tssr_raw
from target_extraction.analysis.sentiment_error_analysis import tssr_subset
from target_extraction.analysis.sentiment_error_analysis import NoSamplesError
from target_extraction.analysis.sentiment_error_analysis import error_analysis_wrapper
from target_extraction.analysis.sentiment_error_analysis import ERROR_SPLIT_SUBSET_NAMES
from target_extraction.analysis.sentiment_error_analysis import subset_metrics
from target_extraction.analysis.sentiment_error_analysis import subset_name_to_error_split
from target_extraction.analysis.sentiment_metrics import accuracy, macro_f1
from target_extraction.analysis.sentiment_error_analysis import error_split_df
from target_extraction.analysis.sentiment_error_analysis import _error_split_df

DATA_DIR = Path(__file__, '..', '..', 'data', 'analysis', 'sentiment_error_analysis').resolve()

def target_text_examples(target_sentiment_values: List[List[str]]
                         ) -> List[TargetText]:
    '''
    :param target_sentiment_values: A list of 3 lists where each inner list 
                                    represent the sentiment values for each 
                                    of the targets in that TargetText object
    '''
    text = 'The laptop case was great and cover was rubbish'
    text_ids = ['0', 'another_id', '2']
    spans = [[Span(4, 15)], [Span(30, 35)], [Span(4, 15), Span(30, 35)]]
    targets = [['laptop case'], ['cover'], ['laptop case', 'cover']]
    categories = [['LAPTOP#CASE'], ['LAPTOP'], ['LAPTOP#CASE', 'LAPTOP']]

    target_text_examples = []
    for i in range(3):
        example = TargetText(text, text_ids[i], targets=targets[i],
                                spans=spans[i], 
                                target_sentiments=target_sentiment_values[i],
                                categories=categories[i])
        target_text_examples.append(example)
    return target_text_examples

def get_error_counts(dataset: TargetTextCollection, error_key: str
                     ) -> List[List[int]]:
    error_counts: List[List[int]] = []
    for target_data in dataset.values():
        error_counts.append(target_data[error_key])
    return error_counts

@pytest.mark.parametrize("lower", (False, True))
def test_same_one_sentiment(lower: bool):
    # The train and test collection where there are no targets 
    # with the same sentiment
    pos, neg, neu = 'positive', 'negative', 'neutral'
    train_sentiments = [[pos], [neg], [neg, pos]]
    test_sentiments = [[neu], [neu], [neu, neu]]
    train = TargetTextCollection(target_text_examples(train_sentiments))
    test = TargetTextCollection(target_text_examples(test_sentiments))
    test = same_one_sentiment(test, train, lower)
    answer = [[0], [0], [0,0]]
    assert answer == get_error_counts(test, 'same_one_sentiment')
    # The train and test collection where one target contains two sentiment 
    # where one is the same but the other is not
    test_sentiments = [[pos], [neu], [pos, neu]]
    test = TargetTextCollection(target_text_examples(test_sentiments))
    test = same_one_sentiment(test, train, lower)
    answer = [[0], [0], [0,0]]
    assert answer == get_error_counts(test, 'same_one_sentiment')
    # The train and test collection where one target has the same sentiment
    train_sentiments = [[pos], [neg], [pos, pos]]
    train = TargetTextCollection(target_text_examples(train_sentiments))
    test = same_one_sentiment(test, train, lower)
    answer = [[1], [0], [1,0]]
    assert answer == get_error_counts(test, 'same_one_sentiment')
    # Where there is a target with the same sentiment if it is not lower 
    # cased
    train_sentiments = [[pos], [neg], [neg, pos]]
    test_sentiments = [[pos], [neu], [pos, neu]]
        
    train = TargetTextCollection(target_text_examples(train_sentiments))
    test = TargetTextCollection(target_text_examples(test_sentiments))
    answer = [[1], [0], [1,0]]
    if not lower:
        train['2']._storage['targets'] = ['Laptop case', 'cover']
    else:
        answer = [[0], [0], [0,0]]
    test = same_one_sentiment(test, train, lower)
    assert answer == get_error_counts(test, 'same_one_sentiment')
    # Where both have no targets
    empty_target = TargetText(text='something', text_id='1', targets=[], 
                              spans=[], target_sentiments=[])
    test = TargetTextCollection([empty_target])
    train = TargetTextCollection([empty_target])
    test = same_one_sentiment(test, train, lower)
    assert [[]] == get_error_counts(test, 'same_one_sentiment')
    # Where the targets are it should return an empty list
    no_targets = TargetText(text='something', text_id='1')
    test = TargetTextCollection([no_targets])
    test = same_one_sentiment(test, train, lower)
    assert [[]] == get_error_counts(test, 'same_one_sentiment')

@pytest.mark.parametrize("lower", (False, True))
def test_same_multi_sentiment(lower: bool):
    # The train and test collection where there are no targets 
    # with the same sentiment
    pos, neg, neu = 'positive', 'negative', 'neutral'
    train_sentiments = [[pos], [neg], [neg, pos]]
    test_sentiments = [[neu], [neu], [neu, neu]]
    train = TargetTextCollection(target_text_examples(train_sentiments))
    test = TargetTextCollection(target_text_examples(test_sentiments))
    test = same_multi_sentiment(test, train, lower)
    answer = [[0], [0], [0,0]]
    assert answer == get_error_counts(test, 'same_multi_sentiment')
    # The train and test collection where one target contains two sentiment 
    # where one is the same but the other is not
    test_sentiments = [[pos], [neu], [pos, neu]]
    test = TargetTextCollection(target_text_examples(test_sentiments))
    test = same_multi_sentiment(test, train, lower)
    answer = [[0], [0], [0,0]]
    assert answer == get_error_counts(test, 'same_multi_sentiment')
    # The train and test collection where one target has the same sentiment
    train_sentiments = [[pos], [neg], [pos, pos]]
    train = TargetTextCollection(target_text_examples(train_sentiments))
    test = same_multi_sentiment(test, train, lower)
    answer = [[0], [0], [0,0]]
    assert answer == get_error_counts(test, 'same_multi_sentiment')
    # Where one of the targets has two different sentiments that are the same
    train_sentiments = [[pos], [neu], [pos, neg]]
    train = TargetTextCollection(target_text_examples(train_sentiments))
    test_sentiments = [[pos], [neu], [pos, neg]]
    test = TargetTextCollection(target_text_examples(test_sentiments))
    test = same_multi_sentiment(test, train, lower)
    answer = [[0], [1], [0,1]]
    assert answer == get_error_counts(test, 'same_multi_sentiment')
    # The same as before but will not work in the cased version
    train = TargetTextCollection(target_text_examples(train_sentiments))
    test = TargetTextCollection(target_text_examples(test_sentiments))
    test['2']._storage['targets'] = ['laptop case', 'Cover']
    test['2']._storage['text'] = 'The laptop case was great and Cover was rubbish'
    train['2']._storage['targets'] = ['laptop case', 'Cover']
    train['2']._storage['text'] = 'The laptop case was great and Cover was rubbish'
    if not lower:
        answer = [[0], [0], [0,0]]
    test = same_multi_sentiment(test, train, lower)
    assert answer == get_error_counts(test, 'same_multi_sentiment')
    # Where both have no targets
    empty_target = TargetText(text='something', text_id='1', targets=[], 
                              spans=[], target_sentiments=[])
    test = TargetTextCollection([empty_target])
    train = TargetTextCollection([empty_target])
    test = same_multi_sentiment(test, train, lower)
    assert [[]] == get_error_counts(test, 'same_multi_sentiment')
    # Where the targets are it should return an empty list
    no_targets = TargetText(text='something', text_id='1')
    test = TargetTextCollection([no_targets])
    test = same_multi_sentiment(test, train, lower)
    assert [[]] == get_error_counts(test, 'same_multi_sentiment')

@pytest.mark.parametrize("lower", (False, True))
def test_similar_sentiment(lower: bool):
    # The train and test collection where there are no targets 
    # with the same sentiment
    pos, neg, neu = 'positive', 'negative', 'neutral'
    train_sentiments = [[pos], [neg], [neg, pos]]
    test_sentiments = [[neu], [neu], [neu, neu]]
    train = TargetTextCollection(target_text_examples(train_sentiments))
    test = TargetTextCollection(target_text_examples(test_sentiments))
    test = similar_sentiment(test, train, lower)
    answer = [[0], [0], [0,0]]
    assert answer == get_error_counts(test, 'similar_sentiment')
    # The train and test collection where one target contains two sentiment 
    # where one is the same but the other is not
    test_sentiments = [[pos], [neu], [pos, neu]]
    test = TargetTextCollection(target_text_examples(test_sentiments))
    test = similar_sentiment(test, train, lower)
    answer = [[1], [0], [1,0]]
    assert answer == get_error_counts(test, 'similar_sentiment')
    # The train and test collection where one target has the same sentiment
    train_sentiments = [[pos], [neg], [pos, pos]]
    train = TargetTextCollection(target_text_examples(train_sentiments))
    test = similar_sentiment(test, train, lower)
    answer = [[0], [0], [0,0]]
    assert answer == get_error_counts(test, 'similar_sentiment')
    # Where one of the targets has two different sentiments that are the same
    train_sentiments = [[pos], [neu], [pos, neg]]
    train = TargetTextCollection(target_text_examples(train_sentiments))
    test_sentiments = [[pos], [neu], [pos, neg]]
    test = TargetTextCollection(target_text_examples(test_sentiments))
    test = similar_sentiment(test, train, lower)
    answer = [[0], [0], [0,0]]
    assert answer == get_error_counts(test, 'similar_sentiment')
    # The same as the one with [[1], [0], [1,0]] before but will not work in 
    # the cased version
    train_sentiments = [[pos], [neg], [neg, pos]]
    test_sentiments = [[pos], [neu], [pos, neu]]
    train = TargetTextCollection(target_text_examples(train_sentiments))
    test = TargetTextCollection(target_text_examples(test_sentiments))
    test['2']._storage['targets'] = ['Laptop case', 'cover']
    test['2']._storage['text'] = 'The Laptop case was great and cover was rubbish'
    train['2']._storage['targets'] = ['Laptop case', 'cover']
    train['2']._storage['text'] = 'The Laptop case was great and cover was rubbish'
    answer = [[1], [0], [1,0]]
    if not lower:
        answer = [[0], [0], [0,0]]
    test = similar_sentiment(test, train, lower)
    assert answer == get_error_counts(test, 'similar_sentiment')
    # Where both have no targets
    empty_target = TargetText(text='something', text_id='1', targets=[], 
                              spans=[], target_sentiments=[])
    test = TargetTextCollection([empty_target])
    train = TargetTextCollection([empty_target])
    test = similar_sentiment(test, train, lower)
    assert [[]] == get_error_counts(test, 'similar_sentiment')
    # Where the targets are it should return an empty list
    no_targets = TargetText(text='something', text_id='1')
    test = TargetTextCollection([no_targets])
    test = similar_sentiment(test, train, lower)
    assert [[]] == get_error_counts(test, 'similar_sentiment')

@pytest.mark.parametrize("lower", (False, True))
def test_different_sentiment(lower: bool):
    # The train and test collection where there are no targets 
    # with the same sentiment
    pos, neg, neu = 'positive', 'negative', 'neutral'
    train_sentiments = [[pos], [neg], [neg, pos]]
    test_sentiments = [[neu], [neu], [neu, neu]]
    train = TargetTextCollection(target_text_examples(train_sentiments))
    test = TargetTextCollection(target_text_examples(test_sentiments))
    test = different_sentiment(test, train, lower)
    answer = [[1], [1], [1,1]]
    assert answer == get_error_counts(test, 'different_sentiment')
    # The train and test collection where one target contains two sentiment 
    # where one is the same but the other is not
    test_sentiments = [[pos], [neu], [pos, neu]]
    test = TargetTextCollection(target_text_examples(test_sentiments))
    test = different_sentiment(test, train, lower)
    answer = [[0], [1], [0,1]]
    assert answer == get_error_counts(test, 'different_sentiment')
    # The train and test collection where one target has the same sentiment
    train_sentiments = [[pos], [neg], [pos, pos]]
    train = TargetTextCollection(target_text_examples(train_sentiments))
    test = different_sentiment(test, train, lower)
    answer = [[0], [1], [0,1]]
    assert answer == get_error_counts(test, 'different_sentiment')
    # Where one of the targets has two different sentiments that are the same
    train_sentiments = [[pos], [neu], [pos, neg]]
    train = TargetTextCollection(target_text_examples(train_sentiments))
    test_sentiments = [[pos], [neu], [pos, neg]]
    test = TargetTextCollection(target_text_examples(test_sentiments))
    test = different_sentiment(test, train, lower)
    answer = [[0], [0], [0,0]]
    assert answer == get_error_counts(test, 'different_sentiment')
    # The same as the one with [[0], [1], [0,1]] before but will only work in 
    # the cased version
    train_sentiments = [[pos], [neg], [neg, pos]]
    test_sentiments = [[pos], [neu], [pos, neu]]
    train = TargetTextCollection(target_text_examples(train_sentiments))
    test = TargetTextCollection(target_text_examples(test_sentiments))
    test['2']._storage['targets'] = ['laptop case', 'Cover']
    test['2']._storage['text'] = 'The laptop case was great and Cover was rubbish'
    test['2']._storage['target_sentiments'] = [pos, pos]
    train['2']._storage['targets'] = ['laptop case', 'Cover']
    train['2']._storage['text'] = 'The laptop case was great and Cover was rubbish'
    answer = [[0], [0], [0,0]]
    if not lower:
        answer = [[0], [1], [0,0]]
    test = different_sentiment(test, train, lower)
    assert answer == get_error_counts(test, 'different_sentiment')
    # Where both have no targets
    empty_target = TargetText(text='something', text_id='1', targets=[], 
                              spans=[], target_sentiments=[])
    test = TargetTextCollection([empty_target])
    train = TargetTextCollection([empty_target])
    test = different_sentiment(test, train, lower)
    assert [[]] == get_error_counts(test, 'different_sentiment')
    # Where the targets are it should return an empty list
    no_targets = TargetText(text='something', text_id='1')
    test = TargetTextCollection([no_targets])
    test = different_sentiment(test, train, lower)
    assert [[]] == get_error_counts(test, 'different_sentiment')

@pytest.mark.parametrize("lower", (False, True))
def test_known_sentiment_known_target(lower: bool):
    # The train and test collection where there are no targets 
    # with the same sentiment
    pos, neg, neu = 'positive', 'negative', 'neutral'
    train_sentiments = [[pos], [neg], [neg, pos]]
    test_sentiments = [[neu], [neu], [neu, neu]]
    train = TargetTextCollection(target_text_examples(train_sentiments))
    test = TargetTextCollection(target_text_examples(test_sentiments))
    test = known_sentiment_known_target(test, train, lower)
    answer = [[0], [0], [0,0]]
    assert answer == get_error_counts(test, 'known_sentiment_known_target')
    # The train and test collection where one target contains two sentiment 
    # where one is the same but the other is not
    test_sentiments = [[neg], [pos], [neu, neu]]
    test = TargetTextCollection(target_text_examples(test_sentiments))
    test = known_sentiment_known_target(test, train, lower)
    answer = [[1], [1], [0,0]]
    assert answer == get_error_counts(test, 'known_sentiment_known_target')
    # all targets are highlighted
    train_sentiments = [[neg], [neu], [neu, pos]]
    train = TargetTextCollection(target_text_examples(train_sentiments))
    test = known_sentiment_known_target(test, train, lower)
    answer = [[1], [1], [1,1]]
    assert answer == get_error_counts(test, 'known_sentiment_known_target')
    # Case version
    train_sentiments = [[pos], [neg], [neg, pos]]
    test_sentiments = [[neu], [pos], [neu, neu]]
    train = TargetTextCollection(target_text_examples(train_sentiments))
    test = TargetTextCollection(target_text_examples(test_sentiments))
    test['2']._storage['targets'] = ['laptop case', 'Cover']
    test['2']._storage['text'] = 'The laptop case was great and Cover was rubbish'
    train['2']._storage['targets'] = ['laptop case', 'Cover']
    train['2']._storage['text'] = 'The laptop case was great and Cover was rubbish'
    answer = [[0], [1], [0,0]]
    if not lower:
        answer = [[0], [0], [0,0]]
    test = known_sentiment_known_target(test, train, lower)
    assert answer == get_error_counts(test, 'known_sentiment_known_target')
    # Where both have no targets
    empty_target = TargetText(text='something', text_id='1', targets=[], 
                                spans=[], target_sentiments=[])
    test = TargetTextCollection([empty_target])
    train = TargetTextCollection([empty_target])
    test = known_sentiment_known_target(test, train, lower)
    assert [[]] == get_error_counts(test, 'known_sentiment_known_target')
    # Where the targets are it should return an empty list
    no_targets = TargetText(text='something', text_id='1')
    test = TargetTextCollection([no_targets])
    test = known_sentiment_known_target(test, train, lower)
    assert [[]] == get_error_counts(test, 'known_sentiment_known_target')

@pytest.mark.parametrize("lower", (False, True))
def test_unknown_sentiment_known_target(lower: bool):
    # The train and test collection where there are no targets 
    # with the same sentiment
    pos, neg, neu = 'positive', 'negative', 'neutral'
    train_sentiments = [[pos], [neg], [neg, pos]]
    test_sentiments = [[neu], [neu], [neu, neu]]
    train = TargetTextCollection(target_text_examples(train_sentiments))
    test = TargetTextCollection(target_text_examples(test_sentiments))
    test = unknown_sentiment_known_target(test, train, lower)
    answer = [[1], [1], [1,1]]
    assert answer == get_error_counts(test, 'unknown_sentiment_known_target')
    # The train and test collection where one target contains two sentiment 
    # where one is the same but the other is not
    test_sentiments = [[neg], [pos], [neu, neu]]
    test = TargetTextCollection(target_text_examples(test_sentiments))
    test = unknown_sentiment_known_target(test, train, lower)
    answer = [[0], [0], [1,1]]
    assert answer == get_error_counts(test, 'unknown_sentiment_known_target')
    # all targets are highlighted
    train_sentiments = [[neg], [neu], [neu, pos]]
    train = TargetTextCollection(target_text_examples(train_sentiments))
    test = unknown_sentiment_known_target(test, train, lower)
    answer = [[0], [0], [0,0]]
    assert answer == get_error_counts(test, 'unknown_sentiment_known_target')
    # Case version
    train_sentiments = [[pos], [neg], [neg, pos]]
    test_sentiments = [[neu], [pos], [neu, neu]]
    train = TargetTextCollection(target_text_examples(train_sentiments))
    test = TargetTextCollection(target_text_examples(test_sentiments))
    test['2']._storage['targets'] = ['laptop case', 'Cover']
    test['2']._storage['text'] = 'The laptop case was great and Cover was rubbish'
    train['2']._storage['targets'] = ['laptop case', 'Cover']
    train['2']._storage['text'] = 'The laptop case was great and Cover was rubbish'
    answer = [[1], [0], [1,1]]
    if not lower:
        answer = [[1], [1], [1,1]]
    test = unknown_sentiment_known_target(test, train, lower)
    assert answer == get_error_counts(test, 'unknown_sentiment_known_target')
    # Where both have no targets
    empty_target = TargetText(text='something', text_id='1', targets=[], 
                                spans=[], target_sentiments=[])
    test = TargetTextCollection([empty_target])
    train = TargetTextCollection([empty_target])
    test = unknown_sentiment_known_target(test, train, lower)
    assert [[]] == get_error_counts(test, 'unknown_sentiment_known_target')
    # Where the targets are it should return an empty list
    no_targets = TargetText(text='something', text_id='1')
    test = TargetTextCollection([no_targets])
    test = unknown_sentiment_known_target(test, train, lower)
    assert [[]] == get_error_counts(test, 'unknown_sentiment_known_target')
        

@pytest.mark.parametrize("lower", (False, True))
def test_unknown_targets(lower: bool):
    # The train and test collection where there are no targets 
    # with the same sentiment
    pos, neg, neu = 'positive', 'negative', 'neutral'
    train_sentiments = [[pos], [neg], [neg, pos]]
    test_sentiments = [[neu], [neu], [neu, neu]]
    train = TargetTextCollection(target_text_examples(train_sentiments))
    test = TargetTextCollection(target_text_examples(test_sentiments))
    test = unknown_targets(test, train, lower)
    answer = [[0], [0], [0,0]]
    assert answer == get_error_counts(test, 'unknown_targets')
    # The train and test collection where one target contains two sentiment 
    # where one is the same but the other is not
    test_sentiments = [[pos], [neu], [pos, neu]]
    test = TargetTextCollection(target_text_examples(test_sentiments))
    test = unknown_targets(test, train, lower)
    answer = [[0], [0], [0,0]]
    assert answer == get_error_counts(test, 'unknown_targets')
    # The train and test collection where one target has the same sentiment
    train_sentiments = [[pos], [neg], [pos, pos]]
    train = TargetTextCollection(target_text_examples(train_sentiments))
    test = unknown_targets(test, train, lower)
    answer = [[0], [0], [0,0]]
    assert answer == get_error_counts(test, 'unknown_targets')
    # Where one of the targets has two different sentiments that are the same
    train_sentiments = [[pos], [neu], [pos, neg]]
    train = TargetTextCollection(target_text_examples(train_sentiments))
    test_sentiments = [[pos], [neu], [pos, neg]]
    test = TargetTextCollection(target_text_examples(test_sentiments))
    test = unknown_targets(test, train, lower)
    answer = [[0], [0], [0,0]]
    assert answer == get_error_counts(test, 'unknown_targets')
    # This tests both lower and not as well as the only time that it can 
    # be True as in the only time a target is in Test and not in Train.
    train_sentiments = [[pos], [neg], [neg, pos]]
    test_sentiments = [[pos], [neu], [pos, neu]]
    train = TargetTextCollection(target_text_examples(train_sentiments))
    test = TargetTextCollection(target_text_examples(test_sentiments))
    test['2']._storage['targets'] = ['laptop case', 'Cover']
    test['2']._storage['text'] = 'The laptop case was great and Cover was rubbish'
    answer = [[0], [0], [0,0]]
    if not lower:
        answer = [[0], [0], [0,1]]
    test = unknown_targets(test, train, lower)
    assert answer == get_error_counts(test, 'unknown_targets')
    # Where both have no targets
    empty_target = TargetText(text='something', text_id='1', targets=[], 
                              spans=[], target_sentiments=[])
    test = TargetTextCollection([empty_target])
    train = TargetTextCollection([empty_target])
    test = unknown_targets(test, train, lower)
    assert [[]] == get_error_counts(test, 'unknown_targets')
    # Where the targets are it should return an empty list
    no_targets = TargetText(text='something', text_id='1')
    test = TargetTextCollection([no_targets])
    test = unknown_targets(test, train, lower)
    assert [[]] == get_error_counts(test, 'unknown_targets')
        
def test_count_error_key_occurrence():
    # The zero case
    pos, neg, neu = 'positive', 'negative', 'neutral'
    train_sentiments = [[pos], [neg], [neg, pos]]
    test_sentiments = [[neu], [neu], [neu, neu]]
    train = TargetTextCollection(target_text_examples(train_sentiments))
    test = TargetTextCollection(target_text_examples(test_sentiments))
    test = same_one_sentiment(test, train, True)
    assert 0 == count_error_key_occurrence(test, 'same_one_sentiment')
    # One where that one comes from a sample that only has one target
    train_sentiments = [[pos], [neg], [pos, pos]]
    test_sentiments = [[neu], [neu], [pos, neu]]
    train = TargetTextCollection(target_text_examples(train_sentiments))
    train['2']._storage['targets'] = ['Laptop case', 'cover']
    test = TargetTextCollection(target_text_examples(test_sentiments))
    test['2']._storage['targets'] = ['Laptop case', 'cover']
    test['2']._storage['text'] = 'The Laptop case was great and cover was rubbish'
    test = same_one_sentiment(test, train, False)
    assert 1 == count_error_key_occurrence(test, 'same_one_sentiment')
    # Two case where the two come from two different samples
    pos, neg, neu = 'positive', 'negative', 'neutral'
    train_sentiments = [[pos], [neg], [pos, pos]]
    test_sentiments = [[pos], [neu], [pos, neu]]
    train = TargetTextCollection(target_text_examples(train_sentiments))
    test = TargetTextCollection(target_text_examples(test_sentiments))
    test = same_one_sentiment(test, train, True)
    assert 2 == count_error_key_occurrence(test, 'same_one_sentiment')
    # All case where all of the samples are errors
    pos, neg, neu = 'positive', 'negative', 'neutral'
    train_sentiments = [[neu], [neu], [neu, neu]]
    test_sentiments = [[neu], [neu], [neu, neu]]
    train = TargetTextCollection(target_text_examples(train_sentiments))
    test = TargetTextCollection(target_text_examples(test_sentiments))
    test = same_one_sentiment(test, train, True)
    assert 4 == count_error_key_occurrence(test, 'same_one_sentiment')
    # Case where there are no targets
    empty_target = TargetText(text='something', text_id='1', targets=[], 
                              spans=[], target_sentiments=[])
    train = TargetTextCollection([empty_target])
    test = TargetTextCollection([empty_target])
    test = same_one_sentiment(test, train, True)
    assert 0 == count_error_key_occurrence(test, 'same_one_sentiment')

def test_reduce_collection_by_key_occurrence():
    normal_associated_keys = ['targets', 'spans', 'target_sentiments']
    # The zero case
    pos, neg, neu = 'positive', 'negative', 'neutral'
    train_sentiments = [[pos], [neg], [neg, pos]]
    test_sentiments = [[neu], [neu], [neu, neu]]
    train = TargetTextCollection(target_text_examples(train_sentiments))
    test = TargetTextCollection(target_text_examples(test_sentiments))
    test = same_one_sentiment(test, train, True)
    zero_case =  reduce_collection_by_key_occurrence(test, 'same_one_sentiment', 
                                                     normal_associated_keys)
    assert 0 == len(zero_case)
    # Contains at least one target
    train_sentiments = [[pos], [neg], [pos, pos]]
    test_sentiments = [[neu], [neu], [pos, neu]]
    train = TargetTextCollection(target_text_examples(train_sentiments))
    train['2']._storage['targets'] = ['Laptop case', 'cover']
    test = TargetTextCollection(target_text_examples(test_sentiments))
    test['2']._storage['targets'] = ['Laptop case', 'cover']
    test['2']._storage['text'] = 'The Laptop case was great and cover was rubbish'
    test = same_one_sentiment(test, train, False)
    one_case = reduce_collection_by_key_occurrence(test, 'same_one_sentiment',
                                                   normal_associated_keys)
    assert 1 == len(one_case)
    assert ['Laptop case'] == one_case['2']['targets']
    assert [pos] == one_case['2']['target_sentiments']
    assert [Span(4,15)] == one_case['2']['spans']

    # Two case where the two come from two different samples
    train_sentiments = [[pos], [neg], [pos, pos]]
    test_sentiments = [[pos], [neu], [pos, neu]]
    train = TargetTextCollection(target_text_examples(train_sentiments))
    test = TargetTextCollection(target_text_examples(test_sentiments))
    test = same_one_sentiment(test, train, True)
    two_case = reduce_collection_by_key_occurrence(test, 'same_one_sentiment',
                                                   normal_associated_keys)
    assert 2 == len(two_case)
    assert 2 == two_case.number_targets()
    assert ['laptop case'] == two_case['0']['targets']
    # Test the case when the number of `error_key` is more than one i.e. the 
    # list case
    test = distinct_sentiment(test, True)
    three_case = reduce_collection_by_key_occurrence(test, ['same_one_sentiment', 
                                                            'distinct_sentiment_2'],
                                                     normal_associated_keys)
    assert 2 == len(three_case)
    assert set(['0', '2']) == set(three_case.keys())
    assert 3 == three_case.number_targets()
    assert ['laptop case'] == three_case['0']['targets']
    # The case where the error labels overlap
    three_case = reduce_collection_by_key_occurrence(test, ['same_one_sentiment', 
                                                            'distinct_sentiment_1'],
                                                     normal_associated_keys)
    assert 3 == len(three_case)
    assert 3 == three_case.number_targets()
    assert ['laptop case'] == three_case['0']['targets']
    # Ensure that the assertion statement will be raised if two error keys do 
    # not have the same number of values in them
    test['2']['distinct_sentiment_2'] = [1,1,1]
    with pytest.raises(AssertionError):
        reduce_collection_by_key_occurrence(test, ['same_one_sentiment', 
                                                   'distinct_sentiment_2'],
                                            normal_associated_keys)
    # All case where all of the samples are errors
    train_sentiments = [[neu], [neu], [neu, neu]]
    test_sentiments = [[neu], [neu], [neu, neu]]
    train = TargetTextCollection(target_text_examples(train_sentiments))
    test = TargetTextCollection(target_text_examples(test_sentiments))
    test = same_one_sentiment(test, train, True)
    four_case = reduce_collection_by_key_occurrence(test, 'same_one_sentiment',
                                                    normal_associated_keys)
    assert 3 == len(four_case)
    assert 4 == four_case.number_targets()
    assert ['laptop case', 'cover'] == four_case['2']['targets']
    assert [neu, neu] == four_case['2']['target_sentiments']
    assert [Span(4, 15), Span(30, 35)] == four_case['2']['spans']
    # Case where there are no targets
    empty_target = TargetText(text='something', text_id='1', targets=[], 
                              spans=[], target_sentiments=[])
    train = TargetTextCollection([empty_target])
    test = TargetTextCollection([empty_target])
    test = same_one_sentiment(test, train, True)
    zero_case = reduce_collection_by_key_occurrence(test, 'same_one_sentiment',
                                                    normal_associated_keys)
    assert 0 == len(zero_case)
    # Test that Key Errors
    train_sentiments = [[pos], [neg], [pos, pos]]
    test_sentiments = [[neu], [neu], [pos, neu]]
    train = TargetTextCollection(target_text_examples(train_sentiments))
    train['2']._storage['targets'] = ['Laptop case', 'cover']
    test = TargetTextCollection(target_text_examples(test_sentiments))
    test['2']._storage['targets'] = ['Laptop case', 'cover']
    test['2']._storage['another'] = [3, 4]
    test['another_id']._storage['another'] = [2]
    test['0']._storage['another'] = [1]
    test['2']._storage['text'] = 'The Laptop case was great and cover was rubbish'
    test = same_one_sentiment(test, train, False)
    with pytest.raises(KeyError):
        reduce_collection_by_key_occurrence(test, 'same_multi_sentiment',
                                            normal_associated_keys)
    with pytest.raises(KeyError):
        abnormal_keys = ['other_key', *normal_associated_keys]
        reduce_collection_by_key_occurrence(test, 'same_one_sentiment',
                                            abnormal_keys)
    # Ensure ValueError occurs when not all associated keys required are stated
    with pytest.raises(ValueError):
        reduce_collection_by_key_occurrence(test, 'same_one_sentiment',
                                            ['targets', 'spans'])
    with pytest.raises(ValueError):
        reduce_collection_by_key_occurrence(test, 'same_one_sentiment',
                                            ['targets', 'target_sentiments'])
    with pytest.raises(ValueError):
        reduce_collection_by_key_occurrence(test, 'same_one_sentiment',
                                            ['target_sentiments', 'spans'])
    # Ensure that when there are extra associated keys that the correct values 
    # are removed
    one_value = reduce_collection_by_key_occurrence(test, 'same_one_sentiment',
                                                    normal_associated_keys)
    assert [3, 4] == one_value['2']['another'] 
    extra_keys = ['another', *normal_associated_keys]
    one_value = reduce_collection_by_key_occurrence(test, 'same_one_sentiment',
                                                    extra_keys)
    assert [3] == one_value['2']['another']  


def larger_target_text_examples(target_sentiment_values: List[List[str]]
                                ) -> List[TargetText]:
    '''
    :param target_sentiment_values: A list of 4 lists where each inner list 
                                    represent the sentiment values for each 
                                    of the targets in that TargetText object
    '''
    text = 'The laptop case was great and cover was rubbish'
    text_ids = ['0', 'another_id', '2', '4']
    spans = [[Span(4, 15)], [Span(30, 35)], [Span(4, 15), Span(30, 35)],
             [Span(0, 3), Span(4, 15), Span(20, 25), Span(30, 35)]]
    targets = [['laptop case'], ['cover'], ['laptop case', 'cover'],
               ['The', 'laptop case', 'great', 'cover']]
    categories = [['LAPTOP#CASE'], ['LAPTOP'], ['LAPTOP#CASE', 'LAPTOP'],
                  ['LAPTOP#CASE', 'LAPTOP', 'LAPTOP#CASE', 'LAPTOP']]

    target_text_examples = []
    for i in range(4):
        example = TargetText(text, text_ids[i], targets=targets[i],
                                spans=spans[i], 
                                target_sentiments=target_sentiment_values[i],
                                categories=categories[i])
        target_text_examples.append(example)
    return target_text_examples

@pytest.mark.parametrize("separate_labels", (False, True))
def test_distinct_sentiment(separate_labels: bool):
    # Test the case with up to 2 distinct sentiments
    pos, neg, neu = 'positive', 'negative', 'neutral'
    train_sentiments = [[pos], [neg], [neg, pos], [neg, neu, neg, neg]]
    dataset = TargetTextCollection(larger_target_text_examples(train_sentiments))
    dataset = distinct_sentiment(dataset, separate_labels)
    if separate_labels:
        true_labels = {'distinct_sentiment_1': [[1], [1], [0,0], [0,0,0,0]],
                       'distinct_sentiment_2': [[0], [0], [1,1], [1,1,1,1]]}
        for i in range(1, 3):
            ds_key = f'distinct_sentiment_{i}'
            assert true_labels[ds_key] == get_error_counts(dataset, ds_key)
    else:
        answer = [[1], [1], [2,2], [2,2,2,2]]
        assert answer == get_error_counts(dataset, 'distinct_sentiment')
    # Test the case where they can have all three
    train_sentiments = [[pos], [neg], [neu, pos], [neg, neu, neg, pos]]
    dataset = TargetTextCollection(larger_target_text_examples(train_sentiments))
    dataset = distinct_sentiment(dataset, separate_labels)
    if separate_labels:
        true_labels = {'distinct_sentiment_1': [[1], [1], [0,0], [0,0,0,0]],
                       'distinct_sentiment_2': [[0], [0], [1,1], [0,0,0,0]],
                       'distinct_sentiment_3': [[0], [0], [0,0], [1,1,1,1]]}
        for i in range(1, 4):
            ds_key = f'distinct_sentiment_{i}'
            assert true_labels[ds_key] == get_error_counts(dataset, ds_key)
    else:
        answer = [[1], [1], [2,2], [3,3,3,3]]
        assert answer == get_error_counts(dataset, 'distinct_sentiment')
    # Case where there are no targets
    empty_target = TargetText(text='something', text_id='1', targets=[], 
                              spans=[], target_sentiments=[])
    dataset = TargetTextCollection([empty_target])
    if separate_labels:
        with pytest.raises(ValueError):
            distinct_sentiment(dataset, separate_labels)
    else:
        dataset = distinct_sentiment(dataset, separate_labels)
        assert [[]] == get_error_counts(dataset, 'distinct_sentiment')
    # Case where there are no targets where the targets are None values 
    # instead
    empty_target = TargetText(text='something', text_id='1', targets=None, 
                              spans=None, target_sentiments=None)
    dataset = TargetTextCollection([empty_target])
    if separate_labels:
        with pytest.raises(TypeError):
            distinct_sentiment(dataset, separate_labels)
    else:
        dataset = distinct_sentiment(dataset, separate_labels)
        assert [[]] == get_error_counts(dataset, 'distinct_sentiment')

@pytest.mark.parametrize("lower", (False, True))
@pytest.mark.parametrize("error_name", ('error', 'anything'))
def test_n_shot_targets(lower: bool, error_name: str):
    # this testing function does not care about sentiment just the targets
    pos, neg, neu = 'positive', 'negative', 'neutral'
    train_sentiments = [[pos], [neg], [neg, pos], [neg, neu, neg, neg]]
    target_text_obj = larger_target_text_examples(train_sentiments)
    all_targets = TargetTextCollection([target_text_obj[-1]])
    laptop_target = TargetTextCollection([target_text_obj[0]])
    cover_target = TargetTextCollection([target_text_obj[1]])
    laptop_2_cover_2 = TargetTextCollection([target_text_obj[0], target_text_obj[1], 
                                             target_text_obj[2]])
    # Case where zero shot target
    counts = get_error_counts(n_shot_targets(laptop_target, cover_target, 
                              lambda x: x==0, lower=lower, error_name=error_name), error_name)
    assert [[1]] == counts
    assert [[0]] != counts
    # Case zero shot target for one of the two targets
    counts = get_error_counts(n_shot_targets(laptop_2_cover_2, cover_target, 
                              lambda x: x==0, lower=lower, error_name=error_name), error_name)
    assert [[1],[0],[1,0]] == counts
    # Case where one shot target
    counts = get_error_counts(n_shot_targets(laptop_2_cover_2, cover_target, 
                              lambda x: x==1, lower=lower, error_name=error_name), error_name)
    assert [[0],[1],[0,1]] == counts
    # Case where we do >0
    counts = get_error_counts(n_shot_targets(laptop_2_cover_2, cover_target, 
                              lambda x: x>0, lower=lower, error_name=error_name), error_name)
    assert [[0],[1],[0,1]] == counts
    # Case where all are not equal to 0
    counts = get_error_counts(n_shot_targets(laptop_2_cover_2, laptop_2_cover_2, 
                              lambda x: x!=0, lower=lower, error_name=error_name), error_name)
    assert [[1],[1],[1,1]] == counts
    # Case where exists more than once in the train but only once in the test
    counts = get_error_counts(n_shot_targets(all_targets, laptop_2_cover_2, 
                              lambda x: x>1, lower=lower, error_name=error_name), error_name)
    assert [[0,1,0,1]] == counts
    counts = get_error_counts(n_shot_targets(all_targets, laptop_2_cover_2, 
                              lambda x: x==2, lower=lower, error_name=error_name), error_name)
    assert [[0,1,0,1]] == counts
    # Case of the rest being zero
    counts = get_error_counts(n_shot_targets(all_targets, laptop_2_cover_2, 
                              lambda x: x==0, lower=lower, error_name=error_name), error_name)
    assert [[1,0,1,0]] == counts

    # Test the case for lower
    target_text_obj[-1]._storage['targets'] = ['The', 'Laptop case', 'great', 'Cover']
    target_text_obj[-1]._storage['text'] = 'The Laptop case was great and Cover was rubbish'
    all_targets = TargetTextCollection([target_text_obj[-1]])
    counts = get_error_counts(n_shot_targets(all_targets, laptop_2_cover_2, 
                                lambda x: x==0, lower=lower, error_name=error_name), error_name)
    if lower:
        assert [[1,0,1,0]] == counts
    else:
        assert [[1,1,1,1]] == counts

@pytest.mark.parametrize("return_n_values", (True, False))
def test_n_shot_subsets(return_n_values: bool):
    # this testing function does not care about sentiment just the targets
    train_fp = Path(DATA_DIR, 'train.json').resolve()
    test_fp = Path(DATA_DIR, 'test.json').resolve()
    train_collection = TargetTextCollection.load_json(train_fp)
    test_collection = TargetTextCollection.load_json(test_fp)

    if return_n_values:
        collection, n_values = n_shot_subsets(test_collection, train_collection, 
                                            True, return_n_values)
    else:
        collection = n_shot_subsets(test_collection, train_collection, 
                                    True, return_n_values)
    zero_values = get_error_counts(collection, 'zero-shot')
    correct_zero = [[0], [0], [1,1], [0], [1,1,1,1,1], [1,1,1,1], [0,1,1,1,0,1,1]]
    assert correct_zero == zero_values

    low_values = get_error_counts(collection, 'low-shot')
    correct_low = [[1], [1], [0,0], [0], [0,0,0,0,0], [0,0,0,0], [0,0,0,0,0,0,0]]
    assert correct_low == low_values

    med_values = get_error_counts(collection, 'med-shot')
    correct_med = [[0], [0], [0,0], [1], [0,0,0,0,0], [0,0,0,0], [1,0,0,0,0,0,0]]
    assert correct_med == med_values

    high_values = get_error_counts(collection, 'high-shot')
    correct_high = [[0], [0], [0,0], [0], [0,0,0,0,0], [0,0,0,0], [0,0,0,0,1,0,0]]
    assert correct_high == high_values

    if return_n_values:
        correct_n_values = [(0,0), (1,1), (2,2), (3,3)]
        assert correct_n_values == n_values

    assert 2 == count_error_key_occurrence(collection, 'low-shot')
    assert 2 == count_error_key_occurrence(collection, 'med-shot')
    assert 1 == count_error_key_occurrence(collection, 'high-shot')

@pytest.mark.parametrize("return_n_values", (True, False))
def test_num_targets_subset(return_n_values: bool):
    # this testing function does not care about sentiment just the targets
    test_fp = Path(DATA_DIR, 'test.json').resolve()
    test_collection = TargetTextCollection.load_json(test_fp)

    if return_n_values:
        collection, n_values = num_targets_subset(test_collection, return_n_values)
    else:
        collection = num_targets_subset(test_collection, return_n_values)
    one_values = get_error_counts(collection, '1-target')
    correct_one = [[1], [1], [0,0], [1], [0,0,0,0,0], [0,0,0,0], [0,0,0,0,0,0,0]]
    assert correct_one == one_values

    low_values = get_error_counts(collection, 'low-targets')
    correct_low = [[0], [0], [1,1], [0], [0,0,0,0,0], [1,1,1,1], [0,0,0,0,0,0,0]]
    assert correct_low == low_values

    med_values = get_error_counts(collection, 'med-targets')
    correct_med = [[0], [0], [0,0], [0], [1,1,1,1,1], [0,0,0,0], [0,0,0,0,0,0,0]]
    assert correct_med == med_values

    high_values = get_error_counts(collection, 'high-targets')
    correct_high = [[0], [0], [0,0], [0], [0,0,0,0,0], [0,0,0,0], [1,1,1,1,1,1,1]]
    assert correct_high == high_values
    
    if return_n_values:
        correct_n_values = [(1,1), (2,4), (5,5), (7,7)]
        assert correct_n_values == n_values

    assert 6 == count_error_key_occurrence(collection, 'low-targets')
    assert 5 == count_error_key_occurrence(collection, 'med-targets')
    assert 7 == count_error_key_occurrence(collection, 'high-targets')
    assert 3 == count_error_key_occurrence(collection, '1-target')

def test_tssr_raw():
    # First see if it works when there are no targets
    test_collection = TargetTextCollection()
    collection, tssr_values = tssr_raw(test_collection)
    assert {} == tssr_values
    # Test on a collection with various amounts of targets
    test_fp = Path(DATA_DIR, 'test.json').resolve()
    test_collection = TargetTextCollection.load_json(test_fp)

    collection, tssr_values = tssr_raw(test_collection)
    # Test that the TSSR values are correct
    true_tssr_values = {1.0: 5, 0.6: 3, 0.2: 2, 0.75: 3, 0.25: 1,
                        0.57: 4, 0.14: 1, 0.29: 2}
    for true_tssr_value, num in true_tssr_values.items():
        assert num == tssr_values[str(true_tssr_value)]
    assert len(true_tssr_values) == len(tssr_values)
    
    # 1.0 values
    one_values = get_error_counts(collection, '1.0')
    correct_one = [[1], [1], [1,1], [1], [0,0,0,0,0], [0,0,0,0], [0,0,0,0,0,0,0]]
    assert correct_one == one_values
    # 0.6 values
    one_values = get_error_counts(collection, '0.6')
    correct_one = [[0], [0], [0,0], [0], [0,1,1,0,1], [0,0,0,0], [0,0,0,0,0,0,0]]
    assert correct_one == one_values
    # 0.2 values
    one_values = get_error_counts(collection, '0.2')
    correct_one = [[0], [0], [0,0], [0], [1,0,0,1,0], [0,0,0,0], [0,0,0,0,0,0,0]]
    assert correct_one == one_values
    # 0.75 values
    one_values = get_error_counts(collection, '0.75')
    correct_one = [[0], [0], [0,0], [0], [0,0,0,0,0], [1,0,1,1], [0,0,0,0,0,0,0]]
    assert correct_one == one_values
    # 0.25 value
    one_values = get_error_counts(collection, '0.25')
    correct_one = [[0], [0], [0,0], [0], [0,0,0,0,0], [0,1,0,0], [0,0,0,0,0,0,0]]
    assert correct_one == one_values
    # 0.57 value
    one_values = get_error_counts(collection, '0.57')
    correct_one = [[0], [0], [0,0], [0], [0,0,0,0,0], [0,0,0,0], [1,1,0,1,1,0,0]]
    assert correct_one == one_values
    # 0.14 value
    one_values = get_error_counts(collection, '0.14')
    correct_one = [[0], [0], [0,0], [0], [0,0,0,0,0], [0,0,0,0], [0,0,1,0,0,0,0]]
    assert correct_one == one_values
    # 0.29 value
    one_values = get_error_counts(collection, '0.29')
    correct_one = [[0], [0], [0,0], [0], [0,0,0,0,0], [0,0,0,0], [0,0,0,0,0,1,1]]
    assert correct_one == one_values

@pytest.mark.parametrize("return_tssr_boundaries", (True, False))
def test_tssr_subset(return_tssr_boundaries: bool):
    # Test that it raises a ValueError when the TargetTextCollection is empty
    test_collection = TargetTextCollection()
    with pytest.raises(NoSamplesError):
        tssr_subset(test_collection, return_tssr_boundaries)
    # Test on a collection with various amounts of targets
    test_fp = Path(DATA_DIR, 'test.json').resolve()
    test_collection = TargetTextCollection.load_json(test_fp)

    
    if return_tssr_boundaries:
        collection, tssr_bound = tssr_subset(test_collection, return_tssr_boundaries)
        true_tssr_bound = [(1.0, 1.0), (0.75, 0.6), (0.57, 0.14)]
        assert len(true_tssr_bound) == len(tssr_bound)
        assert true_tssr_bound == tssr_bound
    else:   
        collection = tssr_subset(test_collection, return_tssr_boundaries)

    one_values = get_error_counts(collection, '1-TSSR')
    correct_one = [[1], [1], [0,0], [1], [0,0,0,0,0], [0,0,0,0], [0,0,0,0,0,0,0]]
    assert correct_one == one_values

    one_values = get_error_counts(collection, '1-multi-TSSR')
    correct_one = [[0], [0], [1,1], [0], [0,0,0,0,0], [0,0,0,0], [0,0,0,0,0,0,0]]
    assert correct_one == one_values

    one_values = get_error_counts(collection, 'high-TSSR')
    correct_one = [[0], [0], [0,0], [0], [0,1,1,0,1], [1,0,1,1], [0,0,0,0,0,0,0]]
    assert correct_one == one_values

    one_values = get_error_counts(collection, 'low-TSSR')
    correct_one = [[0], [0], [0,0], [0], [1,0,0,1,0], [0,1,0,0], [1,1,1,1,1,1,1]]
    assert correct_one == one_values

def example_collections_runs_first() -> TargetTextCollection:
    text = 'The laptop case was great and cover was rubbish'
    text_ids = ['0', 'another_id', '2']
    spans = [[Span(4, 15)], [Span(4, 15), Span(30, 35)], 
             [Span(4, 15), Span(30, 35), Span(40, 47)]]
    targets = [['laptop case'], ['laptop case', 'cover'], 
               ['laptop case', 'cover', 'rubbish']]
    target_sentiments = [[0], [1, 0], [2, 1, 0]]
    predicted_sentiments = [[[1], [2], [0], [1]], 
                            [[1, 0], [2, 0], [1, 0], [2, 1]], 
                            [[1, 0, 2], [1, 2, 0], [0, 1, 0], [1, 2, 1]]]
    target_text_examples = []
    for i in range(3):
        example = TargetText(text, text_ids[i], targets=targets[i],
                             spans=spans[i], 
                             target_sentiments=target_sentiments[i],
                             predicted_sentiments=predicted_sentiments[i])
        target_text_examples.append(example)
    return TargetTextCollection(target_text_examples)


def test_swap_list_dimensions():
    normal_case_collection = example_collections_runs_first()
    test_case_collection = swap_list_dimensions(normal_case_collection, 
                                                'predicted_sentiments')
    zero_preds = test_case_collection['0']['predicted_sentiments']
    assert 1 == len(zero_preds)
    assert 4 == len(zero_preds[0])
    assert [[1,2,0,1]] == zero_preds

    another_preds = test_case_collection['another_id']['predicted_sentiments']
    assert 2 == len(another_preds)
    assert 4 == len(another_preds[0])
    assert [[1,2,1,2], [0,0,0,1]] == another_preds

    two_preds = test_case_collection['2']['predicted_sentiments']
    assert 3 == len(two_preds)
    assert 4 == len(two_preds[0])
    assert [[1,1,0,1], [0,2,1,2], [2,0,0,1]] == two_preds
    
    # Checks that the copy works and it does not modify the original Collection
    zero_normal_preds = normal_case_collection['0']['predicted_sentiments']
    assert 4 == len(zero_normal_preds)
    assert 1 == len(zero_normal_preds[0])
    assert [[1],[2],[0],[1]] == zero_normal_preds

    # When you run the swap list dimensions twice it should revet back to 
    # the original version
    invert_test = swap_list_dimensions(test_case_collection, 
                                       'predicted_sentiments')
    zero_preds = invert_test['0']['predicted_sentiments']
    assert 4 == len(zero_preds)
    assert 1 == len(zero_preds[0])
    assert [[1],[2],[0],[1]] == zero_preds

    another_preds = invert_test['another_id']['predicted_sentiments']
    assert 4 == len(another_preds)
    assert 2 == len(another_preds[0])
    assert [[1, 0], [2, 0], [1, 0],[2, 1]] == another_preds

    two_preds = invert_test['2']['predicted_sentiments']
    assert 4 == len(two_preds)
    assert 3 == len(two_preds[0])
    assert [[1, 0, 2], [1, 2, 0], [0, 1, 0], [1, 2, 1]] == two_preds

    # Test Key Raises
    with pytest.raises(KeyError):
        del test_case_collection['0']['predicted_sentiments']
        swap_list_dimensions(test_case_collection, 'predicted_sentiments')

def test_error_analysis_wrapper():
    def test_tssr(tssr_func):
        test_fp = Path(DATA_DIR, 'test.json').resolve()
        test_collection = TargetTextCollection.load_json(test_fp)
        collection = tssr_func(None, test_collection, False)
        one_values = get_error_counts(collection, '1-TSSR')
        correct_one = [[1], [1], [0,0], [1], [0,0,0,0,0], [0,0,0,0], [0,0,0,0,0,0,0]]
        assert correct_one == one_values

        one_values = get_error_counts(collection, '1-multi-TSSR')
        correct_one = [[0], [0], [1,1], [0], [0,0,0,0,0], [0,0,0,0], [0,0,0,0,0,0,0]]
        assert correct_one == one_values

        one_values = get_error_counts(collection, 'high-TSSR')
        correct_one = [[0], [0], [0,0], [0], [0,1,1,0,1], [1,0,1,1], [0,0,0,0,0,0,0]]
        assert correct_one == one_values

        one_values = get_error_counts(collection, 'low-TSSR')
        correct_one = [[0], [0], [0,0], [0], [1,0,0,1,0], [0,1,0,0], [1,1,1,1,1,1,1]]
        assert correct_one == one_values
    
    def test_nt(nt_func):
        test_fp = Path(DATA_DIR, 'test.json').resolve()
        test_collection = TargetTextCollection.load_json(test_fp)
        collection = nt_func(None, test_collection, False)
        one_values = get_error_counts(collection, '1-target')
        correct_one = [[1], [1], [0,0], [1], [0,0,0,0,0], [0,0,0,0], [0,0,0,0,0,0,0]]
        assert correct_one == one_values

        low_values = get_error_counts(collection, 'low-targets')
        correct_low = [[0], [0], [1,1], [0], [0,0,0,0,0], [1,1,1,1], [0,0,0,0,0,0,0]]
        assert correct_low == low_values

        med_values = get_error_counts(collection, 'med-targets')
        correct_med = [[0], [0], [0,0], [0], [1,1,1,1,1], [0,0,0,0], [0,0,0,0,0,0,0]]
        assert correct_med == med_values

        high_values = get_error_counts(collection, 'high-targets')
        correct_high = [[0], [0], [0,0], [0], [0,0,0,0,0], [0,0,0,0], [1,1,1,1,1,1,1]]
        assert correct_high == high_values

    def test_ds(ds_func):
        pos, neg, neu = 'positive', 'negative', 'neutral'
        train_sentiments = [[pos], [neg], [neu, pos], [neg, neu, neg, pos]]
        dataset = TargetTextCollection(larger_target_text_examples(train_sentiments))
        dataset = ds_func(None, dataset, False)
        true_labels = {'distinct_sentiment_1': [[1], [1], [0,0], [0,0,0,0]],
                       'distinct_sentiment_2': [[0], [0], [1,1], [0,0,0,0]],
                       'distinct_sentiment_3': [[0], [0], [0,0], [1,1,1,1]]}
        for i in range(1, 4):
            ds_key = f'distinct_sentiment_{i}'
            assert true_labels[ds_key] == get_error_counts(dataset, ds_key)

    def test_n_shot(n_shot_func):
        train_fp = Path(DATA_DIR, 'train.json').resolve()
        test_fp = Path(DATA_DIR, 'test.json').resolve()
        train_collection = TargetTextCollection.load_json(train_fp)
        test_collection = TargetTextCollection.load_json(test_fp)
        collection = n_shot_func(train_collection, test_collection, True)
        zero_values = get_error_counts(collection, 'zero-shot')
        correct_zero = [[0], [0], [1,1], [0], [1,1,1,1,1], [1,1,1,1], [0,1,1,1,0,1,1]]
        assert correct_zero == zero_values

        low_values = get_error_counts(collection, 'low-shot')
        correct_low = [[1], [1], [0,0], [0], [0,0,0,0,0], [0,0,0,0], [0,0,0,0,0,0,0]]
        assert correct_low == low_values

        med_values = get_error_counts(collection, 'med-shot')
        correct_med = [[0], [0], [0,0], [1], [0,0,0,0,0], [0,0,0,0], [1,0,0,0,0,0,0]]
        assert correct_med == med_values

        high_values = get_error_counts(collection, 'high-shot')
        correct_high = [[0], [0], [0,0], [0], [0,0,0,0,0], [0,0,0,0], [0,0,0,0,1,0,0]]
        assert correct_high == high_values

    def test_tsr(tsr_func):
        pos, neg, neu = 'positive', 'negative', 'neutral'
        train_sentiments = [[pos], [neg], [neg, pos]]
        test_sentiments = [[pos], [neu], [pos, neu]]
        train = TargetTextCollection(target_text_examples(train_sentiments))
        test = TargetTextCollection(target_text_examples(test_sentiments))
        test['2']._storage['targets'] = ['laptop case', 'Cover']
        test['2']._storage['text'] = 'The laptop case was great and Cover was rubbish'

        unknown_answer = [[0], [0], [0,1]]
        known_target_answer = [[1], [0], [1, 0]]
        unknown_sentiment_answer = [[0], [1], [0, 0]]
        test = tsr_func(train, test, False)
        assert unknown_answer == get_error_counts(test, 'unknown_targets')
        assert known_target_answer == get_error_counts(test, 'known_sentiment_known_target')
        assert unknown_sentiment_answer == get_error_counts(test, 'unknown_sentiment_known_target')



    for error_split_name in ERROR_SPLIT_SUBSET_NAMES.keys():
        error_func = error_analysis_wrapper(error_split_name)
        if error_split_name == 'TSSR':
            test_tssr(error_func)
        elif error_split_name == 'NT':
            test_nt(error_func)
        elif error_split_name == 'DS':
            test_ds(error_func)
        elif error_split_name == 'n-shot':
            test_n_shot(error_func)
        elif error_split_name == 'TSR':
            test_tsr(error_func)
    with pytest.raises(ValueError):
        error_analysis_wrapper('error')


@pytest.mark.parametrize("metric_func_name", ('Accuracy', 'Score'))
def test_subset_metrics(metric_func_name: str):
    test_fp = Path(DATA_DIR, 'test.json').resolve()
    train_fp = Path(DATA_DIR, 'train.json').resolve()
    test_collection = TargetTextCollection.load_json(test_fp)
    train_collection = TargetTextCollection.load_json(train_fp)
    error_func = error_analysis_wrapper('n-shot')
    error_func(train_collection, test_collection, True)

    # Average version
    average_kwargs = {'average': True, 'array_scores': False, 
                      'assert_number_labels': 2, 
                      'true_sentiment_key': 'target_sentiments',
                      'predicted_sentiment_key': 'pred_sentiments'}
    metric_dict = subset_metrics(test_collection, 'low-shot', [accuracy], 
                                 [metric_func_name], average_kwargs)
    assert 0.25 == metric_dict[metric_func_name]
    # Array scores version
    array_scores_kwargs = average_kwargs
    array_scores_kwargs['average'] = False
    array_scores_kwargs['array_scores'] = True
    metric_dict = subset_metrics(test_collection, 'low-shot', [accuracy], 
                                 [metric_func_name], average_kwargs)
    assert [0.5, 0.0] == metric_dict[metric_func_name]

    # Test the case for multiple metrics
    metric_dict = subset_metrics(test_collection, 'low-shot', [accuracy, macro_f1], 
                                 [metric_func_name, 'macro'], average_kwargs)
    assert [0.5, 0.0] == metric_dict[metric_func_name]
    assert [1/3, 0.0] == metric_dict['macro']

    # Test the dataset size test.
    metric_dict = subset_metrics(test_collection, 'low-shot', [accuracy, macro_f1], 
                                 [metric_func_name, 'macro'], average_kwargs,
                                 include_dataset_size=True)
    assert [0.5, 0.0] == metric_dict[metric_func_name]
    assert [1/3, 0.0] == metric_dict['macro']
    assert test_collection.number_targets() != metric_dict['dataset size']
    assert 2 == metric_dict['dataset size']

@pytest.mark.parametrize("table_format_return", (False, True))
@pytest.mark.parametrize("include_dataset_size", (False, True))
def test__error_split_df(include_dataset_size: bool, table_format_return: bool):
    test_fp = Path(DATA_DIR, 'test.json').resolve()
    train_fp = Path(DATA_DIR, 'train.json').resolve()
    test_collection = TargetTextCollection.load_json(test_fp)
    train_collection = TargetTextCollection.load_json(train_fp)
    for error_func_name in ['n-shot', 'NT']:
        error_func = error_analysis_wrapper(f'{error_func_name}')
        test_collection = error_func(train_collection, test_collection, True)
    error_split_dict = {'n-shot': ['low-shot', 'med-shot'], 
                        'NT': ['low-targets', 'med-targets']}
    test_df = _error_split_df(test_collection, ['pred_sentiments'], 
                              'target_sentiments', error_split_dict, 
                              accuracy, None, 
                              include_dataset_size=include_dataset_size,
                              table_format_return=table_format_return)
    low_shot_score = [0.5, 0.0]
    med_shot_score = [1.0, 1.0]
    low_targets_score = [2/3, 2/3]
    med_targets_score = [0.6, 0.8]
    name_scores = {'low-shot': low_shot_score, 'med-shot': med_shot_score,
                   'low-targets': low_targets_score, 
                   'med-targets': med_targets_score}
    name_size = {'low-shot': [2, 2], 'med-shot': [2, 2], 'low-targets': [6, 6], 
                 'med-targets': [5, 5]}
    if table_format_return:
        if include_dataset_size:
            temp_name_scores = {}
            for name, score in name_scores.items():
                temp_name_scores[('Metric', f'{name}')] = score
                temp_name_scores[('Dataset Size', f'{name}')] = name_size[name]
            name_scores = temp_name_scores

        index_list = [('pred_sentiments', 0), ('pred_sentiments', 1)]
        assert index_list == test_df.index.tolist()
        column_list = list(name_scores.keys())
        for column_name in column_list:
            assert column_name in list(test_df.columns)
        assert len(column_list) == len(list(test_df.columns))

        for column_name, column_score in name_scores.items():
            assert column_score == test_df[column_name].tolist(), column_name
    else:
        assert list(range(8)) == test_df.index.tolist()
        column_list = ['prediction key', 'run number', 'subset names', 'Metric']
        if include_dataset_size:
            column_list.append('Dataset Size')
        assert set(column_list) == set(test_df.columns.tolist())
        for name, score in name_scores.items():
            name_df = test_df[test_df['subset names']==name]
            assert score == name_df['Metric'].tolist()
            if include_dataset_size:
                assert name_size[name] == name_df['Dataset Size'].tolist()

    test_df = _error_split_df(test_collection, ['pred_sentiments'], 
                              'target_sentiments', error_split_dict, 
                              accuracy, None, 
                              collection_subsetting=[['zero-shot', 'low-shot']],
                              include_dataset_size=include_dataset_size,
                              table_format_return=table_format_return)
    low_shot_score = [0.5, 0.0]
    med_shot_score = [0.0, 0.0]
    low_targets_score = [2/3, 2/3]
    med_targets_score = [0.6, 0.8]
    name_scores = {'low-shot': low_shot_score, 'med-shot': med_shot_score,
                   'low-targets': low_targets_score, 
                   'med-targets': med_targets_score}
    if table_format_return:
        name_size['med-shot'] = [0, 0]
        if include_dataset_size:
            temp_name_scores = {}
            for name, score in name_scores.items():
                temp_name_scores[('Metric', f'{name}')] = score
                temp_name_scores[('Dataset Size', f'{name}')] = name_size[name]
            name_scores = temp_name_scores
        index_list = [('pred_sentiments', 0), ('pred_sentiments', 1)]
        assert index_list == test_df.index.tolist()
        column_list = list(name_scores.keys())
        for column_name in column_list:
            assert column_name in list(test_df.columns)
        assert len(column_list) == len(list(test_df.columns))

        for column_name, column_score in name_scores.items():
            column_values = test_df[column_name].tolist()
            assert column_score == column_values, column_name
    else:
        name_size['med-shot'] = [0]
        name_scores['med-shot'] = [0]
        assert list(range(7)) == test_df.index.tolist()
        column_list = ['prediction key', 'run number', 'subset names', 'Metric']
        if include_dataset_size:
            column_list.append('Dataset Size')
        assert set(column_list) == set(test_df.columns.tolist())
        for name, score in name_scores.items():
            name_df = test_df[test_df['subset names']==name]
            assert score == name_df['Metric'].tolist()
            if include_dataset_size:
                assert name_size[name] == name_df['Dataset Size'].tolist()

def test_error_split_df():
    test_fp = Path(DATA_DIR, 'test.json').resolve()
    train_fp = Path(DATA_DIR, 'train.json').resolve()
    test_collection = TargetTextCollection.load_json(test_fp)
    train_collection = TargetTextCollection.load_json(train_fp)
    error_split_dict = {'n-shot': ['low-shot', 'med-shot'], 
                        'NT': ['low-targets', 'med-targets']}
    test_df = error_split_df(train_collection, test_collection, 
                             ['pred_sentiments'], 
                             'target_sentiments', error_split_dict, 
                              accuracy, None)
    low_shot_score = [0.5, 0.0]
    med_shot_score = [1.0, 1.0]
    low_targets_score = [2/3, 2/3]
    med_targets_score = [0.6, 0.8]
    name_scores = {'low-shot': low_shot_score, 'med-shot': med_shot_score,
                   'low-targets': low_targets_score, 
                   'med-targets': med_targets_score}

    index_list = [('pred_sentiments', 0), ('pred_sentiments', 1)]
    assert index_list == test_df.index.tolist()
    column_list = list(name_scores.keys())
    for column_name in column_list:
        assert column_name in list(test_df.columns)
    assert len(column_list) == len(list(test_df.columns))

    for column_name, column_score in name_scores.items():
        assert column_score == test_df[column_name].tolist(), column_name

def test_subset_name_to_error_split():
    assert 'DS' == subset_name_to_error_split('distinct_sentiment_1')
    assert 'DS' == subset_name_to_error_split('distinct_sentiment_2')
    assert 'DS' == subset_name_to_error_split('distinct_sentiment_3')
    assert 'NT' == subset_name_to_error_split('1-target')
    assert 'NT' == subset_name_to_error_split('low-targets')
    assert 'NT' == subset_name_to_error_split('med-targets')
    assert 'NT' == subset_name_to_error_split('high-targets')
    assert 'TSSR' == subset_name_to_error_split('1-TSSR')
    assert 'TSSR' == subset_name_to_error_split('1-multi-TSSR')
    assert 'TSSR' == subset_name_to_error_split('low-TSSR')
    assert 'TSSR' == subset_name_to_error_split('high-TSSR')
    assert 'TSR' == subset_name_to_error_split('unknown_sentiment_known_target')
    assert 'TSR' == subset_name_to_error_split('unknown_targets')
    assert 'TSR' == subset_name_to_error_split('known_sentiment_known_target')
    assert 'n-shot' == subset_name_to_error_split('zero-shot')
    assert 'n-shot' == subset_name_to_error_split('low-shot')
    assert 'n-shot' == subset_name_to_error_split('med-shot')
    assert 'n-shot' == subset_name_to_error_split('high-shot')