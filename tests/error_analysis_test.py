from typing import List

import pytest

from target_extraction.data_types_util import Span
from target_extraction.data_types import TargetTextCollection, TargetText
from target_extraction.error_analysis import same_one_sentiment
from target_extraction.error_analysis import same_multi_sentiment
from target_extraction.error_analysis import similar_sentiment
from target_extraction.error_analysis import different_sentiment
from target_extraction.error_analysis import unknown_targets
from target_extraction.error_analysis import known_sentiment_known_target
from target_extraction.error_analysis import unknown_sentiment_known_target
from target_extraction.error_analysis import distinct_sentiment
from target_extraction.error_analysis import count_error_key_occurrence
from target_extraction.error_analysis import n_shot_targets


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
    
def test_distinct_sentiment():
    # Test the case with up to 2 distinct sentiments
    pos, neg, neu = 'positive', 'negative', 'neutral'
    train_sentiments = [[pos], [neg], [neg, pos], [neg, neu, neg, neg]]
    dataset = TargetTextCollection(larger_target_text_examples(train_sentiments))
    dataset = distinct_sentiment(dataset)
    answer = [[1], [1], [2,2], [2,2,2,2]]
    assert answer == get_error_counts(dataset, 'distinct_sentiment')
    # Test the case where they can have all three
    train_sentiments = [[pos], [neg], [neu, pos], [neg, neu, neg, pos]]
    dataset = TargetTextCollection(larger_target_text_examples(train_sentiments))
    dataset = distinct_sentiment(dataset)
    answer = [[1], [1], [2,2], [3,3,3,3]]
    assert answer == get_error_counts(dataset, 'distinct_sentiment')
    # Case where there are no targets
    empty_target = TargetText(text='something', text_id='1', targets=[], 
                              spans=[], target_sentiments=[])
    dataset = TargetTextCollection([empty_target])
    dataset = distinct_sentiment(dataset)
    assert [[]] == get_error_counts(dataset, 'distinct_sentiment')
    # Case where there are no targets where the targets are None values 
    # instead
    empty_target = TargetText(text='something', text_id='1', targets=None, 
                              spans=None, target_sentiments=None)
    dataset = TargetTextCollection([empty_target])
    dataset = distinct_sentiment(dataset)
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