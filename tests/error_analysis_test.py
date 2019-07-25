from typing import List

import pytest

from target_extraction.data_types_util import Span
from target_extraction.data_types import TargetTextCollection, TargetText
from target_extraction.error_analysis import same_one_sentiment
from target_extraction.error_analysis import count_error_key_occurence


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
def test_target_sentiments(lower: bool):
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
                              spans=[])
    test = TargetTextCollection([empty_target])
    train = TargetTextCollection([empty_target])
    test = same_one_sentiment(test, train, lower)
    assert [[]] == get_error_counts(test, 'same_one_sentiment')
    # Where the targets are None it should raise a ValueError
    no_targets = TargetText(text='something', text_id='1')
    test = TargetTextCollection([no_targets])
    with pytest.raises(ValueError):
        same_one_sentiment(test, train, lower)

def test_count_error_key_occurence():
    # The zero case
    pos, neg, neu = 'positive', 'negative', 'neutral'
    train_sentiments = [[pos], [neg], [neg, pos]]
    test_sentiments = [[neu], [neu], [neu, neu]]
    train = TargetTextCollection(target_text_examples(train_sentiments))
    test = TargetTextCollection(target_text_examples(test_sentiments))
    test = same_one_sentiment(test, train, True)
    assert 0 == count_error_key_occurence(test, 'same_one_sentiment')
    # One where that one comes from a sample that only has one target
    train_sentiments = [[pos], [neg], [pos, pos]]
    test_sentiments = [[neu], [neu], [pos, neu]]
    train = TargetTextCollection(target_text_examples(train_sentiments))
    train['2']._storage['targets'] = ['Laptop case', 'cover']
    test = TargetTextCollection(target_text_examples(test_sentiments))
    test['2']._storage['targets'] = ['Laptop case', 'cover']
    test['2']._storage['text'] = 'The Laptop case was great and cover was rubbish'
    test = same_one_sentiment(test, train, False)
    assert 1 == count_error_key_occurence(test, 'same_one_sentiment')
    # Two case where the two come from two different samples
    pos, neg, neu = 'positive', 'negative', 'neutral'
    train_sentiments = [[pos], [neg], [pos, pos]]
    test_sentiments = [[pos], [neu], [pos, neu]]
    train = TargetTextCollection(target_text_examples(train_sentiments))
    test = TargetTextCollection(target_text_examples(test_sentiments))
    test = same_one_sentiment(test, train, True)
    assert 2 == count_error_key_occurence(test, 'same_one_sentiment')
    # All case where all of the samples are errors
    pos, neg, neu = 'positive', 'negative', 'neutral'
    train_sentiments = [[neu], [neu], [neu, neu]]
    test_sentiments = [[neu], [neu], [neu, neu]]
    train = TargetTextCollection(target_text_examples(train_sentiments))
    test = TargetTextCollection(target_text_examples(test_sentiments))
    test = same_one_sentiment(test, train, True)
    assert 4 == count_error_key_occurence(test, 'same_one_sentiment')
    # Case where there are no targets
    empty_target = TargetText(text='something', text_id='1', targets=[], 
                              spans=[])
    train = TargetTextCollection([empty_target])
    test = TargetTextCollection([empty_target])
    test = same_one_sentiment(test, train, True)
    assert 0 == count_error_key_occurence(test, 'same_one_sentiment')