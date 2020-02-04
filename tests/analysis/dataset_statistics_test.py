from pathlib import Path
import math
import re

import pytest

from target_extraction.data_types import TargetTextCollection, TargetText
from target_extraction.data_types_util import Span
from target_extraction.analysis.dataset_statistics import get_sentiment_counts
from target_extraction.analysis.dataset_statistics import average_target_per_sentences
from target_extraction.analysis.dataset_statistics import dataset_target_sentiment_statistics
from target_extraction.analysis.dataset_statistics import tokens_per_target
from target_extraction.analysis.dataset_statistics import dataset_target_extraction_statistics
from target_extraction.tokenizers import whitespace

DATA_DIR = Path(__file__, '..', '..', 'data', 'analysis', 'sentiment_error_analysis').resolve()
TRAIN_COLLECTION =  TargetTextCollection.load_json(Path(DATA_DIR, 'train_with_blank.json'))
SENTIMENT_KEY = 'target_sentiments'

["neutral", "neutral", "neutral", "neutral", "negative"]
["negative", "neutral", "neutral", "neutral"]
["neutral", "negative", "negative", "negative"]
["neutral", "positive", "neutral"]
def test_get_sentiment_counts():
    num_pos = 2
    num_neu = 12
    num_neg = 5
    total = 19.0
    true_sentiment_counts = dict([('positive', num_pos), ('neutral', num_neu), 
                                  ('negative', num_neg)])
    sentiment_counts = get_sentiment_counts(TRAIN_COLLECTION, normalised=False, 
                                            sentiment_key=SENTIMENT_KEY)
    assert len(true_sentiment_counts) == len(sentiment_counts)
    for sentiment, count in true_sentiment_counts.items():
        assert count == sentiment_counts[sentiment]

    sentiment_counts = get_sentiment_counts(TRAIN_COLLECTION, SENTIMENT_KEY)
    assert len(true_sentiment_counts) == len(sentiment_counts)
    for sentiment, count in true_sentiment_counts.items():
        assert count/total == sentiment_counts[sentiment] 
    with pytest.raises(KeyError):
        get_sentiment_counts(TRAIN_COLLECTION, 'wrong_key')

def test_average_target_per_sentences():
    number_targets = 19.0
    number_sentences = 6.0
    true_ats = number_targets / number_sentences
    assert true_ats == average_target_per_sentences(TRAIN_COLLECTION, 
                                                    sentence_must_contain_targets=False)
    number_sentences = 5.0
    true_ats = number_targets / number_sentences
    assert true_ats == average_target_per_sentences(TRAIN_COLLECTION, 
                                                    sentence_must_contain_targets=True)

@pytest.mark.parametrize("lower", (False, None))
def test_dataset_target_sentiment_statistics(lower: bool):
    TRAIN_COLLECTION.name = 'train'
    if lower is not None:
        target_stats = dataset_target_sentiment_statistics([TRAIN_COLLECTION], 
                                                 lower_target=lower)
    else:
        target_stats = dataset_target_sentiment_statistics([TRAIN_COLLECTION])
    pos_percent = get_sentiment_counts(TRAIN_COLLECTION, SENTIMENT_KEY)['positive'] * 100
    neu_percent = get_sentiment_counts(TRAIN_COLLECTION, SENTIMENT_KEY)['neutral'] * 100
    neg_percent = get_sentiment_counts(TRAIN_COLLECTION, SENTIMENT_KEY)['negative'] * 100
    true_stats = {'Name': 'train', 'No. Sentences': 6, 'No. Sentences(t)': 5,
                  'No. Targets': 19, 'No. Uniq Targets': 13, 'ATS': 19/6.0,
                  'ATS(t)': 19/5.0, 'POS %': pos_percent, 'NEG %': neg_percent,
                  'NEU %': neu_percent, 'TL (1)': 17/19.0, 'TL (2)': 2/19.0,
                  'TL (3+)': 0.0}
    if lower == False:
        true_stats['No. Uniq Targets'] = 14
    print(target_stats)
    assert 1 == len(target_stats)
    target_stats = target_stats[0]
    assert len(true_stats) == len(target_stats)
    for stat_name, stat in true_stats.items():
        if re.search(r'^TL', stat_name):
            assert math.isclose(stat, target_stats[stat_name], rel_tol=0.001)
        else:
            assert stat == target_stats[stat_name], stat_name

    # Multiple collections, where one collection is just the subset of the other
    subcollection = TargetTextCollection(name='sub')
    subcollection.add(TRAIN_COLLECTION["81207500773427072"])
    subcollection.add(TRAIN_COLLECTION["78522643479064576"])
    if lower is not None:
        target_stats = dataset_target_sentiment_statistics([subcollection, TRAIN_COLLECTION], 
                                                 lower_target=lower)
    else:
        target_stats = dataset_target_sentiment_statistics([subcollection, TRAIN_COLLECTION])
    
    pos_percent = get_sentiment_counts(subcollection, SENTIMENT_KEY)['positive'] * 100
    neu_percent = get_sentiment_counts(subcollection, SENTIMENT_KEY)['neutral'] * 100
    neg_percent = get_sentiment_counts(subcollection, SENTIMENT_KEY)['negative'] * 100
    sub_stats = {'Name': 'sub', 'No. Sentences': 2, 'No. Sentences(t)': 2,
                 'No. Targets': 7, 'No. Uniq Targets': 7, 'ATS': 7/2.0,
                 'ATS(t)': 7/2.0, 'POS %': pos_percent, 'NEG %': neg_percent,
                 'NEU %': neu_percent, 'TL (1)': 6/7.0, 'TL (2)': 1/7.0, 
                 'TL (3+)': 0.0}
    true_stats = [sub_stats, true_stats]
    assert len(true_stats) == len(target_stats)
    for stat_index, stat in enumerate(true_stats):
        test_stat = target_stats[stat_index]
        assert len(stat) == len(test_stat)
        for stat_name, stat_value in stat.items():
            if re.search(r'^TL', stat_name):
                assert math.isclose(stat_value, test_stat[stat_name], rel_tol=0.001)
            else:
                assert stat_value == test_stat[stat_name], stat_name


["immigration", "patients", "immigration", "NHS", "Tory"]
["Ed Balls", "@politics_co_uk", "SNPout", "osborne"]
["BattleForNumber10", "NHS", "immigration", "spending cuts"]
["Police", "crime", "Conservatives"]
["police", "crime", "Conservatives"]
def test_tokens_per_target():
    # standard/normal case
    length_count = tokens_per_target(TRAIN_COLLECTION, 'targets', whitespace())
    true_length_count = {1: 17, 2: 2}
    assert len(length_count) == len(true_length_count)
    for length, count in true_length_count.items():
        assert count == length_count[length]
    # normalise
    length_frac = tokens_per_target(TRAIN_COLLECTION, 'targets', whitespace(), 
                                    normalise=True)
    true_length_frac = {1: 17/19.0, 2: 2/19.0}
    assert len(length_frac) == len(true_length_frac)
    for length, frac in true_length_frac.items():
        assert math.isclose(frac, length_frac[length], rel_tol=0.01)
    # cumulative percentage
    length_dist = tokens_per_target(TRAIN_COLLECTION, 'targets', whitespace(), 
                                    cumulative_percentage=True)
    true_length_dist = {1: 17/19.0, 2: 1.0}
    assert len(length_dist) == len(true_length_dist)
    for length, dist in true_length_dist.items():
        assert math.isclose(dist * 100, length_dist[length], rel_tol=0.01)


@pytest.mark.parametrize("lower", (False, None))
def test_dataset_target_extraction_statistics(lower: bool):
    TRAIN_COLLECTION.name = 'train'
    if lower is not None:
        target_stats = dataset_target_extraction_statistics([TRAIN_COLLECTION], 
                                                            lower_target=lower)
    else:
        target_stats = dataset_target_extraction_statistics([TRAIN_COLLECTION])
    true_stats = {'Name': 'train', 'No. Sentences': 6, 'No. Sentences(t)': 5,
                  'No. Targets': 19, 'No. Uniq Targets': 13, 'ATS': 19/6.0,
                  'ATS(t)': 19/5.0, 'TL (1)': 17/19.0, 'TL (2)': 2/19.0,
                  'TL (3+)': 0.0}
    if lower == False:
        true_stats['No. Uniq Targets'] = 14
    assert 1 == len(target_stats)
    target_stats = target_stats[0]
    assert len(true_stats) == len(target_stats)
    for stat_name, stat in true_stats.items():
        if re.search(r'^TL', stat_name):
            assert math.isclose(stat, target_stats[stat_name], rel_tol=0.001)
        else:
            assert stat == target_stats[stat_name], stat_name

    # Multiple collections, where one collection is just the subset of the other
    subcollection = TargetTextCollection(name='sub')
    subcollection.add(TRAIN_COLLECTION["81207500773427072"])
    subcollection.add(TRAIN_COLLECTION["78522643479064576"])
    long_target = TargetText(text='some text that contains a long target or two',
                             spans=[Span(0,14), Span(15, 37)], 
                             targets=['some text that', 'contains a long target'],
                             target_sentiments=['positive', 'negative'], 
                             text_id='100')
    subcollection.add(long_target)
    if lower is not None:
        target_stats = dataset_target_extraction_statistics([subcollection, TRAIN_COLLECTION], 
                                                            lower_target=lower)
    else:
        target_stats = dataset_target_extraction_statistics([subcollection, TRAIN_COLLECTION])
    
    sub_stats = {'Name': 'sub', 'No. Sentences': 3, 'No. Sentences(t)': 3,
                 'No. Targets': 9, 'No. Uniq Targets': 9, 'ATS': 9/3.0,
                 'ATS(t)': 9/3.0, 'TL (1)': 6/9.0, 'TL (2)': 1/9.0,
                 'TL (3+)': 2/9.0}
    true_stats = [sub_stats, true_stats]
    assert len(true_stats) == len(target_stats)
    for stat_index, stat in enumerate(true_stats):
        test_stat = target_stats[stat_index]
        assert len(stat) == len(test_stat)
        for stat_name, stat_value in stat.items():
            if re.search(r'^TL', stat_name):
                assert math.isclose(stat_value, test_stat[stat_name], rel_tol=0.001)
            else: 
                assert stat_value == test_stat[stat_name], stat_name