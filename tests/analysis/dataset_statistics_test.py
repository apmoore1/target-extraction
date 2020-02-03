from pathlib import Path

import pytest

from target_extraction.data_types import TargetTextCollection
from target_extraction.analysis.dataset_statistics import get_sentiment_counts
from target_extraction.analysis.dataset_statistics import average_target_per_sentences
from target_extraction.analysis.dataset_statistics import dataset_target_statistics

DATA_DIR = Path(__file__, '..', '..', 'data', 'analysis', 'sentiment_error_analysis').resolve()
TRAIN_COLLECTION =  TargetTextCollection.load_json(Path(DATA_DIR, 'train_with_blank.json'))
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
    sentiment_counts = get_sentiment_counts(TRAIN_COLLECTION, normalised=False)
    assert len(true_sentiment_counts) == len(sentiment_counts)
    for sentiment, count in true_sentiment_counts.items():
        assert count == sentiment_counts[sentiment]

    sentiment_counts = get_sentiment_counts(TRAIN_COLLECTION)
    assert len(true_sentiment_counts) == len(sentiment_counts)
    for sentiment, count in true_sentiment_counts.items():
        assert count/total == sentiment_counts[sentiment] 

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


["immigration", "patients", "immigration", "NHS", "Tory"]
["Ed Balls", "@politics_co_uk", "SNPout", "osborne"]
["BattleForNumber10", "NHS", "immigration", "spending cuts"]
["Police", "crime", "Conservatives"]
["police", "crime", "Conservatives"]

["negative", "neutral", "neutral", "neutral"]
["neutral", "positive", "neutral"]
@pytest.mark.parametrize("lower", (False, None))
def test_dataset_target_statistics(lower: bool):
    TRAIN_COLLECTION.name = 'train'
    if lower is not None:
        target_stats = dataset_target_statistics([TRAIN_COLLECTION], 
                                                 lower_target=lower)
    else:
        target_stats = dataset_target_statistics([TRAIN_COLLECTION])
    pos_percent = get_sentiment_counts(TRAIN_COLLECTION)['positive'] * 100
    neu_percent = get_sentiment_counts(TRAIN_COLLECTION)['neutral'] * 100
    neg_percent = get_sentiment_counts(TRAIN_COLLECTION)['negative'] * 100
    true_stats = {'Name': 'train', 'No. Sentences': 6, 'No. Sentences(t)': 5,
                  'No. Targets': 19, 'No. Uniq Targets': 13, 'ATS': 19/6.0,
                  'ATS(t)': 19/5.0, 'POS %': pos_percent, 'NEG %': neg_percent,
                  'NEU %': neu_percent}
    if lower == False:
        true_stats['No. Uniq Targets'] = 14
    assert 1 == len(target_stats)
    target_stats = target_stats[0]
    assert len(true_stats) == len(target_stats)
    for stat_name, stat in true_stats.items():
        assert stat == target_stats[stat_name], stat_name

    # Multiple collections, where one collection is just the subset of the other
    subcollection = TargetTextCollection(name='sub')
    subcollection.add(TRAIN_COLLECTION["81207500773427072"])
    subcollection.add(TRAIN_COLLECTION["78522643479064576"])
    if lower is not None:
        target_stats = dataset_target_statistics([subcollection, TRAIN_COLLECTION], 
                                                 lower_target=lower)
    else:
        target_stats = dataset_target_statistics([subcollection, TRAIN_COLLECTION])
    
    pos_percent = get_sentiment_counts(subcollection)['positive'] * 100
    neu_percent = get_sentiment_counts(subcollection)['neutral'] * 100
    neg_percent = get_sentiment_counts(subcollection)['negative'] * 100
    sub_stats = {'Name': 'sub', 'No. Sentences': 2, 'No. Sentences(t)': 2,
                 'No. Targets': 7, 'No. Uniq Targets': 7, 'ATS': 7/2.0,
                 'ATS(t)': 7/2.0, 'POS %': pos_percent, 'NEG %': neg_percent,
                 'NEU %': neu_percent}
    true_stats = [sub_stats, true_stats]
    assert len(true_stats) == len(target_stats)
    for stat_index, stat in enumerate(true_stats):
        test_stat = target_stats[stat_index]
        assert len(stat) == len(test_stat)
        for stat_name, stat_value in stat.items():
            assert stat_value == test_stat[stat_name], stat_name

