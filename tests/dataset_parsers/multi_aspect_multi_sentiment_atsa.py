from collections import defaultdict
from pathlib import Path
from typing import Dict
import tempfile

import pytest

from target_extraction.data_types import TargetTextCollection
from target_extraction.dataset_parsers import multi_aspect_multi_sentiment_atsa
from target_extraction.dataset_parsers import CACHE_DIRECTORY

def test_multi_aspect_multi_sentiment_atsa():

    def get_samples_per_sentiment(collection: TargetTextCollection) -> Dict[str, int]:
        '''
        :param collection: The collection being tested
        :returns: A dictionary where keys are the sentiment value names and the 
                  the value is the number of samples that have that sentiment 
                  value name e.g. {'pos': 500, 'neg': 400}
        '''
        sentiment_samples = defaultdict(lambda: 0)
        for value in collection.values():
            for sentiment in value['target_sentiments']:
                sentiment_samples[sentiment] += 1
        return sentiment_samples
    dataset_stats = {'train': {'positive': 3380, 'neutral': 5042, 'negative': 2764},
                     'val': {'positive': 403, 'negative': 325, 'neutral': 604},
                     'test': {'positive': 400, 'negative': 329, 'neutral': 607}}
    for dataset, stat in dataset_stats.items():
        dataset_collection = multi_aspect_multi_sentiment_atsa(dataset)
        sentiment_breakdown = get_samples_per_sentiment(dataset_collection)
        for sentiment_name, sample_count in stat.items():
            assert sample_count == sentiment_breakdown[sentiment_name]
        assert sum(stat.values()) == dataset_collection.number_targets()
        assert 3 == len(sentiment_breakdown.keys())
    default_cache = Path(CACHE_DIRECTORY, 'Jiang 2019 MAMS ATSA')
    assert 6 == len(list(default_cache.iterdir()))

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_fp = Path(temp_dir)
        for dataset in dataset_stats.keys():
            multi_aspect_multi_sentiment_atsa(dataset, cache_dir=temp_dir_fp)
        assert 6 == len(list(Path(temp_dir_fp, 'Jiang 2019 MAMS ATSA').iterdir()))
    
    with pytest.raises(ValueError):
        multi_aspect_multi_sentiment_atsa('error')


