from collections import defaultdict
from pathlib import Path
from typing import Dict
import tempfile

import pytest

from target_extraction.data_types import TargetTextCollection
from target_extraction.dataset_parsers import multi_aspect_multi_sentiment_acsa
from target_extraction.dataset_parsers import CACHE_DIRECTORY

def test_multi_aspect_multi_sentiment_acsa():

    def get_samples_per_sentiment(collection: TargetTextCollection) -> Dict[str, int]:
        '''
        :param collection: The collection being tested
        :returns: A dictionary where keys are the sentiment value names and the 
                  the value is the number of samples that have that sentiment 
                  value name e.g. {'pos': 500, 'neg': 400}
        '''
        sentiment_samples = defaultdict(lambda: 0)
        for value in collection.values():
            for sentiment in value['category_sentiments']:
                sentiment_samples[sentiment] += 1
        return sentiment_samples
    dataset_stats = {'train': {'positive': 1929, 'neutral': 3077, 'negative': 2084},
                     'val': {'positive': 241, 'negative': 259, 'neutral': 388},
                     'test': {'positive': 245, 'negative': 263, 'neutral': 393}}
    unique_categories = {'food', 'service', 'staff', 'price', 'ambience', 
                         'menu', 'place', 'miscellaneous'}
    for dataset, stat in dataset_stats.items():
        dataset_collection = multi_aspect_multi_sentiment_acsa(dataset)
        sentiment_breakdown = get_samples_per_sentiment(dataset_collection)
        for sentiment_name, sample_count in stat.items():
            assert sample_count == sentiment_breakdown[sentiment_name]
        assert sum(stat.values()) == dataset_collection.number_categories()
        assert 3 == len(sentiment_breakdown.keys())

        assert len(unique_categories) == len(dataset_collection.category_count())
        assert unique_categories == set(dataset_collection.category_count().keys())
    default_cache = Path(CACHE_DIRECTORY, 'Jiang 2019 MAMS ACSA')
    assert 9 == len(list(default_cache.iterdir()))

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_fp = Path(temp_dir)
        for dataset in dataset_stats.keys():
            multi_aspect_multi_sentiment_acsa(dataset, cache_dir=temp_dir_fp)
        assert 9 == len(list(Path(temp_dir_fp, 'Jiang 2019 MAMS ACSA').iterdir()))
    
    with pytest.raises(ValueError):
        multi_aspect_multi_sentiment_acsa('error')


