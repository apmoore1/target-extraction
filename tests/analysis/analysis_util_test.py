from typing import List, Tuple
import math

import pytest
import numpy as np
import pandas as pd

from target_extraction.analysis import util, sentiment_metrics
from target_extraction.data_types import TargetTextCollection, TargetText

def passable_example_multiple_preds(true_sentiment_key: str, 
                                    predicted_sentiment_key: str
                                    ) -> TargetTextCollection:
    example_1 = TargetText(text_id='1', text='some text')
    example_1[true_sentiment_key] = ['pos', 'neg']
    example_1[predicted_sentiment_key] = [['pos', 'neg'], ['pos', 'neg']]
    example_2 = TargetText(text_id='2', text='some text')
    example_2[true_sentiment_key] = ['pos', 'neg', 'neu']
    if predicted_sentiment_key == 'model_2':
        example_2[predicted_sentiment_key] = [['pos', 'neg', 'neu'], ['neg', 'neg', 'pos']]
    else:
        example_2[predicted_sentiment_key] = [['neu', 'neg', 'pos'], ['neg', 'neg', 'pos']]
    return TargetTextCollection([example_1, example_2])

@pytest.mark.parametrize('include_run_number', (True, None))
@pytest.mark.parametrize('metric_name', (None, 'scores'))
@pytest.mark.parametrize('metric_function_name', ('macro_f1', 'accuracy'))
def test_metric_df(metric_function_name: str, metric_name: str, 
                   include_run_number: bool):
    model_1_collection = passable_example_multiple_preds('true_sentiments', 'model_1')
    model_2_collection = passable_example_multiple_preds('true_sentiments', 'model_2')
    combined_collection = TargetTextCollection()
    for key, value in model_1_collection.items():
        combined_collection.add(value)
        combined_collection[key]['model_2'] = model_2_collection[key]['model_2']
    metric_function = getattr(sentiment_metrics, metric_function_name)
    # Test the array score version first
    model_1_scores = metric_function(model_1_collection, 'true_sentiments', 'model_1', 
                                     average=False, array_scores=True)
    model_2_scores = metric_function(model_2_collection, 'true_sentiments', 'model_2', 
                                     average=False, array_scores=True)
    test_df = util.metric_df(combined_collection, metric_function, 'true_sentiments', 
                             predicted_sentiment_keys=['model_1', 'model_2'], 
                             average=False, array_scores=True, metric_name=metric_name,
                             include_run_number=include_run_number)
    get_metric_name = 'metric' if None else metric_name
    if include_run_number:
        assert (4, 3) == test_df.shape
    else:
        assert (4, 2) == test_df.shape
    for model_name, true_model_scores in [('model_1', model_1_scores), 
                                          ('model_2', model_2_scores)]:
        test_model_scores = test_df.loc[test_df['prediction key']==f'{model_name}'][f'{get_metric_name}']
        assert true_model_scores == test_model_scores.to_list()
        if include_run_number:
            test_run_numbers = test_df.loc[test_df['prediction key']==f'{model_name}']['run number']
            test_run_numbers = test_run_numbers.to_list()
            assert [0, 1] == test_run_numbers
    # Test the average version
    model_1_scores = metric_function(model_1_collection, 'true_sentiments', 'model_1', 
                                     average=True, array_scores=False)
    model_2_scores = metric_function(model_2_collection, 'true_sentiments', 'model_2', 
                                     average=True, array_scores=False)
    if include_run_number:
        with pytest.raises(ValueError):
            util.metric_df(combined_collection, metric_function, 'true_sentiments', 
                           predicted_sentiment_keys=['model_1', 'model_2'], 
                           average=True, array_scores=False, metric_name=metric_name,
                           include_run_number=include_run_number)
    else:
        test_df = util.metric_df(combined_collection, metric_function, 'true_sentiments', 
                                predicted_sentiment_keys=['model_1', 'model_2'], 
                                average=True, array_scores=False, metric_name=metric_name,
                                include_run_number=include_run_number)
        get_metric_name = 'metric' if None else metric_name
        assert (2,2) == test_df.shape
        for model_name, true_model_scores in [('model_1', model_1_scores), 
                                            ('model_2', model_2_scores)]:
            test_model_scores = test_df.loc[test_df['prediction key']==f'{model_name}'][f'{get_metric_name}']
            test_model_scores = test_model_scores.to_list()
            assert 1 == len(test_model_scores)
            assert true_model_scores == test_model_scores[0]

def test_add_metadata_to_df():
    model_1_collection = passable_example_multiple_preds('true_sentiments', 'model_1')
    model_2_collection = passable_example_multiple_preds('true_sentiments', 'model_2')
    combined_collection = TargetTextCollection()
    for key, value in model_1_collection.items():
        combined_collection.add(value)
        combined_collection[key]['model_2'] = model_2_collection[key]['model_2']
    # get test metric_df
    combined_collection.metadata = None
    metric_df = util.metric_df(combined_collection, sentiment_metrics.accuracy, 
                               'true_sentiments', 
                               predicted_sentiment_keys=['model_1', 'model_2'], 
                               average=False, array_scores=True, metric_name='metric')
    # Test the case where the TargetTextCollection has no metadata, should 
    # just return the dataframe as is without change
    test_df = util.add_metadata_to_df(metric_df.copy(deep=True), combined_collection, 
                                      'non-existing-key')
    assert metric_df.equals(test_df)
    assert combined_collection.metadata is None
    # Testing the case where the metadata is not None but does not contain the 
    # `metadata_prediction_key` = `non-existing-key`
    combined_collection.name = 'combined collection'
    assert combined_collection.metadata is not None
    test_df = util.add_metadata_to_df(metric_df.copy(deep=True), combined_collection, 
                                      'non-existing-key')
    assert metric_df.equals(test_df)

    # Test the normal cases where we are adding metadata
    key_metadata_normal = {'model_1': {'embedding': True, 'value': '10'}, 
                           'model_2': {'embedding': False, 'value': '5'}}
    key_metadata_alt = {'model_1': {'embedding': True, 'value': '10'}, 
                        'model_2': {'embedding': False, 'value': '5'},
                        'model_3': {'embedding': 'low', 'value': '12'}}
    key_metadata_diff = {'model_1': {'embedding': True, 'value': '10', 'special': 12}, 
                         'model_2': {'embedding': False, 'value': '5', 'diff': 30.0}}
    key_metadataer = [key_metadata_normal, key_metadata_alt, key_metadata_diff]
    for key_metadata in key_metadataer:
        combined_collection.metadata['non-existing-key'] = key_metadata
        if 'special' in key_metadata['model_1']:
            test_df = util.add_metadata_to_df(metric_df.copy(deep=True), combined_collection, 
                                              'non-existing-key', 
                                              metadata_keys=['embedding', 'value'])
        else:
            test_df = util.add_metadata_to_df(metric_df.copy(deep=True), combined_collection, 
                                              'non-existing-key')
        assert not metric_df.equals(test_df)
        assert (4, 4) == test_df.shape
        assert [True, True] == test_df.loc[test_df['prediction key']=='model_1']['embedding'].to_list()
        assert [False, False] == test_df.loc[test_df['prediction key']=='model_2']['embedding'].to_list()
        assert ['10', '10'] == test_df.loc[test_df['prediction key']=='model_1']['value'].to_list()
        assert ['5', '5'] == test_df.loc[test_df['prediction key']=='model_2']['value'].to_list()
    # Test the case where some of the metadata exists for some of the models in
    # the collection but not all of them
    combined_collection.metadata['non-existing-key'] = key_metadata_diff
    test_df = util.add_metadata_to_df(metric_df.copy(deep=True), combined_collection, 
                                      'non-existing-key')
    assert not metric_df.equals(test_df)
    assert (4, 6) == test_df.shape
    assert [True, True] == test_df.loc[test_df['prediction key']=='model_1']['embedding'].to_list()
    assert [False, False] == test_df.loc[test_df['prediction key']=='model_2']['embedding'].to_list()
    assert ['10', '10'] == test_df.loc[test_df['prediction key']=='model_1']['value'].to_list()
    assert ['5', '5'] == test_df.loc[test_df['prediction key']=='model_2']['value'].to_list()
    assert [12, 12] == test_df.loc[test_df['prediction key']=='model_1']['special'].to_list()
    nan_values = test_df.loc[test_df['prediction key']=='model_2']['special'].to_list()
    assert 2 == len(nan_values)
    for test_value in nan_values:
        assert math.isnan(test_value)
    nan_values = test_df.loc[test_df['prediction key']=='model_1']['diff'].to_list()
    assert 2 == len(nan_values)
    for test_value in nan_values:
        assert math.isnan(test_value)
    assert [30.0, 30.0] == test_df.loc[test_df['prediction key']=='model_2']['diff'].to_list()

    # Test the KeyError cases
    # Prediction keys that exist in the metric df but not in the collection
    metric_copy_df = metric_df.copy(deep=True)
    alt_metric_df = pd.DataFrame({'prediction key': ['model_3', 'model_3'], 
                                  'metric': [0.4, 0.5]})
    metric_copy_df = metric_copy_df.append(alt_metric_df)
    assert (6, 2) == metric_copy_df.shape
    with pytest.raises(KeyError):
        util.add_metadata_to_df(metric_copy_df, combined_collection, 'non-existing-key')
    # Prediction keys exist in the dataframe and target texts but not in the 
    # metadata
    combined_collection.metadata['non-existing-key'] = {'model_1': {'embedding': True, 'value': '10'}}
    with pytest.raises(KeyError):
        util.add_metadata_to_df(metric_df.copy(deep=True), combined_collection, 'non-existing-key')