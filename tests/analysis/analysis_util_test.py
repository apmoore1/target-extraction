from typing import List, Tuple
import math

import pytest
import numpy as np
import pandas as pd

from target_extraction.analysis import util, sentiment_metrics
from target_extraction.data_types import TargetTextCollection, TargetText
from target_extraction.data_types_util import Span

def passable_example_multiple_preds(true_sentiment_key: str, 
                                    predicted_sentiment_key: str
                                    ) -> TargetTextCollection:
    example_1 = TargetText(text_id='1', text='some text', targets=['some', 'text'], 
                           spans=[Span(0,4), Span(5, 9)])
    example_1[true_sentiment_key] = ['pos', 'neg']
    example_1[predicted_sentiment_key] = [['pos', 'neg'], ['pos', 'neg']]
    example_2 = TargetText(text_id='2', text='some text is', targets=['some', 'text', 'is'], 
                           spans=[Span(0,4), Span(5, 9), Span(10,12)])
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

@pytest.mark.parametrize('include_run_number', (True, False))
def test_combine_metrics(include_run_number: bool):
    collection_acc = passable_example_multiple_preds('true_sentiments', 'model_1')
    acc_df = util.metric_df(collection_acc, sentiment_metrics.accuracy, 'true_sentiments', 
                            predicted_sentiment_keys=['model_1'], 
                            average=False, array_scores=True, metric_name='Accuracy',
                            include_run_number=include_run_number)
    f1_df = util.metric_df(collection_acc, sentiment_metrics.macro_f1, 'true_sentiments', 
                           predicted_sentiment_keys=['model_1'], 
                           average=False, array_scores=True, metric_name='Macro F1',
                           include_run_number=include_run_number)
    if include_run_number:
        combined_df = util.combine_metrics(acc_df, f1_df, 'Macro F1')
        assert pd.Series.equals(f1_df['Macro F1'], combined_df['Macro F1'])
        assert pd.Series.equals(acc_df['Accuracy'], combined_df['Accuracy'])
        assert (2, 3) == acc_df.shape
        assert (2, 3) == f1_df.shape
        assert (2, 4) == combined_df.shape
    else:
        with pytest.raises(KeyError):
            util.combine_metrics(acc_df, f1_df, 'Macro F1')

def test_long_format_metrics():
    # 2 metrics
    collection_acc = passable_example_multiple_preds('true_sentiments', 'model_1')
    acc_df = util.metric_df(collection_acc, sentiment_metrics.accuracy, 'true_sentiments', 
                            predicted_sentiment_keys=['model_1'], 
                            average=False, array_scores=True, metric_name='Accuracy',
                            include_run_number=True)
    f1_df = util.metric_df(collection_acc, sentiment_metrics.macro_f1, 'true_sentiments', 
                           predicted_sentiment_keys=['model_1'], 
                           average=False, array_scores=True, metric_name='Macro F1',
                           include_run_number=True)
    combined_df = util.combine_metrics(acc_df, f1_df, 'Macro F1')
    combined_rows, combined_cols = combined_df.shape
    long_df = util.long_format_metrics(combined_df, ['Accuracy', 'Macro F1'])
    assert combined_rows * 2 == long_df.shape[0]
    assert combined_cols == long_df.shape[1]
    # 3 metrics
    ano_f1_df = util.metric_df(collection_acc, sentiment_metrics.macro_f1, 'true_sentiments', 
                           predicted_sentiment_keys=['model_1'], 
                           average=False, array_scores=True, metric_name='Another Macro F1',
                           include_run_number=True)
    combined_df = util.combine_metrics(combined_df, ano_f1_df, 'Another Macro F1')
    long_df = util.long_format_metrics(combined_df, ['Accuracy', 'Macro F1', 'Another Macro F1'])
    combined_rows, combined_cols = combined_df.shape
    assert combined_rows * 3 == long_df.shape[0]
    assert combined_cols - 1 == long_df.shape[1]
    for metric_column in ['Accuracy', 'Macro F1', 'Another Macro F1']:
        long_df_scores = long_df[long_df['Metric']==f'{metric_column}']['Metric Score'].tolist()
        combined_df_scores = combined_df[f'{metric_column}'].tolist()
        assert long_df_scores == combined_df_scores

#def test_plot_error_subsets():
#    # All that will be tested here is that the plots do not raise any error
#    util.plot_error_subsets
@pytest.mark.parametrize('true_sentiment_key', ('true_sentiments', None))
@pytest.mark.parametrize('include_metadata', (True, False))
@pytest.mark.parametrize('strict_accuracy_metrics', (True, False))
def test_overall_metric_results(true_sentiment_key: str, 
                                include_metadata: bool,
                                strict_accuracy_metrics: bool):
    if true_sentiment_key is None:
        true_sentiment_key = 'target_sentiments'
    model_1_collection = passable_example_multiple_preds(true_sentiment_key, 'model_1')
    model_1_collection.add(TargetText(text='a', text_id='200', spans=[Span(0,1)], targets=['a'],
                                      model_1=[['pos'], ['neg']], **{f'{true_sentiment_key}': ['pos']}))
    model_1_collection.add(TargetText(text='a', text_id='201', spans=[Span(0,1)], targets=['a'],
                                      model_1=[['pos'], ['neg']], **{f'{true_sentiment_key}': ['neg']}))
    model_1_collection.add(TargetText(text='a', text_id='202', spans=[Span(0,1)], targets=['a'],
                                      model_1=[['pos'], ['neg']], **{f'{true_sentiment_key}': ['neu']}))
    print(true_sentiment_key)
    print(model_1_collection['1'])
    print(model_1_collection['200'])
    model_2_collection = passable_example_multiple_preds(true_sentiment_key, 'model_2')
    combined_collection = TargetTextCollection()
    
    standard_columns = ['Dataset', 'Macro F1', 'Accuracy', 'run number', 
                        'prediction key']
    if strict_accuracy_metrics:
        standard_columns = standard_columns + ['STA', 'STA 1', 'STA Multi']
    if include_metadata:
        metadata = {'predicted_target_sentiment_key': {'model_1': {'CWR': True},
                                                       'model_2': {'CWR': False}}}
        combined_collection.name = 'name'
        combined_collection.metadata = metadata
        standard_columns.append('CWR')
    number_df_columns = len(standard_columns)

    for key, value in model_1_collection.items():
        if key in ['200', '201', '202']:
            combined_collection.add(value)
            combined_collection[key]['model_2'] = [['neg'], ['pos']]
            continue
        combined_collection.add(value)
        combined_collection[key]['model_2'] = model_2_collection[key]['model_2']
    if true_sentiment_key is None:
        result_df = util.overall_metric_results(combined_collection, 
                                                ['model_1', 'model_2'], 
                                                strict_accuracy_metrics=strict_accuracy_metrics)
    else:
        result_df = util.overall_metric_results(combined_collection, 
                                                ['model_1', 'model_2'],
                                                true_sentiment_key, 
                                                strict_accuracy_metrics=strict_accuracy_metrics)
    assert (4, number_df_columns) == result_df.shape
    assert set(standard_columns) == set(result_df.columns)
    if include_metadata:
        assert ['name'] * 4 == result_df['Dataset'].tolist()
    else:
        assert [''] * 4 == result_df['Dataset'].tolist()
    # Test the case where only one model is used
    if true_sentiment_key is None:
        result_df = util.overall_metric_results(combined_collection, 
                                                ['model_1'], 
                                                strict_accuracy_metrics=strict_accuracy_metrics)
    else:
        result_df = util.overall_metric_results(combined_collection, 
                                                ['model_1'],
                                                true_sentiment_key, 
                                                strict_accuracy_metrics=strict_accuracy_metrics)
    assert (2, number_df_columns) == result_df.shape
    # Test the case where the model names come from the metadata
    if include_metadata:
        result_df = util.overall_metric_results(combined_collection, 
                                                true_sentiment_key=true_sentiment_key, 
                                                strict_accuracy_metrics=strict_accuracy_metrics)
        assert (4, number_df_columns) == result_df.shape
    else:
        with pytest.raises(KeyError):
            util.overall_metric_results(combined_collection, 
                                        true_sentiment_key=true_sentiment_key, 
                                        strict_accuracy_metrics=strict_accuracy_metrics)
    


