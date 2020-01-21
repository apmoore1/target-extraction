from typing import List, Tuple, Callable
import math
from pathlib import Path

import pytest
from flaky import flaky
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from target_extraction.analysis import util, sentiment_metrics
from target_extraction.data_types import TargetTextCollection, TargetText
from target_extraction.data_types_util import Span
from target_extraction.analysis.sentiment_error_analysis import (ERROR_SPLIT_SUBSET_NAMES,
                                                                 subset_name_to_error_split,
                                                                 PLOT_SUBSET_ABBREVIATION)

DATA_DIR = Path(__file__, '..', '..', 'data', 'analysis', 'util').resolve()

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

@flaky
@pytest.mark.parametrize('assume_normal', (True, False))
def test_metric_p_values(assume_normal: bool):
    # Laptops between the datasets should be no difference but there should be 
    # a different on the Restaurant dataset
    datasets = ['Laptop'] * 200
    datasets = datasets + ['Restaurant'] * 14
    models = ['TDLSTM'] * 100
    models = models + ['IAN'] * 100
    models = models + ['TDLSTM'] * 7
    models = models + ['IAN'] * 7
    metric = np.random.normal(loc=2.2, scale=0.1, size=100).tolist()
    metric = metric + np.random.normal(loc=2.4, scale=0.1, size=100).tolist()
    metric = metric + np.random.normal(loc=2.5, scale=0.3, size=7).tolist()
    metric = metric + np.random.normal(loc=1.5, scale=0.3, size=7).tolist()
    df = pd.DataFrame({'Dataset': datasets, 'Model': models, 'Accuracy': metric})
    p_value_df = util.metric_p_values(df, 'TDLSTM', ['IAN'], ['Laptop', 'Restaurant'], 
                                      [('Accuracy', assume_normal)], 
                                      better_and_compare_column_name='Model')
    assert (2, 5) == p_value_df.shape
    columns = set(['Metric', 'Dataset', 'P-Value', 'Compared Model', 'Better Model'])
    assert columns == set(p_value_df.columns)
    assert 1 == sum(p_value_df['P-Value'] < 0.05)

    # IAN is never better than TDLSTM
    p_value_df = util.metric_p_values(df, 'IAN', ['TDLSTM'], ['Laptop', 'Restaurant'], 
                                      [('Accuracy', assume_normal)], 
                                      better_and_compare_column_name='Model')
    assert 1 == sum(p_value_df['P-Value'] > 0.05)

    # Test only having one dataset
    p_value_df = util.metric_p_values(df, 'IAN', ['TDLSTM'], ['Restaurant'], 
                                      [('Accuracy', assume_normal)], 
                                      better_and_compare_column_name='Model')
    assert 1 == sum(p_value_df['P-Value'] > 0.05)
    assert (1, 5) == p_value_df.shape

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

@pytest.mark.parametrize('plotting_one_row', (True, False))
def test_plot_error_subsets(plotting_one_row: bool):
    # All that will be tested here is that the plots do not raise any error
    # this is probably not the best way to test this function.
    plotting_data = Path(DATA_DIR, 'plotting_data.tsv')
    plotting_data = pd.read_csv(plotting_data, sep='\t')
    plotting_data = plotting_data.drop(columns=['index', 'Unnamed: 0'])
    # The data needs formatting before plotting
    all_subset_names = [name for subset_names in ERROR_SPLIT_SUBSET_NAMES.values() 
                        for name in subset_names]
    plotting_data = util.long_format_metrics(plotting_data, all_subset_names)
    plotting_data = plotting_data.rename(columns={'Accuracy': 'Overall Accuracy'})
    plotting_data['Error Split'] = plotting_data.apply(lambda x: subset_name_to_error_split(x['Metric']), 1)
    plotting_data = plotting_data.rename(columns={'Metric': 'Error Subset'})
    plotting_data['Accuracy'] = plotting_data['Metric Score']
    plotting_data = plotting_data.drop(columns=['Metric Score'])

    axs_shape = (5, 3)
    if plotting_one_row:
        error_split_name = plotting_data['Error Split'].unique().tolist()[0]
        plotting_data = plotting_data[plotting_data['Error Split']==error_split_name]
        axs_shape = (3,)
    fig, axs = util.plot_error_subsets(plotting_data, 'Dataset', 'Error Split', 
                                      'Error Subset', 'Accuracy', df_hue_name='Model',
                                      legend_column=1, title_on_every_plot=True)
    assert axs_shape == axs.shape
    # Non-Standard plot
    fig, axs = util.plot_error_subsets(plotting_data, 'Dataset', 'Error Split', 
                                      'Error Subset', 'Accuracy', df_hue_name='Model',
                                      seaborn_plot_name='boxenplot',
                                      legend_column=1, title_on_every_plot=True)
    assert axs_shape == axs.shape
    # Non-Standard plot with kwargs to the plot function
    fig, axs = util.plot_error_subsets(plotting_data, 'Dataset', 'Error Split', 
                                      'Error Subset', 'Accuracy', df_hue_name='Model',
                                      seaborn_plot_name='boxenplot',
                                      seaborn_kwargs={'dodge': True},
                                      legend_column=1, title_on_every_plot=True)
    # with a different figure size and not all plots having titles
    fig, axs = util.plot_error_subsets(plotting_data, 'Dataset', 'Error Split', 
                                      'Error Subset', 'Accuracy', df_hue_name='Model',
                                      figsize=(10,12),
                                      legend_column=1, title_on_every_plot=False)
    assert axs_shape == axs.shape
    # Plotting the overall metric as well. E.g. another plot on each plot 
    fig, axs = util.plot_error_subsets(plotting_data, 'Dataset', 'Error Split', 
                                      'Error Subset', 'Accuracy', df_hue_name='Model',
                                      df_overall_metric='Overall Accuracy',
                                      overall_seaborn_plot_name='lineplot',
                                      overall_seaborn_kwargs={'ci': 'sd'},
                                      legend_column=1, title_on_every_plot=False)
    assert axs_shape == axs.shape

@pytest.mark.parametrize('line_indxes', (True, False))
@pytest.mark.parametrize('heatmap_kwargs', (True, False))
@pytest.mark.parametrize('ax', (True, False))
@pytest.mark.parametrize('lines', (True, False))
def test_create_subset_heatmap(lines: bool, ax: bool, heatmap_kwargs: bool, 
                               line_indxes: bool):
    # All that will be tested here is that the plots do not raise any error
    # this is probably not the best way to test this function.
    all_results = Path(DATA_DIR, 'plotting_data.tsv')
    all_results = pd.read_csv(all_results, sep='\t')
    all_results = all_results.drop(columns=['index', 'Unnamed: 0'])

    # Get the subset metric data
    all_subset_names = [name for subset_names in ERROR_SPLIT_SUBSET_NAMES.values() 
                        for name in subset_names]
    all_subset_results = util.long_format_metrics(all_results, all_subset_names)
    all_subset_results = all_subset_results.rename(columns={'Accuracy': 'Overall Accuracy'})
    all_subset_results['Overall Accuracy'] = all_subset_results['Overall Accuracy'] * 100
    all_subset_results = all_subset_results.drop(columns='Macro F1')
    all_subset_results['Error Split'] = all_subset_results.apply(lambda x: subset_name_to_error_split(x['Metric']), 1)
    all_subset_results['Accuracy'] = all_subset_results['Metric Score'] * 100
    all_subset_results = all_subset_results.rename(columns={'Metric': 'Error Subset'})
    all_subset_results['Error Subset'] = all_subset_results.apply(lambda x: PLOT_SUBSET_ABBREVIATION[x['Error Subset']], 1)
    all_subset_results = all_subset_results.drop(columns=['Metric Score'])

    metric_assumed_normal = [('Accuracy', True)]
    dataset_names = ['Laptop', 'Restaurant', 'Election']
    tdsa_model_names = ['TDLSTM', 'Att-AE', 'IAN']
    all_error_subset_p_values = []
    error_splits = all_subset_results['Error Split'].unique().tolist()
    for error_split in error_splits:
        error_split_data_df = all_subset_results[all_subset_results['Error Split']==error_split]
        error_subsets = error_split_data_df['Error Subset'].unique().tolist()
        for error_subset in error_subsets:
            error_subset_df = error_split_data_df[error_split_data_df['Error Subset'] == error_subset]
            for tdsa_model_name in tdsa_model_names:
                p_value_df = util.metric_p_values(error_subset_df, 
                                                  f'{tdsa_model_name}', 
                                                  ['CNN'], dataset_names, 
                                                  metric_assumed_normal)
                p_value_df['Error Subset'] = error_subset
                p_value_df['Error Split'] = error_split
                all_error_subset_p_values.append(p_value_df)
    combined_error_subset_p_values = pd.concat(all_error_subset_p_values, sort=False, 
                                               ignore_index=True)
    if ax:
        _, axs = plt.subplots(1,3)
    else:
        axs = [None, None, None]
    if line_indxes:
        vertical_lines_index = [0,1]
        horizontal_lines_index = [1]
    else:
        vertical_lines_index = None
        horizontal_lines_index = None
    if heatmap_kwargs:
        heatmap_kwargs = {'annot': True}
    else:
        heatmap_kwargs = None
    # Normal test case
    ax = util.create_subset_heatmap(combined_error_subset_p_values, 'P-Value', lines=lines,
                                    ax=axs[0], heatmap_kwargs=heatmap_kwargs,
                                    vertical_lines_index=vertical_lines_index,
                                    horizontal_lines_index=horizontal_lines_index)
    # Different plot colors
    ax = util.create_subset_heatmap(combined_error_subset_p_values, 'P-Value', lines=lines,
                                    cubehelix_palette_kwargs={'light': 0.8},
                                    ax=axs[1], heatmap_kwargs=heatmap_kwargs,
                                    vertical_lines_index=vertical_lines_index,
                                    horizontal_lines_index=horizontal_lines_index)
    # Custom agg function
    alpha = 0.05
    def p_value_count(alpha: float) -> Callable[[pd.Series], float]:
        def alpha_count(p_values: pd.Series) -> float:
            significant_p_values = p_values <= alpha
            return int(np.sum(significant_p_values))
        return alpha_count
    ax = util.create_subset_heatmap(combined_error_subset_p_values, 'P-Value', lines=lines,
                                    pivot_table_agg_func=p_value_count(alpha),
                                    ax=axs[2], heatmap_kwargs=heatmap_kwargs,
                                    vertical_lines_index=vertical_lines_index,
                                    horizontal_lines_index=horizontal_lines_index)

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
        standard_columns = standard_columns + ['STAC', 'STAC 1', 'STAC Multi']
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
    


