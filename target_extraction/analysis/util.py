import copy
from collections import defaultdict
from typing import Optional, Callable, Union, List, Dict, Any, Tuple

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from target_extraction.data_types import TargetTextCollection, TargetText
from target_extraction.analysis import sentiment_metrics
from target_extraction.analysis.sentiment_error_analysis import (distinct_sentiment,
                                                                 swap_list_dimensions,
                                                                 reduce_collection_by_key_occurrence)

def metric_df(target_collection: TargetTextCollection, 
              metric_function: Callable[[TargetTextCollection, str, str, bool, bool, Optional[int]], 
                                        Union[float, List[float]]],
              true_sentiment_key: str, predicted_sentiment_keys: List[str], 
              average: bool, array_scores: bool, 
              assert_number_labels: Optional[int] = None, 
              metric_name: str = 'metric', 
              include_run_number: bool = False) -> pd.DataFrame:
    '''
    :param target_collection: Collection of targets that have true and predicted 
                              sentiment values.
    :param metric_function: A metric function from 
                            :py:func:`target_extraction.analysis.sentiment_metrics`
    :param true_sentiment_key: Key in the `target_collection` targets that 
                               contains the true sentiment scores for each 
                               target in the TargetTextCollection
    :param predicted_sentiment_keys: The name of the preidcted sentiment keys 
                                     within the TargetTextCollection for 
                                     which the metric function should be applied
                                     to.
    :param average: For each predicted sentiment key it will return the 
                    average metric score across the *N* predictions made for 
                    each predicted sentiment key.
    :param array_scores: If average is False then this will return all of the 
                         *N* model runs metric scores.
    :param assert_number_labels: Whether or not to assert this many number of unique  
                                 labels must exist in the true sentiment key. 
                                 If this is None then the assertion is not raised.
    :param metric_name: The name to give to the metric value column.
    :param include_run_number: If `array_scores` is True then this will add an 
                               extra column to the returned dataframe (`run number`) 
                               which will include the model run number. This can 
                               be used to uniquely identify each row when combined 
                               with the `prediction key` string.
    :returns: A pandas DataFrame with two columns: 1. The prediction 
              key string 2. The metric value. Where the number of rows in the 
              DataFrame is either Number of prediction keys when `average` is 
              `True` or Number of prediction keys * Number of model runs when 
              `array_scores` is `True`
    :raises ValueError: If `include_run_number` is True and `array_scores` is 
                        False.
    '''
    if include_run_number is not None and not array_scores:
        raise ValueError('Can only have `include_run_number` as True if '
                         '`array_scores` is also True')
    df_predicted_keys = []
    df_metric_values = []
    df_run_numbers = []

    for predicted_sentiment_key in predicted_sentiment_keys:
        metric_scroes = metric_function(target_collection, true_sentiment_key, 
                                        predicted_sentiment_key, average=average, 
                                        array_scores=array_scores, 
                                        assert_number_labels=assert_number_labels)
        if isinstance(metric_scroes, list):
            for metric_index, metric_score in enumerate(metric_scroes):
                df_metric_values.append(metric_score)
                df_predicted_keys.append(predicted_sentiment_key)
                df_run_numbers.append(metric_index)
        else:
            df_metric_values.append(metric_scroes)
            df_predicted_keys.append(predicted_sentiment_key)
    df_dict = {f'{metric_name}': df_metric_values, 
               'prediction key': df_predicted_keys}
    if include_run_number:
        df_dict['run number'] = df_run_numbers
    return pd.DataFrame(df_dict)

def add_metadata_to_df(df: pd.DataFrame, target_collection: TargetTextCollection, 
                       metadata_prediction_key: str, 
                       metadata_keys: Optional[List[str]] = None
                       ) -> pd.DataFrame:
    '''
    :param df: A DataFrame that contains at least one column named `prediction key`
               of which the values in `prediction key` releate to the keys within
               TargetTextCollection that store the related to predicted values
    :param target_collection: The collection that stores `prediction key` and 
                              the metadata within `target_collection.metadata`
    :param metadata_prediction_key: The key that stores all of the metadata 
                                    associated to the `prediction key` values 
                                    within `target_collection.metadata`
    :param metadata_keys: If not None will only add the metadata keys that relate 
                          to the `prediction key` that are stated in this list of 
                          Strings else will add all.
    :returns: The `df` dataframe but with new columns that are the names of the 
              metadata fields with the values being the values from those metadata
              fields that relate to the `prediction key` value.
    :raises KeyError: If any of the `prediction key` values are not keys within 
                      the TargetTextCollection targets.
    :raises KeyError: If any of the prediction key values in the dataframe are not 
                      in the `target_collection` metadata.
    '''
    metadata = target_collection.metadata
    if metadata is None:
        return df
    if metadata_prediction_key not in metadata:
        return df
    prediction_keys = df['prediction key'].unique()
    for prediction_key in prediction_keys:
        for target_text in target_collection.values():
            target_text: TargetText
            target_text._key_error(prediction_key)
    
    
    metadata_prediction_data: Dict[str, Any] = metadata[metadata_prediction_key]
    prediction_key_metadata: Dict[str, Dict[str, Any]] = defaultdict(dict)
    metadata_column_names = set()
    
    for prediction_key in prediction_keys:
        if prediction_key not in metadata_prediction_data:
            raise KeyError(f'The following prediction key {prediction_key} is '
                           f'not in the metadata {metadata_prediction_key} '
                           f'dictionary {metadata_prediction_data}')
        for column_name, column_value in metadata_prediction_data[prediction_key].items():
            # Skip metadata columns that are to be skipped as stated in the
            # arguments
            if metadata_keys is not None:
                if column_name not in metadata_keys:
                    continue
            prediction_key_metadata[prediction_key][column_name] = column_value
            metadata_column_names.add(column_name)
    # default values for the metadata
    for metadata_column in metadata_column_names:
        df[f'{metadata_column}'] = np.nan
    # Add the metadata to the dataframe
    for prediction_key, name_value in prediction_key_metadata.items():
        for column_name, column_value in name_value.items():
            df.loc[df['prediction key']==prediction_key, column_name] = column_value
    return df

def combine_metrics(metric_df: pd.DataFrame, other_metric_df: pd.DataFrame, 
                    other_metric_name: str) -> pd.DataFrame:
    '''
    :param metric_df: DataFrame that contains all the metrics to be kept
    :param other_metric_df: Contains metric scores that are to be added to a copy 
                            of `metric_df`
    :param other_metric_name: Name of the column of the metric scores to be copied
                              from `other_metric_df`
    :returns: A copy of the `metric_df` with a new column `other_metric_name`
            that contains the other metric scores.
    :Note: This assumes that the two dataframes come from 
           :py:func:`target_extraction.analysis.util.metric_df` with the argument 
           `include_run_number` as True. This is due to the columns used to 
           combine the metric scores are `prediction key` and `run number`.
    :raises KeyError: If `prediction key` and `run number` are not columns 
                      within `metric_df` and `other_metric_df`
    '''
    index_keys = ['prediction key', 'run number']
    for df_name, df in [('metric_df', metric_df), ('other_metric_df', other_metric_df)]:
        df_columns = df.columns
        for index_key in index_keys:
                if index_key not in df_columns:
                    raise KeyError(f'The following column {index_key} does not'
                                   f'exist in {df_name} dataframe. The following'
                                   f' columns do exist {df_columns}')
    
    new_metric_df = metric_df.copy(deep=True)
    new_metric_df = new_metric_df.set_index(index_keys)
    other_metric_scores = other_metric_df.set_index(index_keys)[other_metric_name]
    new_metric_df[other_metric_name] = other_metric_scores
    new_metric_df = new_metric_df.reset_index()
    return new_metric_df

def long_format_metrics(metric_df: pd.DataFrame, 
                        metric_column_names: List[str]) -> pd.DataFrame:
    '''
    :param metric_df: DataFrame from :py:func:`target_extraction.analysis.util.metric_df`
                      that contains more than one metric score e.g. Accuracy and
                      Macro F1
    :param metric_column_names: The list of the metrics columns names that exist  
                                in `metric_df`
    :returns: A long format metric version of the `metric_df` e.g. converts a 
              DataFrame that contains `Accuracy` and `Macro F1` scores to a
              DataFrame that contains `Metric` and `Metric Score` columns where 
              the `Metric` column contains either `Accuracy` or `Macro F1` score 
              and the `Metric Score` contains the relevant metric score. This 
              will increase the number of row in `metric_df` by *N* where 
              *N* is the length of `metric_column_names`.
    '''
    columns = list(metric_df.columns)
    for metric_column in metric_column_names:
        columns.remove(metric_column)
    return pd.melt(metric_df, id_vars=columns, value_vars=metric_column_names, 
                   var_name='Metric', value_name='Metric Score')

def overall_metric_results(collection: TargetTextCollection, 
                           prediction_keys: Optional[List[str]] = None,
                           true_sentiment_key: str = 'target_sentiments',
                           strict_accuracy_metrics: bool = False
                           ) -> pd.DataFrame:
    '''
    :param collection: Dataset that contains all of the results. Furthermore it 
                       should have the name attribute as something meaningful 
                       e.g. `Laptop` for the Laptop dataset.
    :param prediction_keys: A list of prediction keys that you want the results 
                            for. If None then it will get all of the prediction 
                            keys from 
                            `collection.metadatap['predicted_target_sentiment_key']`.
    :param true_sentiment_key: Key in the `target_collection` targets that 
                               contains the true sentiment scores for each 
                               target in the TargetTextCollection.
    :param strict_accuracy_metrics: If this is True the dataframe will also 
                                    contain three additional columns: 'STAC',
                                    'STAC 1', and 'STAC Multi'. Where 'STAC'
                                    is the Strict Target Accuracy (STAC) on the 
                                    whole dataset, 'STAC 1' and 'STAC Multi' is 
                                    the STAC metric performed on the subset of 
                                    the dataset that contain either one unique 
                                    sentiment or more than one unique sentiment 
                                    per text respectively.
    :returns: A pandas dataframe with the following columns: `['prediction key', 
              'run number', 'Accuracy', 'Macro F1', 'Dataset']`. The `Dataset`
              column will contain one unique value and that will come from 
              the `name` attribute of the `collection`. The DataFrame will 
              also contain columns and values from the associated metadata see
              :py:func:`add_metadata_to_df` for more details.
    '''
    def swap_and_reduce(_collection: TargetTextCollection, 
                        subset_key: Union[str, List[str]]) -> TargetTextCollection:
        reduce_keys = ['targets', 'spans', true_sentiment_key] + prediction_keys
        for prediction_key in prediction_keys:
            _collection = swap_list_dimensions(_collection, prediction_key)
        _collection = reduce_collection_by_key_occurrence(_collection, 
                                                          subset_key, 
                                                          reduce_keys)
        for prediction_key in prediction_keys:
            _collection = swap_list_dimensions(_collection, prediction_key)
        return _collection

    if prediction_keys is None:
        prediction_keys = list(collection.metadata['predicted_target_sentiment_key'].keys())
    acc_df = metric_df(collection, sentiment_metrics.accuracy, 
                       true_sentiment_key, prediction_keys,
                       array_scores=True, assert_number_labels=3, 
                       metric_name='Accuracy', average=False, include_run_number=True)
    acc_df = add_metadata_to_df(acc_df, collection, 'predicted_target_sentiment_key')
    f1_df = metric_df(collection, sentiment_metrics.macro_f1, 
                      true_sentiment_key, prediction_keys,
                      array_scores=True, assert_number_labels=3, 
                      metric_name='Macro F1', average=False, include_run_number=True)
    combined_df = combine_metrics(acc_df, f1_df, 'Macro F1')
    if strict_accuracy_metrics:
        collection_copy = copy.deepcopy(collection)
        collection_copy = distinct_sentiment(collection_copy, separate_labels=True, 
                                             true_sentiment_key=true_sentiment_key)
        stac_multi_collection = copy.deepcopy(collection_copy)
        stac_multi_collection = swap_and_reduce(stac_multi_collection, 
                                               ['distinct_sentiment_2', 'distinct_sentiment_3'])
        stac_multi = metric_df(stac_multi_collection, sentiment_metrics.strict_text_accuracy, 
                              true_sentiment_key, prediction_keys,
                              array_scores=True, assert_number_labels=3, 
                               metric_name='STAC Multi', average=False, 
                               include_run_number=True)
        combined_df = combine_metrics(combined_df, stac_multi, 'STAC Multi')
        del stac_multi_collection
        stac_1_collection = copy.deepcopy(collection_copy)
        stac_1_collection = swap_and_reduce(stac_1_collection, 
                                           'distinct_sentiment_1')
        stac_1 = metric_df(stac_1_collection, sentiment_metrics.strict_text_accuracy, 
                           true_sentiment_key, prediction_keys,
                           array_scores=True, assert_number_labels=3, 
                           metric_name='STAC 1', average=False, 
                           include_run_number=True)
        combined_df = combine_metrics(combined_df, stac_1, 'STAC 1')
        del stac_1_collection
        del collection_copy
        stac = metric_df(collection, sentiment_metrics.strict_text_accuracy, 
                         true_sentiment_key, prediction_keys,
                         array_scores=True, assert_number_labels=3, 
                         metric_name='STAC', average=False, 
                         include_run_number=True)
        combined_df = combine_metrics(combined_df, stac, 'STAC')

    combined_df['Dataset'] = [collection.name] * combined_df.shape[0]
    return combined_df

def plot_error_subsets(metric_df: pd.DataFrame, df_column_name: str, 
                       df_row_name: str, df_x_name: str, df_y_name: str,
                       df_hue_name: str = 'Model', 
                       seaborn_plot_name: str = 'pointplot',
                       seaborn_kwargs: Optional[Dict[str, Any]] = None,
                       legend_column: Optional[int] = 0,
                       figsize: Optional[Tuple[float, float]] = None,
                       legend_bbox_to_anchor: Tuple[float, float] = (-0.13, 1.1),
                       fontsize: int = 14, legend_fontsize: int = 10,
                       tick_font_size: int = 12, 
                       title_on_every_plot: bool = False,
                       df_overall_metric: Optional[str] = None,
                       overall_seaborn_plot_name: Optional[str] = None,
                       overall_seaborn_kwargs: Optional[Dict[str, Any]] = None
                       ) -> Tuple[matplotlib.figure.Figure, 
                                  List[List[matplotlib.axes.Axes]]]:
    '''
    This function is named what it is as it is a good way to visualise the 
    different error subsets and thus error splits after running different 
    error functions from 
    :py:func`target_extraction.analysis.sentiment_error_analysis.error_analysis_wrapper`
    and further more if you are exploring them over different datasets. 
    To create a graph with these different error analysis subsets, Models, and datasets 
    the following column and row names may be useful: `df_column_name` = `Dataset`,
    `df_row_name` = `Error Split`, `df_x_name` = `Error Subset`, `df_y_name` 
    = `Accuracy (%)`, and `df_hue_name` = `Model`.

    :param metric_df: A DataFrame that will 
    :param df_column_name: Name of the column in `metric_df` that will be used 
                           to determine the categorical variables to facet the 
                           column part of the returned figure
    :param df_row_name: Name of the column in `metric_df` that will be used 
                        to determine the categorical variables to facet the 
                        row part of the returned figure
    :param df_x_name: Name of the column in `metric_df` that will be used to 
                      represent the X-axis in the figure.
    :param df_y_name: Name of the column in `metric_df` that will be used to 
                      represent the Y-axis in the figure.
    :param df_hue_name: Name of the column in `metric_df` that will be used to 
                        represent the hue in the figure
    :param seaborn_plot_name: Name of the seaborn plotting function to use as 
                              the plots within the figure
    :param seaborn_kwargs: The key word arguments to give to the seaborn 
                           plotting function.
    :param legend_column: Which column in the figure the legend should be 
                          associated too. The row the legend is associated 
                          with is fixed at row 0.
    :param figsize: Size of the figure, this is passed to the 
                    :py:func:`matplotlib.pyplot.subplots` as an argument.
    :param legend_bbox_to_anchor: Where the legend box should be within the 
                                  figure. This is passed as the `bbox_to_anchor`
                                  argument to 
                                  :py:func:`matplotlib.pyplot.Axes.legend`
    :param fontsize: Size of the font for the title, y-axis label, and 
                     x-axis label.
    :param legend_fontsize: Size of the font for the legend.
    :param tick_font_size: Size of the font on the y and x axis ticks.
    :param title_on_every_plot: Whether or not to have the title above every 
                                plot in the grid or just over the top row 
                                of plots.
    :param df_overall_metric: Name of the column in `metric_df` that stores 
                              the overall metric score for the entire dataset 
                              and not just the `subsets`.
    :param overall_seaborn_plot_name: Same as the `seaborn_plot_name` but for 
                                      plotting the overall metric
    :param overall_seaborn_kwargs: Same as the `seaborn_kwargs` but for the 
                                   overall metric plot.
    :returns: A tuple of 1. The figure  2. The associated axes within the 
              figure. The figure will contain N x M plots where N is the number 
              of unique values in the `metric_df` `df_column_name` column and 
              M is the number of unique values in the `metric_df` 
              `df_row_name` column.
    '''
    def plot_error_split(df: pd.DataFrame, 
                        error_axs: List[matplotlib.axes.Axes], 
                        column_names: List[str],
                        first_row: bool, last_row: bool,
                        number_hue_values: int = 1) -> None:
        for col_index, column_name in enumerate(column_names):
            _df = df[df[df_column_name]==column_name]
            ax = error_axs[col_index]
            getattr(sns, seaborn_plot_name)(x=df_x_name, y=df_y_name, 
                                            hue=df_hue_name, data=_df, 
                                            ax=ax, **seaborn_kwargs)
            # Required if plotting the overall metrics
            if df_overall_metric:
                _temp_overall_df: pd.DataFrame = _df.copy(deep=True)
                _temp_overall_df = _temp_overall_df.drop(columns=df_y_name)
                _temp_overall_df = _temp_overall_df.rename(columns={df_overall_metric: df_y_name})
                getattr(sns, overall_seaborn_plot_name)(x=df_x_name, y=df_y_name, 
                                                        hue=df_hue_name, 
                                                        data=_temp_overall_df, 
                                                        ax=ax, 
                                                        **overall_seaborn_kwargs)
            
            # Y axis labelling
            row_name = _df[df_row_name].unique()
            row_name_err = ('There should only be one unique row name {row_name} '
                            f'from the row column {df_row_name}')
            assert len(row_name) == 1, row_name_err
            row_name = row_name[0]
            if col_index != 0:
                ax.set_ylabel('')
            else:
                ax.set_ylabel(f'{df_row_name}={row_name}\n{df_y_name}', 
                            fontsize=fontsize)
            # X axis labelling
            if last_row:
                ax.set_xlabel(df_x_name, fontsize=fontsize)
            else:
                ax.set_xlabel('')
            # Title
            if first_row or title_on_every_plot:
                ax.set_title(f'{df_column_name}={column_name}', fontsize=fontsize)
            # Legend
            if col_index == legend_column and first_row:
                ax.legend(bbox_to_anchor=legend_bbox_to_anchor, 
                            loc='lower left', fontsize=legend_fontsize, 
                            ncol=number_hue_values, borderaxespad=0.)
            else:
                ax.get_legend().remove()
    plt.rc('xtick', labelsize=tick_font_size)
    plt.rc('ytick', labelsize=tick_font_size)
    # Seaborn plotting options
    if seaborn_kwargs is None and seaborn_plot_name=='pointplot':
        seaborn_kwargs = {'join': False, 'ci': 'sd', 'dodge': 0.4, 
                          'capsize': 0.05}
    elif seaborn_kwargs is None:
        seaborn_kwargs = {}
    # Ensure that all the values in hue column will always be the same
    hue_values = metric_df[df_hue_name].unique().tolist()
    number_hue_values = len(hue_values)
    palette = dict(zip(hue_values, sns.color_palette()))
    seaborn_kwargs['palette'] = palette

    # Determine the number of rows
    row_names = metric_df[df_row_name].unique().tolist()
    num_rows = len(row_names)
    # Number of columns
    column_names = metric_df[df_column_name].unique().tolist()
    number_columns = len(column_names)

    if figsize is None:
        length = num_rows * 4
        width = number_columns * 5
        figsize = (width, length)
    fig, axs = plt.subplots(nrows=num_rows, ncols=number_columns, 
                            figsize=figsize)
    # row 
    if num_rows > 1:
        # columns
        for row_index, row_name in enumerate(row_names):
            row_metric_df = metric_df[metric_df[df_row_name]==row_name]
            row_axs = axs[row_index]
            if row_index == (num_rows - 1):
                plot_error_split(row_metric_df, row_axs, column_names, False, 
                                    True, number_hue_values)
            elif row_index == 0:
                plot_error_split(row_metric_df, row_axs, column_names, True, 
                                    False, number_hue_values)
            else:
                plot_error_split(row_metric_df, row_axs, column_names, False, 
                                    False, number_hue_values)
    # Only 1 row but multiple columns
    else:
        plot_error_split(metric_df, axs, column_names, True, True, 
                         number_hue_values)
    return fig, axs

def create_subset_heatmap(subset_df: pd.DataFrame, value_column: str, 
                          pivot_table_agg_func: Optional[Callable[[pd.Series], Any]] = None,
                          font_label_size: int = 10,
                          cubehelix_palette_kwargs: Optional[Dict[str, Any]] = None,
                          lines: bool = True, line_color: str = 'k'
                          ) -> matplotlib.pyplot.Axes:
    '''
    :param subset_df: A DataFrame that contains the following columns: 
                      1. Error Split, 2. Error Subset, 3. Dataset, 
                      and 4. `value_column`
    :param value_column: The column that contains the value to be plotted in the 
                         heatmap.
    :param pivot_table_agg_func: As a pivot table is created to create the heatmap.
                                 This allows the replacement default aggregation 
                                 function (np.mean) with a custom function. The 
                                 pivot table aggregates the `value_column` by 
                                 Dataset, Error Split, and Error Subset.
    :param font_label_size: Font sizes of the labels on the returned plot
    :param cubehelix_palette_kwargs: Keywords arguments to give to the 
                                     seaborn.cubehelix_palette
                                     https://seaborn.pydata.org/generated/seaborn.cubehelix_palette.html.
                                     Default produces white to dark red.
    :param lines: Whether or not lines should appear on the plot to define the 
                  different error splits.
    :param line_color: Color of the lines if the lines are to be displayed. The 
                       choice of color names can be found here: 
                       https://matplotlib.org/3.1.1/gallery/color/named_colors.html#sphx-glr-gallery-color-named-colors-py
    :returns: A heatmap where the Y-axis represents the datasets, X-axis 
              represents the Error subsets formatted when appropriate with the 
              Error split name, and the values come from the `value_column`. The 
              heatmap assumes the `value_column` contains discrete values as the 
              color bar is discete rather than continous. If you want a continous 
              color bar it is recomended that you use Seaborn heatmap.
    '''
    df_copy = subset_df.copy(deep=True)
    format_error_split = lambda x: f'{x["Error Split"]}' if x["Error Split"] != "DS" else ""
    df_copy['Formatted Error Split'] =  df_copy.apply(format_error_split, 1)
    combined_split_subset = lambda x: f'{x["Formatted Error Split"]} {x["Error Subset"]}'.strip()
    df_copy['Combined Error Subset'] = df_copy.apply(combined_split_subset, 1)
    if pivot_table_agg_func is None:
        pivot_table_agg_func = np.mean
    df_copy = pd.pivot_table(data=df_copy, values=value_column, 
                             columns=['Combined Error Subset'], 
                             index=['Dataset'], aggfunc=pivot_table_agg_func)

    column_order = ['DS1', 'DS2', 'DS3', 'TSSR 1', 'TSSR 1-Multi', 'TSSR High', 
                    'TSSR Low', 'NT 1', 'NT Low', 'NT Med', 'NT High', 
                    'n-shot Zero', 'n-shot Low', 'n-shot Med', 'n-shot High', 
                    'TSR USKT', 'TSR UT', 'TSR KSKT']
    df_copy = df_copy.reindex(column_order, axis=1)

    plt.rc('xtick',labelsize=font_label_size)
    plt.rc('ytick',labelsize=font_label_size)
    unique_values = np.unique(df_copy.values)
    num_unique_values = len(unique_values)
    color_bar_spacing = max(unique_values) / num_unique_values
    half_bar_spacing = color_bar_spacing / 2
    colorbar_values = [(i * color_bar_spacing) + half_bar_spacing 
                       for i in range(len(unique_values))]
    if cubehelix_palette_kwargs is None:
        cubehelix_palette_kwargs = {'hue': 1, 'gamma': 2.2, 'light': 1.0, 
                                    'dark': 0.7}
    cmap = sns.cubehelix_palette(n_colors=num_unique_values, 
                                 **cubehelix_palette_kwargs)
    ax = sns.heatmap(df_copy, linewidths=.5, linecolor='lightgray', 
                     cmap=matplotlib.colors.ListedColormap(cmap))
    cb = ax.collections[0].colorbar
    cb.set_ticks(colorbar_values)
    cb.set_ticklabels(unique_values)
    ax.set_xlabel('Error Subset', fontsize=font_label_size)
    ax.set_ylabel('Dataset', fontsize=font_label_size)
    if lines:
        ax.vlines([0,3,7,11,15,18], colors=line_color, *ax.get_ylim())
        ax.hlines([0,1,2,3], colors=line_color, *ax.get_xlim())
    return ax