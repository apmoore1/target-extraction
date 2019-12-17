from collections import defaultdict
from typing import Optional, Callable, Union, List, Dict, Any

import pandas as pd
import numpy as np

from target_extraction.data_types import TargetTextCollection, TargetText

def metric_df(target_collection: TargetTextCollection, 
              metric_function: Callable[[TargetTextCollection, str, str, bool, bool, Optional[int]], 
                                        Union[float, List[float]]],
              true_sentiment_key: str, predicted_sentiment_keys: List[str], 
              average: bool, array_scores: bool, 
              assert_number_labels: Optional[int] = None, 
              metric_name: str = 'metric') -> pd.DataFrame:
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
    :returns: A pandas DataFrame with two columns: 1. The prediction 
              key string 2. The metric value. Where the number of rows in the 
              DataFrame is either Number of prediction keys when `average` is 
              `True` or Number of prediction keys * Number of model runs when 
              `array_scores` is `True`
    '''
    df_predicted_keys = []
    df_metric_values = []

    for predicted_sentiment_key in predicted_sentiment_keys:
        metric_scroes = metric_function(target_collection, true_sentiment_key, 
                                        predicted_sentiment_key, average=average, 
                                        array_scores=array_scores, 
                                        assert_number_labels=assert_number_labels)
        if isinstance(metric_scroes, list):
            for metric_score in metric_scroes:
                df_metric_values.append(metric_score)
                df_predicted_keys.append(predicted_sentiment_key)
        else:
            df_metric_values.append(metric_scroes)
            df_predicted_keys.append(predicted_sentiment_key)
    df_dict = {f'{metric_name}': df_metric_values, 
               'prediction key': df_predicted_keys}
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
    

    
            
            

    