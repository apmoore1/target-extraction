from typing import Optional, Callable, Union, List

import pandas as pd

from target_extraction.data_types import TargetTextCollection

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