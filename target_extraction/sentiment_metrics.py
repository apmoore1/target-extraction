'''
This module contains functions that expect a TargetTextCollection that contains
`target_sentiments` key that represent the true sentiment values and a prediction
key e.g. `sentiment_predictions`. Given these the function will return either a 
metric score e.g. Accuracy or a list of scores based on the arguments given 
to the function and if the `sentiment_predictions` key is an array of values.

Arguments for all functions in this module:

1. TargetTextCollection -- Contains the true and predicted sentiment scores
2. true_sentiment_key -- Key that contains the true sentiment scores 
   for each target in the TargetTextCollection
3. predicted_sentiment_key -- Key that contains the predicted sentiment scores  
   for each target in the TargetTextCollection
4. average -- If the predicting model was ran *N* times whether or not to 
   average the score over the *N* runs. Assumes array_scores is False.
5. array_scores -- If average is False and you a model that has predicted 
   *N* times then this will return the *N* scores, one for each run.
6. assert_number_labels -- Whether or not to assert this many number of unique  
   labels must exist in the true sentiment key. If this is None then the 
   assertion is not raised.

:raises ValueError: If the the prediction model has ran *N* times where 
                    *N>1* and `average` or `array_scores` are either both 
                    True or both False.
:raises ValueError: If the number of predictions made per target are 
                    different or zero. 
:raises ValueError: If only one set of model prediction exist then 
                    `average` and `array_scores` should be False.
:raises KeyError: If either the `true_sentiment_key` or 
                  `predicted_sentiment_key` does not exist.
:raises LabelError: If `assert_number_labels` is not None and the number of 
                    unique true labels does not equal the `assert_number_labels`
                    this is raised.
'''
from typing import Union, Optional, Callable, Tuple, List, Any
import statistics

import numpy as np
from sklearn.metrics import accuracy_score

from target_extraction.data_types import TargetTextCollection, TargetText

class LabelError(Exception):
   '''
   If the number of unique labels does not match your expected number of 
   unique labels.
   '''
   def __init__(self, true_number_unique_labels: int, 
                number_unique_labels_wanted: int) -> None:
        '''
        :param true_number_unique_labels: Number of unique labels that came 
                                          from the dataset
        :param number_unique_labels_wanted: Expected number of unique labels 
                                            that should be in the dataset.
        '''
        error_string = ('Number of unique labels in the dataset '
                        f'{true_number_unique_labels}. The number of unique '
                        'labels expected in the dataset '
                        f'{number_unique_labels_wanted}')
        super().__init__(error_string)

def metric_error_checks(func: Callable[[TargetTextCollection, str, str, bool, 
                                        bool, Optional[int]], 
                                       Union[float, np.ndarray]]
                        ) -> Callable[[TargetTextCollection, str, str, bool,
                                       bool, Optional[int]],
                                      Union[float, np.ndarray]]:
    '''
    Decorator for the metric functions within this module. Will raise any of 
    the Errors stated above in the module documentation before the metric 
    functions is called.
    '''
    def wrapper(target_collection: TargetTextCollection, 
                true_sentiment_key: str, predicted_sentiment_key: str, 
                average: bool, array_scores: bool, 
                assert_number_labels: Optional[int] = None
                ) -> Union[float, np.ndarray]:
        # Check that the TargetTextCollection contains both the true and 
        # predicted sentiment keys
        unique_label_set = set()
        total_number_model_predictions = 0
        for target_object in target_collection.values():
            target_object: TargetText
            target_object._key_error(true_sentiment_key)
            target_object._key_error(predicted_sentiment_key)
            for true_label in target_object[true_sentiment_key]:
                unique_label_set.add(true_label)
            # Cannot have inconsistent number of model predictions
            number_model_predictions = len(target_object[predicted_sentiment_key])
            if total_number_model_predictions == 0:
                total_number_model_predictions = number_model_predictions
            else:
                if total_number_model_predictions != number_model_predictions:
                    raise ValueError('The number of predictions made per '
                                     'Target within the collection is different')
        # Cannot have zero predictions
        if total_number_model_predictions == 0:
            raise ValueError('The number of predictions made per target are zero')

        # Perform the LabelError check
        if assert_number_labels is not None:
            number_unique_labels = len(unique_label_set)
            if number_unique_labels != assert_number_labels:
                raise LabelError(number_unique_labels, assert_number_labels)
        # If the dataset has one model prediction per target then average and 
        # array_scores should be False
        if number_model_predictions == 1:
            if average or array_scores:
                raise ValueError('When only one set of predictions per target'
                                 ' then `average` and `array_scores` have to '
                                 'be both False')
        else:
            if average == array_scores:
                raise ValueError('As the number of model predictions is > 1 '
                                 'then either `average` or `array_scores` have '
                                 'to be True but not both.') 
        return func(target_collection, true_sentiment_key, 
                    predicted_sentiment_key, average, array_scores, 
                    assert_number_labels)
    return wrapper

def get_labels(target_collection: TargetTextCollection, 
               true_sentiment_key: str, 
               predicted_sentiment_key: str) -> Tuple[List[Any], List[List[Any]]]:
    '''
    :param target_collection: Collection of targets that have true and predicted 
                              sentiment values.
    :param true_sentiment_key: Key that contains the true sentiment scores 
                               for each target in the TargetTextCollection
    :param predicted_sentiment_key: Key that contains the predicted sentiment   
                                    scores for each target in the 
                                    TargetTextCollection
    :returns: A tuple of 1; true sentiment value 2; predicted sentiment values. 
              where the predicted sentiment values is a list of predicted 
              sentiment value, one for each models predictions.
    :raises ValueError: If the number of predicted sentiment values are not 
                        equal to the number true sentiment values.
    :raises ValueError: If the labels in the predicted sentiment values are not 
                        in the true sentiment values.
    :Example of the return: (['pos', 'neg', 'neu'], [['neg', 'pos', 'neu'], 
                             ['neu', 'pos', 'neu']])
    ''' 
    all_predicted_values: List[List[Any]] = []
    all_true_values: List[Any] = []
    for target_object in target_collection.values():
        target_object: TargetText
        
        true_values = target_object[true_sentiment_key]
        all_true_values.extend(true_values)

        predicted_values_lists = target_object[predicted_sentiment_key]
        # Create a list per model predictions
        if all_predicted_values == []:
            for _ in predicted_values_lists:
                all_predicted_values.append([])
        for index, prediction_list in enumerate(predicted_values_lists):
            all_predicted_values[index].extend(prediction_list)
    # Check that the number of values in the predicted values is the same as 
    # the number of values in the true list
    true_number_values = len(all_true_values)
    for prediction_list in all_predicted_values:
        number_predictions = len(prediction_list)
        if number_predictions != true_number_values:
            raise ValueError(f'Number targets predicted {number_predictions}. '
                             f'Number of targets {true_number_values}. '
                             'These should be the same!')
    # Check that the values in True are the same as those in predicted
    unique_true_values = set(all_true_values)
    for prediction_list in all_predicted_values:
        unique_predicted_values = set(prediction_list)
        if unique_predicted_values.difference(unique_true_values):
            raise ValueError('Values in the predicted sentiment are not in the'
                             ' True sentiment values. Values in predicted '
                             f'{unique_predicted_values}, values in True '
                             f'{unique_true_values}')
    return (all_true_values, all_predicted_values)


@metric_error_checks
def accuracy(target_collection: TargetTextCollection, 
             true_sentiment_key: str, predicted_sentiment_key: str, 
             average: bool, array_scores: bool, 
             assert_number_labels: Optional[int] = None
             ) -> Union[float, List[float]]:
    '''
    Accuracy score. Description at top of module explains arguments.
    '''
    true_values, predicted_values_list = get_labels(target_collection, 
                                                    true_sentiment_key, 
                                                    predicted_sentiment_key)
    scores: List[float] = []
    for predicted_values in predicted_values_list:
        scores.append(accuracy_score(true_values, predicted_values))
    if average:
        return statistics.mean(scores)
    elif array_scores:
        return scores
    else:
        assert 1 == len(scores)
        return scores[0]

