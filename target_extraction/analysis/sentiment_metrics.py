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
7. ignore_label_differences -- If True then the ValueError will not be 
   raised if the predicted sentiment values are not in the true 
   sentiment values. See :py:func:`get_labels` for more details.

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
import functools
from typing import Union, Optional, Callable, Tuple, List, Any
import statistics

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

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
                                        bool, Optional[int], bool], 
                                       Union[float, np.ndarray]]
                        ) -> Callable[[TargetTextCollection, str, str, bool,
                                       bool, Optional[int], bool],
                                      Union[float, np.ndarray]]:
    '''
    Decorator for the metric functions within this module. Will raise any of 
    the Errors stated above in the module documentation before the metric 
    functions is called.
    '''
    @functools.wraps(func)
    def wrapper(target_collection: TargetTextCollection, 
                true_sentiment_key: str, predicted_sentiment_key: str, 
                average: bool, array_scores: bool, 
                assert_number_labels: Optional[int] = None,
                ignore_label_differences: bool = True
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
                                     'Target within the collection is different'
                                     f'. This TargetText could have no targets'
                                     ' within the collection thus this error '
                                     'will be raise. TargetText that has an error'
                                     f' {target_object}\nThe number of predcitions'
                                     f' that this object should have: {total_number_model_predictions}')
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
                    assert_number_labels, ignore_label_differences)
    return wrapper

def get_labels(target_collection: TargetTextCollection, 
               true_sentiment_key: str, predicted_sentiment_key: str,
               labels_per_text: bool = False,
               ignore_label_differences: bool = True
               ) -> Tuple[Union[List[Any], List[List[Any]]], 
                          Union[List[List[Any]], List[List[List[Any]]]]]:
    '''
    :param target_collection: Collection of targets that have true and predicted 
                              sentiment values.
    :param true_sentiment_key: Key that contains the true sentiment scores 
                               for each target in the TargetTextCollection
    :param predicted_sentiment_key: Key that contains the predicted sentiment   
                                    scores for each target in the 
                                    TargetTextCollection. It assumes that the 
                                    predictions is a List of List where the 
                                    outer list are the number of model runs and 
                                    the inner list is the number of targets to 
                                    predict for, the the second Tuple of the 
                                    example return for an example of this.
    :param labels_per_text: If True instead of returning a List[Any] it will
                            return a List[List[Any]] where in the inner list 
                            represents the predictions per text rather than in 
                            the normal case where it is all predictions ignoring 
                            which text they came from.
    :param ignore_label_differences: If True then the ValueError will not be 
                                     raised if the predicted sentiment values 
                                     are not in the true sentiment values.
    :returns: A tuple of 1; true sentiment value 2; predicted sentiment values. 
              where the predicted sentiment values is a list of predicted 
              sentiment value, one for each models predictions. 
              See `Example of return 2` for an example of what this means 
              where in that example there are two texts/sentences.
    :raises ValueError: If the number of predicted sentiment values are not 
                        equal to the number true sentiment values.
    :raises ValueError: If the labels in the predicted sentiment values are not 
                        in the true sentiment values.
    :Example of return 1: (['pos', 'neg', 'neu'], [['neg', 'pos', 'neu'], 
                             ['neu', 'pos', 'neu']])
    :Example of return 2: ([['pos'], ['neg', 'neu']], [[['neg'], ['pos', 'neu']], 
                           [['neu'], ['pos', 'neu']]])
    ''' 
    all_predicted_values: List[List[Any]] = []
    all_true_values: List[Any] = []
    for target_object in target_collection.values():
        target_object: TargetText
        
        true_values = target_object[true_sentiment_key]
        if labels_per_text:
            all_true_values.append(true_values)
        else:
            all_true_values.extend(true_values)

        predicted_values_lists = target_object[predicted_sentiment_key]
        # Create a list per model predictions
        if all_predicted_values == []:
            for _ in predicted_values_lists:
                all_predicted_values.append([])
        for index, prediction_list in enumerate(predicted_values_lists):
            if labels_per_text:
                all_predicted_values[index].append(prediction_list)
            else:
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
    if labels_per_text:
        unique_true_values = set([value for values in all_true_values for value in values])
    else:
        unique_true_values = set(all_true_values)
    for prediction_list in all_predicted_values:
        if labels_per_text:
            unique_predicted_values = set([value for values in prediction_list for value in values])
        else:
            unique_predicted_values = set(prediction_list)
        if (unique_predicted_values.difference(unique_true_values) and 
            not ignore_label_differences):
            raise ValueError(f'Values in the predicted sentiment are not in the'
                             ' True sentiment values. Values in predicted '
                             f'{unique_predicted_values}, values in True '
                             f'{unique_true_values}')
    return (all_true_values, all_predicted_values)


@metric_error_checks
def accuracy(target_collection: TargetTextCollection, 
             true_sentiment_key: str, predicted_sentiment_key: str, 
             average: bool, array_scores: bool, 
             assert_number_labels: Optional[int] = None,
             ignore_label_differences: bool = True
             ) -> Union[float, List[float]]:
    '''
    :param ignore_label_differences: See :py:func:`get_labels`

    Accuracy score. Description at top of module explains arguments.
    '''
    true_values, predicted_values_list = get_labels(target_collection, 
                                                    true_sentiment_key, 
                                                    predicted_sentiment_key,
                                                    ignore_label_differences=ignore_label_differences)
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

@metric_error_checks
def macro_f1(target_collection: TargetTextCollection, 
             true_sentiment_key: str, predicted_sentiment_key: str, 
             average: bool, array_scores: bool, 
             assert_number_labels: Optional[int] = None,
             ignore_label_differences: bool = True
             ) -> Union[float, List[float]]:
    '''
    :param ignore_label_differences: See :py:func:`get_labels`

    Macro F1 score. Description at top of module explains arguments.
    '''
    true_values, predicted_values_list = get_labels(target_collection, 
                                                    true_sentiment_key, 
                                                    predicted_sentiment_key,
                                                    ignore_label_differences=ignore_label_differences)
    scores: List[float] = []
    for predicted_values in predicted_values_list:
        scores.append(f1_score(true_values, predicted_values, average='macro'))
    if average:
        return statistics.mean(scores)
    elif array_scores:
        return scores
    else:
        assert 1 == len(scores)
        return scores[0]

@metric_error_checks
def strict_text_accuracy(target_collection: TargetTextCollection, 
                         true_sentiment_key: str, predicted_sentiment_key: str, 
                         average: bool, array_scores: bool, 
                         assert_number_labels: Optional[int] = None,
                         ignore_label_differences: bool = True
                         ) -> Union[float, List[float]]:
    '''
    This is performed at the text/sentence level where a sample is not denoted 
    as one target but as all targets within a text. A sample is correct if all
    targets within the text have been predicted correctly. This will return the 
    average of the correct predictions. Strict Text ACcuracy also known as STAC.

    This metric also assumes that all the texts within the `target_collection`
    also contains at least one target. If it does not a ValueError will be raised.

    :param ignore_label_differences: See :py:func:`get_labels`
    '''
    true_values, predicted_values_list = get_labels(target_collection, 
                                                    true_sentiment_key, 
                                                    predicted_sentiment_key, 
                                                    labels_per_text=True,
                                                    ignore_label_differences=ignore_label_differences)
    true_values: List[List[Any]]
    predicted_values_list: List[List[List[Any]]]
    
    scores: List[float] = []
    num_texts = float(len(true_values))
    for predicted_values in predicted_values_list:
        score = 0
        for true_value, predicted_value in zip(true_values, predicted_values):
            if true_value == predicted_value:
                score += 1
        scores.append(float(score) / num_texts)
    if average:
        return statistics.mean(scores)
    elif array_scores:
        return scores
    else:
        assert 1 == len(scores)
        return scores[0]