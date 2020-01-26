'''
This module is dedicated to creating new TargetTextCollections that are 
subsamples of the original(s) that will allow the user to analysis the 
data with respect to some certain property.
'''
import copy
from collections import defaultdict, Counter
from typing import List, Callable, Dict, Union, Optional, Any, Tuple, Iterable
from multiprocessing import Pool
import math

import pandas as pd

from target_extraction.data_types import TargetTextCollection, TargetText

PLOT_SUBSET_ABBREVIATION = {'distinct_sentiment_1' : 'DS1', 
                            'distinct_sentiment_2': 'DS2', 
                            'distinct_sentiment_3': 'DS3',
                            '1-target': '1', 'low-targets': 'Low', 
                            'med-targets': 'Med', 'high-targets': 'High',
                            '1-TSSR': '1', '1-multi-TSSR': '1-Multi', 
                            'low-TSSR': 'Low', 'high-TSSR': 'High',
                            'unknown_sentiment_known_target': 'USKT', 
                            'unknown_targets': 'UT', 
                            'known_sentiment_known_target': 'KSKT',
                            'zero-shot': 'Zero', 'low-shot': 'Low', 
                            'med-shot': 'Med', 'high-shot': 'High'}

ERROR_SPLIT_SUBSET_NAMES = {'DS': ['distinct_sentiment_1', 'distinct_sentiment_2', 
                                   'distinct_sentiment_3'],
                            'NT': ['1-target', 'low-targets', 'med-targets', 
                                   'high-targets'],
                            'TSSR': ['1-TSSR', '1-multi-TSSR', 'low-TSSR', 
                                     'high-TSSR'],
                            'TSR': ['unknown_sentiment_known_target', 
                                    'unknown_targets', 
                                    'known_sentiment_known_target'],
                            'n-shot': ['zero-shot', 'low-shot', 
                                       'med-shot', 'high-shot']}
SUBSET_NAMES_ERROR_SPLIT = {}

class NoSamplesError(Exception):
   '''
   If there are or will be no samples within a Dataset or subset.
   '''
   def __init__(self, error_string: str) -> None:
        '''
        :param error_string: Error string to generate on Error
        '''
        super().__init__(error_string)

def count_error_key_occurrence(dataset: TargetTextCollection, 
                               error_key: str) -> int:
    '''
    :param dataset: The dataset that contains error analysis key which are 
                    one hot encoding of whether a target is in that 
                    error analysis class or not. Example function that 
                    produces these error keys are 
                    :func:`target_extraction.error_analysis.same_one_sentiment`
    :param error_key: Name of the error key e.g. `same_one_sentiment`
    :returns: The number of targets within the dataset that are in that error
              class.
    :raises KeyError: If the `error_key` does not exist in one or more of the 
                      TargetText objects within the `dataset`
    '''
    count = 0
    for target_data in dataset.values():
        # Will raise a key error if the TargetText object does not have that 
        # error_key
        target_data._key_error(error_key)
        count += sum(target_data[error_key])
    return count

def reduce_collection_by_key_occurrence(dataset: TargetTextCollection, 
                                        error_key: Union[str, List[str]], 
                                        associated_keys: List[str]
                                        ) -> TargetTextCollection:
    '''
    :param dataset: The dataset that contains error analysis key which are 
                    one hot encoding of whether a target is in that 
                    error analysis class or not. Example function that 
                    produces these error keys are 
                    :func:`target_extraction.error_analysis.same_one_sentiment`
    :param error_key: Name of the error key e.g. `same_one_sentiment`. Or it can 
                      be a list of error keys for which this will reduce the 
                      collection so that it includes all samples that contain 
                      at least one of these error keys.
    :param associated_keys: The keys that are associated to the target that 
                            must be kept and are linked to that target. E.g. 
                            `target_sentiments`, `targets`, `spans`, and 
                            `subset error keys`.
    :returns: A new TargetTextCollection that contains only those targets and 
              relevant `associated_keys` within the TargetText's that the
              error analysis key(s) were `True` (1 in the one hot encoding). 
              This could mean that some TargetText's will no longer exist.
    :raises KeyError: If the `error_key` or one or more of the `associated_keys` 
                      does not exist in one or more of the TargetText objects 
                      within the `dataset`
    '''
    reduced_collection = []
    anonymised = False
    if isinstance(error_key, str):
        error_key = [error_key]
    error_keys: List[str] = error_key
    key_check_list = [*associated_keys] + error_keys
    for target_data in dataset.values():
        # Will raise a key error if the TargetText object does not have that 
        # error_key or any of the associated_keys
        for _key in key_check_list:
            target_data._key_error(_key)
        
        new_target_object = copy.deepcopy(dict(target_data))
        # remove the associated values
        for associated_key in associated_keys:
            del new_target_object[associated_key]

        skip_target = True
        error_key_values: List[int] = []
        for error_key in error_keys:
            error_values = target_data[error_key]
            if not error_key_values:
                error_key_values = [0] * len(error_values)
            else:
                error_key_error = ('Not all error keys are of the same length, '
                                   f'of which they should error keys: {error_keys}'
                                   f' TargetObject {target_data}')
                assert len(error_key_values) == len(error_values), error_key_error
            for index, value in enumerate(error_values):
                if value:
                    error_key_values[index] = value

        for index, value in enumerate(error_key_values):
            if value:
                skip_target = False
                for associated_key in associated_keys:
                    associated_value = target_data[associated_key][index]
                    if associated_key in new_target_object:
                        new_target_object[associated_key].append(associated_value)
                    else:
                        new_target_object[associated_key] = [associated_value]
        if skip_target:
            continue
        if 'text' not in new_target_object:
            new_target_object['text'] = None
        if new_target_object['text'] is None:
            anonymised = True
            new_target_object['anonymised'] = True
        new_target_object = TargetText(**new_target_object)
        reduced_collection.append(new_target_object)
    return TargetTextCollection(reduced_collection, anonymised=anonymised)

def swap_and_reduce(_collection: TargetTextCollection, 
                    subset_key: Union[str, List[str]],
                    true_sentiment_key: str,
                    prediction_keys: List[str]) -> TargetTextCollection:
    '''
    Furthermore the keys that will be reduced won't just be the `targets`, `spans`,
    `true_sentiment_key` and all `prediction_keys` but any error subset name from 
    within `PLOT_SUBSET_ABBREVIATION` that is in the TargetTexts in the collection.

    :param _collection: TargetTextCollection to reduce the samples based on the 
                        subset_key argument given.
    :param subset_key: Name of the error key e.g. `same_one_sentiment`. Or it can 
                       be a list of error keys for which this will reduce the 
                       collection so that it includes all samples that contain 
                       at least one of these error keys.
    :param true_sentiment: The key in each TargetText within the collection 
                           that contains the true sentiment labels.
    :param prediction_keys: The list of keys in each TargetText 
                            where each key contains a list of predicted sentiments.
                            These predicted sentiments are expected to be in a 
                            list of a list where the outer list defines the 
                            number of models trained e.g. number of model runs 
                            and the inner list is the length of the number of 
                            predictions required for that text/sentence.
    :returns: A collection that has been reduced based on the subset_key 
              argument. This is a helper function for the 
              `reduce_collection_by_key_occurrence` as this function ensure that 
              the predicted sentiment keys are changed before and after reducing 
              the collection so that they are processed properly as the predicted 
              sentiment labels are of shape (number of model runs, number of sentiments)
              where as all other lists in the TargetText are of (number of sentiments) 
              size. Furthermore if the reduction causes all any of the TargetText's
              in the collection to have no Targets then that TargetText will be 
              removed from the collection, thus you could have a collection of 
              zero.
    '''
    reduce_keys = ['targets', 'spans', true_sentiment_key] + prediction_keys
    target_sample = next(_collection.dict_iterator())
    # Adds the subset error keys to the reduce keys so that the reduction is 
    # performed correctly.
    for key in target_sample.keys():
        if key in PLOT_SUBSET_ABBREVIATION:
            reduce_keys.append(key)
    for prediction_key in prediction_keys:
        _collection = swap_list_dimensions(_collection, prediction_key)
    _collection = reduce_collection_by_key_occurrence(_collection, 
                                                        subset_key, 
                                                        reduce_keys)
    for prediction_key in prediction_keys:
        _collection = swap_list_dimensions(_collection, prediction_key)
    # Remove TargetTexts from the collection that do not contain a target anymore
    _collection = _collection.samples_with_targets()
    return _collection

def _pre_post_subsampling(test_dataset: TargetTextCollection, 
                          train_dataset: TargetTextCollection, 
                          lower: bool, error_key: str,
                          error_func: Callable[[str, Dict[str, List[str]], 
                                                Dict[str, List[str]], 
                                                Union[int, str], TargetText], bool],
                          train_dict: Optional[Dict[str, Any]] = None,
                          test_dict: Optional[Dict[str, Any]] = None
                          ) -> TargetTextCollection:
    train_target_sentiments = train_dataset.target_sentiments(lower=lower, 
                                                              unique_sentiment=True)
    test_target_sentiments = test_dataset.target_sentiments(lower=lower, 
                                                            unique_sentiment=True)
    if train_dict is not None:
        train_target_sentiments = train_dict
    if test_dict is not None:
        test_target_sentiments = test_dict

    for target_data in test_dataset.values():
        test_targets = target_data['targets']
        target_sentiments = target_data['target_sentiments']
        error_values: List[int] = []
        if test_targets is None:
            target_data[error_key] = error_values
            continue
        for target, target_sentiment in zip(test_targets, target_sentiments):
            if lower:
                target = target.lower()
            if error_func(target, train_target_sentiments, 
                          test_target_sentiments, target_sentiment,
                          target_data):
                error_values.append(1)
                continue
            error_values.append(0)
        assert_err_msg = 'This should not occur as the number of targets in '\
                         f'this TargetText object {target_data} should equal '\
                         f'the number of same_one_value integer list {error_values}'
        assert len(test_targets) == len(error_values), assert_err_msg
        target_data[error_key] = error_values
    return test_dataset

def same_one_sentiment(test_dataset: TargetTextCollection, 
                       train_dataset: TargetTextCollection, 
                       lower: bool = True) -> TargetTextCollection:
    '''
    Given a test and train dataset will return the same test dataset but 
    with an additional key `same_one_sentiment` for each TargetText object
    in the test collection. This `same_one_sentiment` key will contain a list 
    the same length as the number of targets in that TargetText object with 
    0's and 1's where a 1 represents the associated target has the same one 
    sentiment label in the train and test where as the 0 means it does not.

    :Note: If the TargetText object `targets` is None as in there are no
           targets in that sample then the `same_one_sentiment` key
           will be represented as an empty list

    :param test_dataset: Test dataset to sub-sample
    :param train_dataset: Train dataset to reference
    :param lower: Whether to lower case the target words
    :returns: The test dataset but with each TargetText object containing a 
              `same_one_sentiment` key and associated list of values.
    '''
    
    def error_func(target: str, 
                   train_target_sentiments: Dict[str, List[str]],
                   test_target_sentiments: Dict[str, List[str]],
                   target_sentiment: Union[str, int], 
                   target_data: TargetText) -> bool:
        if (target in train_target_sentiments and 
            target in test_target_sentiments):
            train_sentiments = train_target_sentiments[target]
            test_sentiments = test_target_sentiments[target]
            if (len(train_sentiments) == 1 and len(test_sentiments) == 1):
                if train_sentiments == test_sentiments:
                    return True
        return False    
    
    return _pre_post_subsampling(test_dataset, train_dataset, lower, 
                                 'same_one_sentiment', error_func)

def same_multi_sentiment(test_dataset: TargetTextCollection, 
                         train_dataset: TargetTextCollection, 
                         lower: bool = True) -> TargetTextCollection:
    '''
    Given a test and train dataset will return the same test dataset but 
    with an additional key `same_multi_sentiment` for each TargetText object
    in the test collection. This `same_multi_sentiment` key will contain a list 
    the same length as the number of targets in that TargetText object with 
    0's and 1's where a 1 represents the associated target has the same 
    sentiment labels (more than one sentiment label e.g. positive and negative
    not just positive or not just negative) in the train and test 
    where as the 0 means it does not.

    :Note: If the TargetText object `targets` is None as in there are no
           targets in that sample then the `same_multi_sentiment` key
           will be represented as an empty list

    :param test_dataset: Test dataset to sub-sample
    :param train_dataset: Train dataset to reference
    :param lower: Whether to lower case the target words
    :returns: The test dataset but with each TargetText object containing a 
              `same_multi_sentiment` key and associated list of values.
    '''
    def error_func(target: str, 
                   train_target_sentiments: Dict[str, List[str]],
                   test_target_sentiments: Dict[str, List[str]],
                   target_sentiment: Union[str, int], 
                   target_data: TargetText) -> bool:
        if (target in train_target_sentiments and 
            target in test_target_sentiments):
            train_sentiments = train_target_sentiments[target]
            test_sentiments = test_target_sentiments[target]
            if (len(train_sentiments) > 1 and len(test_sentiments) > 1):
                if train_sentiments == test_sentiments:
                    return True
        return False    
    
    return _pre_post_subsampling(test_dataset, train_dataset, lower, 
                                 'same_multi_sentiment', error_func)

def similar_sentiment(test_dataset: TargetTextCollection, 
                      train_dataset: TargetTextCollection, 
                      lower: bool = True) -> TargetTextCollection:
    '''
    Given a test and train dataset will return the same test dataset but 
    with an additional key `similar_sentiment` for each TargetText object
    in the test collection. This `similar_sentiment` key will contain a list 
    the same length as the number of targets in that TargetText object with 
    0's and 1's where a 1 represents the associated target has occured more 
    than once in the train or test sets with at least some overlap between the 
    test and train sentiments but not identical. E.g. the target `camera` 
    could occur with `positive` and `negative` sentiment in the test set and 
    only `negative` in the train set.

    :Note: If the TargetText object `targets` is None as in there are no
           targets in that sample then the `similar_sentiment` key
           will be represented as an empty list

    :param test_dataset: Test dataset to sub-sample
    :param train_dataset: Train dataset to reference
    :param lower: Whether to lower case the target words
    :returns: The test dataset but with each TargetText object containing a 
              `similar_sentiment` key and associated list of values.
    '''
    def error_func(target: str, 
                   train_target_sentiments: Dict[str, List[str]],
                   test_target_sentiments: Dict[str, List[str]],
                   target_sentiment: Union[str, int], 
                   target_data: TargetText) -> bool:
        if (target in train_target_sentiments and 
            target in test_target_sentiments):
            train_sentiments = train_target_sentiments[target]
            test_sentiments = test_target_sentiments[target]
            if (len(train_sentiments) > 1 or 
                len(test_sentiments) > 1):
                if train_sentiments == test_sentiments:
                    return False
                if test_sentiments.intersection(train_sentiments):
                    return True
        return False    
    
    return _pre_post_subsampling(test_dataset, train_dataset, lower, 
                                 'similar_sentiment', error_func)

def different_sentiment(test_dataset: TargetTextCollection, 
                        train_dataset: TargetTextCollection, 
                        lower: bool = True) -> TargetTextCollection:
    '''
    Given a test and train dataset will return the same test dataset but 
    with an additional key `different_sentiment` for each TargetText object
    in the test collection. This `different_sentiment` key will contain a list 
    the same length as the number of targets in that TargetText object with 
    0's and 1's where a 1 represents the associated target has no overlap 
    in sentiment labels between the test and the train.

    :Note: If the TargetText object `targets` is None as in there are no
           targets in that sample then the `different_sentiment` key
           will be represented as an empty list

    :param test_dataset: Test dataset to sub-sample
    :param train_dataset: Train dataset to reference
    :param lower: Whether to lower case the target words
    :returns: The test dataset but with each TargetText object containing a 
              `different_sentiment` key and associated list of values.
    '''
    def error_func(target: str, 
                   train_target_sentiments: Dict[str, List[str]],
                   test_target_sentiments: Dict[str, List[str]],
                   target_sentiment: Union[str, int], 
                   target_data: TargetText) -> bool:
        if (target in train_target_sentiments and 
            target in test_target_sentiments):
            train_sentiments = train_target_sentiments[target]
            test_sentiments = test_target_sentiments[target]
            if not test_sentiments.intersection(train_sentiments):
                return True
        return False    
    
    return _pre_post_subsampling(test_dataset, train_dataset, lower, 
                                 'different_sentiment', error_func)

def unknown_targets(test_dataset: TargetTextCollection, 
                    train_dataset: TargetTextCollection, 
                    lower: bool = True) -> TargetTextCollection:
    '''
    Given a test and train dataset will return the same test dataset but 
    with an additional key `unknown_targets` for each TargetText object
    in the test collection. This `unknown_targets` key will contain a list 
    the same length as the number of targets in that TargetText object with 
    0's and 1's where a 1 represents a target that exists in the test set 
    but not in the train.

    :Note: If the TargetText object `targets` is None as in there are no
           targets in that sample then the `unknown_targets` key
           will be represented as an empty list

    :param test_dataset: Test dataset to sub-sample
    :param train_dataset: Train dataset to reference
    :param lower: Whether to lower case the target words
    :returns: The test dataset but with each TargetText object containing a 
              `unknown_targets` key and associated list of values.
    '''
    def error_func(target: str, 
                   train_target_sentiments: Dict[str, List[str]],
                   test_target_sentiments: Dict[str, List[str]],
                   target_sentiment: Union[str, int], 
                   target_data: TargetText) -> bool:
        if (target in train_target_sentiments and 
            target in test_target_sentiments):
            return False
        return True    
    
    return _pre_post_subsampling(test_dataset, train_dataset, lower, 
                                 'unknown_targets', error_func)

def known_sentiment_known_target(test_dataset: TargetTextCollection, 
                                 train_dataset: TargetTextCollection, 
                                 lower: bool = True) -> TargetTextCollection:
    '''
    Given a test and train dataset will return the same test dataset but 
    with an additional key `known_sentiment_known_target` for each 
    TargetText object in the test collection. This 
    `known_sentiment_known_target` key will contain a list 
    the same length as the number of targets in that TargetText object with 
    0's and 1's where a 1 represents a target that exists in both train and 
    test where that target for that instance in the test set has a sentiment 
    that has been seen before in the training set for that target.

    :Note: If the TargetText object `targets` is None as in there are no
           targets in that sample then the `known_sentiment_known_target` key
           will be represented as an empty list

    :param test_dataset: Test dataset to sub-sample
    :param train_dataset: Train dataset to reference
    :param lower: Whether to lower case the target words
    :returns: The test dataset but with each TargetText object containing a 
              `known_sentiment_known_target` key and associated list of values.
    '''
    def error_func(target: str, 
                   train_target_sentiments: Dict[str, List[str]],
                   test_target_sentiments: Dict[str, List[str]],
                   target_sentiment: Union[str, int], 
                   target_data: TargetText) -> bool:
        if (target in train_target_sentiments and 
            target in test_target_sentiments):
            train_sentiments = train_target_sentiments[target]
            if target_sentiment in train_sentiments:
                return True
        return False

    return _pre_post_subsampling(test_dataset, train_dataset, lower, 
                                 'known_sentiment_known_target', error_func)

def unknown_sentiment_known_target(test_dataset: TargetTextCollection, 
                                   train_dataset: TargetTextCollection, 
                                   lower: bool = True) -> TargetTextCollection:
    '''
    Given a test and train dataset will return the same test dataset but 
    with an additional key `unknown_sentiment_known_target` for each 
    TargetText object in the test collection. This 
    `unknown_sentiment_known_target` key will contain a list 
    the same length as the number of targets in that TargetText object with 
    0's and 1's where a 1 represents a target that exists in both train and 
    test where that target for that instance in the test set has a sentiment 
    that has NOT been seen before in the training set for that target.

    :Note: If the TargetText object `targets` is None as in there are no
           targets in that sample then the `unknown_sentiment_known_target` key
           will be represented as an empty list

    :param test_dataset: Test dataset to sub-sample
    :param train_dataset: Train dataset to reference
    :param lower: Whether to lower case the target words
    :returns: The test dataset but with each TargetText object containing a 
              `unknown_sentiment_known_target` key and associated list of values.
    '''
    def error_func(target: str, 
                   train_target_sentiments: Dict[str, List[str]],
                   test_target_sentiments: Dict[str, List[str]],
                   target_sentiment: Union[str, int], 
                   target_data: TargetText) -> bool:
        if (target in train_target_sentiments and 
            target in test_target_sentiments):
            train_sentiments = train_target_sentiments[target]
            if target_sentiment not in train_sentiments:
                return True
        return False

    return _pre_post_subsampling(test_dataset, train_dataset, lower, 
                                 'unknown_sentiment_known_target', error_func)

def distinct_sentiment(dataset: TargetTextCollection, 
                       separate_labels: bool = False,
                       true_sentiment_key: str = 'target_sentiments'
                       ) -> TargetTextCollection:
    '''
    :param dataset: The dataset to add the distinct sentiment labels to
    :param separate_labels: If True instead of having one error key 
                            `distinct_sentiment` which contains a value of 
                            a list of the number of distinct sentiments. There
                            will be `n` error keys of the format 
                            `distinct_sentiment_n` where for each TargetText 
                            object each one will contain 0's apart from the 
                            `n` value which is the correct number of 
                            distinct sentiments. The value `n` is computed 
                            based on the number of unique distinct sentiments 
                            in the collection. Example if there are 2 distinct 
                            sentiment in the collection {2, 3} and the current 
                            TargetText contain 2 targets with 2 distinct 
                            sentiments then it will contain the following keys 
                            and values: `distinct_sentiment_2`: [1,1] and 
                            `distinct_sentiment_3`: [0,0].
    :param true_sentiment_key: Key in the `target_collection` targets that 
                               contains the true sentiment scores for each 
                               target in the TargetTextCollection.
    :returns: The same dataset but with each TargetText object containing a 
              `distinct_sentiment` or `distinct_sentiment_n` key(s) and 
              associated number of distinct sentiments that are in that 
              TargetText object per target.

    :Example: Given a TargetTextCollection that contains a single TargetText 
              object that has three targets where the first two have the label 
              positive and the last is negative it will add the 
              `distinct_sentiment` key to the TargetText object with the
              following value [2,2,2] as there are two unique/distinct 
              sentiments in that TargetText object.
    :raises ValueError: If separate_labels is True and there are no sentiment 
                        labels in the collection.
    '''
    # only used in the separate_labels case
    ds_keys = []
    if separate_labels:
        for unique_ds in dataset.unique_distinct_sentiments(true_sentiment_key):
            ds_keys.append(f'distinct_sentiment_{unique_ds}')
        if len(ds_keys) == 0:
            raise ValueError('There are no Distinct sentiments/sentiments '
                             'in this collection')
        
    for target_data in dataset.values():
        target_sentiments = target_data[true_sentiment_key]
        distinct_sentiments = []
        if target_sentiments is not None:
            num_unique_sentiments = len(set(target_sentiments))
            num_targets = len(target_sentiments)
            if separate_labels:
                distinct_sentiments = [1] * num_targets
            else:
                distinct_sentiments = [num_unique_sentiments 
                                    for _ in range(num_targets)]
        if separate_labels:
            for ds_key in ds_keys:
                target_data[ds_key] = [0] * len(distinct_sentiments)
            ds_key = f'distinct_sentiment_{num_unique_sentiments}'
            target_data[ds_key] = distinct_sentiments
        else:
            target_data['distinct_sentiment'] = distinct_sentiments
    return dataset

def n_shot_subsets(test_dataset: TargetTextCollection, 
                   train_dataset: TargetTextCollection, 
                   lower: bool = True, return_n_values: bool = False
                   ) -> Union[TargetTextCollection, 
                              Tuple[TargetTextCollection, List[Tuple[int, int]]]]:
    '''
    Given a test and train dataset will return the same test dataset but 
    with 4 additional keys denoted as `zero-shot`, `low-shot`, `med-shot`, and 
    `high-shot`. Each one of these represents a different set of *n* values 
    within the *n-shot* setup. The *n-shot* setup is the number of times the 
    target within the test sample has been seen in the training dataset. The 
    `zero-shot` subset contains all targets that have *n=0*. The low, med, and 
    high contain increasing values *n* respectively where each subset will 
    contain approximately 1/3 of all samples in the test dataset once the 
    `zero-shot` subset has been removed.

    :param test_dataset: Test dataset to sub-sample
    :param train_dataset: Train dataset to reference
    :param lower: Whether to lower case the target words
    :param return_n_values: If True will return a tuple containing 1. The 
                            TargetTextCollection with the new error keys and 
                            2. A list of tuples one for each of the error keys 
                            stating the values of *n* that the error keys 
                            are associated too.
    :returns: The test dataset but with each TargetText object containing a 
              `zero-shot`, `low-shot`, `med-shot`, and `high-shot` key and 
              associated list of values.
    '''
    def get_third_n(third_sample_count: int, 
                    n_relation_target: List[Tuple[int, List[str]]],
                    target_sample_count: Dict[str, int]) -> Tuple[int, int]:
        start = True
        start_n = 0
        end_n = 0
        count = 0
        for n_relation, targets in n_relation_target:
            if start:
                start = False
                start_n = n_relation
            for target in targets:
                count += target_sample_count[target]
            if count >= third_sample_count:
                end_n = n_relation
                break
            end_n = n_relation
        if start_n == 0 or end_n == 0:
            raise ValueError('The start nor end can be zero')
        return (start_n, end_n) 

    def error_func(target: str, 
                   train_target_sentiments: Dict[str, List[str]],
                   filtered_test: Dict[str, List[str]],
                   target_sentiment: Union[str, int], 
                   target_data: TargetText) -> bool:
        if target in filtered_test:
            return True
        return False
    
    # Get Target and associated count for both train and test datasets
    train_target_sentiments = train_dataset.target_sentiments(lower=lower, 
                                                              unique_sentiment=False)
    train_target_counts = {target: len(occurrences) 
                           for target, occurrences in train_target_sentiments.items()}
    test_target_sentiments = test_dataset.target_sentiments(lower=lower, 
                                                            unique_sentiment=False)
    test_target_counts = {target: len(occurrences) 
                          for target, occurrences in test_target_sentiments.items()}
    test_target_n_relation = {}
    # Does not cover zero shot target in n_relation_test_target
    n_relation_test_target = defaultdict(list)
    for target in test_target_counts.keys():
        if target not in train_target_counts:
            test_target_n_relation[target] = 0
        else:
            number_times_in_train = train_target_counts[target]
            test_target_n_relation[target] = number_times_in_train
            n_relation_test_target[number_times_in_train].append(target)
    zero_filter = {target: n_relation 
                   for target, n_relation in test_target_n_relation.items() 
                   if n_relation == 0}

    n_relation_test_target = sorted(n_relation_test_target.items(), key=lambda x: x[0])
    number_samples_left = sum([test_target_counts[target] for 
                               n_relation, targets in n_relation_test_target 
                               for target in targets])
    third_samples = int(number_samples_left / 3)
    filter_dict = {0: (zero_filter, (0,0))}
    for i in range(1, 4):
        start_n, end_n = get_third_n(third_samples, n_relation_test_target, 
                                     test_target_counts)
        n_range = list(range(start_n, end_n + 1))
        n_filter = {target: n_relation 
                    for target, n_relation in test_target_n_relation.items() 
                    if n_relation in n_range}
        filter_dict[i] = (n_filter, (start_n, end_n))
        n_relation_test_target = [(n_relation, target) for n_relation, target 
                                   in n_relation_test_target if n_relation not in n_range]
    filter_name_dict = {0: 'zero-shot', 1: 'low-shot', 
                        2: 'med-shot', 3: 'high-shot'}
    n_ranges = []
    for i in range(0, 4):
        filter_con, n_range = filter_dict[i]
        error_name = filter_name_dict[i]
        test_dataset = _pre_post_subsampling(test_dataset, train_dataset, lower, 
                                             error_name, error_func, train_dict={},
                                             test_dict=filter_con)
        n_ranges.append(n_range)
    if return_n_values:
        return (test_dataset, n_ranges)
    else:
        return test_dataset

def n_shot_targets(test_dataset: TargetTextCollection, 
                   train_dataset: TargetTextCollection, 
                   n_condition: Callable[[int], bool], error_name: str,
                   lower: bool = True) -> TargetTextCollection:
    '''
    Given a test and train dataset will return the same test dataset but 
    with an additional key denoted by `error_name` argument for each 
    TargetText object in the test collection. This 
    `error_name` key will contain a list 
    the same length as the number of targets in that TargetText object with 
    0's and 1's where a 1 represents a target that has meet the `n_condition`.
    This allows you to find the performance of n shot target learning where 
    the `n_condition` can allow you to find zero shot target (targets not seen
    in training but in test (also known as unknown targets)) or find >K shot 
    targets where targets have been seen K or more times.

    :Note: If the TargetText object `targets` is None as in there are no
           targets in that sample then the `error_name` argument key
           will be represented as an empty list

    :param test_dataset: Test dataset to sub-sample
    :param train_dataset: Train dataset to reference
    :param n_condition: A callable that denotes the number of times the target 
                        has to be seen in the training dataset to represent a 
                        1 in the error key. Example n_condition `lambda x: x>5` this 
                        means that a target has to be seen more than 5 times in
                        the training set.
    :param error_name: The name of the error key
    :param lower: Whether to lower case the target words
    :returns: The test dataset but with each TargetText object containing a 
              `unknown_sentiment_known_target` key and associated list of values.
    '''
    def error_func(target: str, 
                   train_target_sentiments: Dict[str, List[str]],
                   filtered_test: Dict[str, List[str]],
                   target_sentiment: Union[str, int], 
                   target_data: TargetText) -> bool:
        if target in filtered_test:
            return True
        return False
    
    # Get Target and associated count for both train and test datasets
    train_target_sentiments = train_dataset.target_sentiments(lower=lower, 
                                                              unique_sentiment=False)
    train_target_counts = {target: len(occurrences) 
                           for target, occurrences in train_target_sentiments.items()}
    test_target_sentiments = test_dataset.target_sentiments(lower=lower, 
                                                            unique_sentiment=False)
    test_target_counts = {target: len(occurrences) 
                          for target, occurrences in test_target_sentiments.items()}
    test_target_n_relation = {}
    for target in test_target_counts.keys():
        if target not in train_target_counts:
            test_target_n_relation[target] = 0
        else:
            number_times_in_train = train_target_counts[target]
            test_target_n_relation[target] = number_times_in_train
    # filter by the n_condition
    filter_test_target_n_relation = {target: train_occurrences 
                                     for target, train_occurrences in test_target_n_relation.items() 
                                     if n_condition(train_occurrences)}
    return _pre_post_subsampling(test_dataset, train_dataset, lower, 
                                 error_name, error_func, train_dict={},
                                 test_dict=filter_test_target_n_relation)

def num_targets_subset(dataset: TargetTextCollection, 
                       return_n_values: bool = False
                       ) -> Union[TargetTextCollection, 
                                  Tuple[TargetTextCollection, List[Tuple[int, int]]]]:
    '''
    Given a dataset it will add the following four error keys: `1-target`,
    `low-targets`, `med-targets`, `high-targets` to each target text object. 
    where each value associated to the error keys are a list of 1's or 0's 
    the length of the number of samples where 1 denotes the error key is True 
    and 0 otherwise. `1-target` is 1 when the target text object contains one 
    target. The others are based on the frequency of targets with respect to the 
    number of samples in the dataset where if the target is in the low 1/3 of 
    most frequent targets based on samples then it is 
    binned in the `low-targets`, middle 1/3 `med-targets` etc.

    :param dataset: The dataset to add the following four error keys: `1-target`,
                    `low-targets`, `med-targets`, `high-targets`.
    :param return_n_values: Whether to return the number of targets in the 
                            sentence are associated to the 4 error keys as a 
                            List of Tuples.
    :returns: The same dataset but with each TargetText object containing those 
              four stated error keys and associated list of 1's or 0's denoting 
              if the error key exists or not.
    '''
    def get_third_n(third_sample_count: int, 
                    num_targets_count: List[Tuple[int, int]]) -> Tuple[int, int]:
        start = True
        start_n = 0
        end_n = 0
        total_count = 0
        for num_targets, count in num_targets_count:
            if start:
                start = False
                start_n = num_targets
            total_count += count
            if total_count >= third_sample_count:
                end_n = num_targets
                break
            end_n = num_targets
        if start_n == 0 or end_n == 0:
            raise ValueError('The start nor end can be zero')
        return (start_n, end_n) 

    def error_func(target: str, 
                   train_target_sentiments: Dict[str, List[str]],
                   filtered_test: Dict[str, List[str]],
                   target_sentiment: Union[str, int], 
                   target_data: TargetText) -> bool:
        if len(target_data['targets']) in filtered_test:
            return True
        return False
    
    # Get num targets in sentence and associated counts
    num_targets_count = defaultdict(lambda: 0)
    for target in dataset.values():
        num_targets_in_text = len(target['targets'])
        num_targets_count[num_targets_in_text] += num_targets_in_text
    one_filter = {1: 0}
    number_samples_left = sum([count for num_targets, count in num_targets_count.items() if num_targets != 1])
    third_samples = int(number_samples_left / 3)
    filter_dict = {1: (one_filter, (1,1))}

    num_targets_count = sorted(num_targets_count.items(), key=lambda x: x[0])
    num_targets_count = [(num_target, count) for num_target, count in num_targets_count if num_target != 1]

    for i in range(2, 5):
        start_num, end_num = get_third_n(third_samples, num_targets_count)
        n_range = list(range(start_num, end_num + 1))
        n_filter = {n: 0 for n in n_range}
        filter_dict[i] = (n_filter, (start_num, end_num))
        num_targets_count = [(num_target, count) for num_target, count in num_targets_count if num_target not in n_range]
        number_samples_left = sum([count for num_targets, count in num_targets_count])
        if i == 2:
            third_samples = int(number_samples_left / 2)
            last_target = num_targets_count[-1]
            if (number_samples_left - last_target[1]) < third_samples:
                third_samples = number_samples_left - last_target[1]
        elif i == 3:
            third_samples = number_samples_left
    filter_name_dict = {1: '1-target', 2: 'low-targets', 
                        3: 'med-targets', 4: 'high-targets'}

    n_ranges = []
    for i in range(1, 5):
        filter_con, n_range = filter_dict[i]
        error_name = filter_name_dict[i]
        dataset = _pre_post_subsampling(dataset, dataset, True, 
                                        error_name, error_func, train_dict={},
                                        test_dict=filter_con)
        n_ranges.append(n_range)
    if return_n_values:
        return (dataset, n_ranges)
    else:
        return dataset

def tssr_target_value(target_data: TargetText, 
                      current_target_sentiment: Union[str, int],
                      subset_values: bool = False) -> float:
    '''
    Need to insert the TSSR value equation below:
    `
    `

    :param target_data: The TargetText object that contains the target 
                        associated to the `current_target_sentiment`
    :param current_target_sentiment: The sentiment value associated to the 
                                        target you want the TSSR value for.
    :param subset_values: If True it produceds to different values for when the
                          TSSR value is 1.0. It produces just 1.0 when there 
                          is only one target in the sentence and 
                          1.1 when there is more than one target in the sentence 
                          but all of them are 1.0 TSSR value i.e. the sentence 
                          only contains one sentiment.
    :returns: The TSSR value for a target within `target_data` with 
                `current_target_sentiment` sentiment value.
    '''
    number_targets = len(target_data['targets'])
    sentiment_values = [sentiment for sentiment in 
                        target_data['target_sentiments']]
    sentiment_counts = Counter(sentiment_values)
    current_target_senti_count = sentiment_counts[current_target_sentiment]
    tssr_value = current_target_senti_count / number_targets
    tssr_value = round(tssr_value, 2)
    if subset_values and tssr_value == 1.0:
        if len(target_data['targets']) > 1:
            tssr_value = 1.1
    return tssr_value

def tssr_subset(dataset: TargetTextCollection, 
                return_tssr_boundaries: bool = False
                ) -> Union[TargetTextCollection,
                           Tuple[TargetTextCollection, List[Tuple[float, float]]]]:
    '''
    Given a dataset it will add either `1-multi-TSSR`, `1-TSSR`, `high-TSSR` or `low-TSSR` 
    error keys to each target text object. Each value associated to the error 
    keys are a list of 1's or 0's the length of the number of samples where 1 
    denotes the error key is True and 0 otherwise. For more information on how 
    TSSR is calculated see 
    :py:func`target_extraction.error_analysis.tssr_target_value`. Once you know 
    what TSSR is: `1-TSSR` contains all of the targets that have a TSSR value of  
    1 but each one is the only target in the sentence, `1-multi-TSSR` contains 
    all of the targets that have a TSSR value of 1 and the sentence it comes 
    from contain more than one target. `high-TSSR` are targets that are in the 
    top 50% of the TSSR values for this dataset excluding the `1-TSSR` samples, 
    `low-TSSR` are the bottom 50% of the TSSR values.

    :param dataset: The dataset to add the continuos TSSR error keys too.
    :param return_tssr_boundaries: If to return the TSSR value boundaries for the 
                                   `1-TSSR`, `high-TSSR`, and `low-TSSR` 
                                   subsets. NOTE that `1-multi-TSSR` is not 
                                   in that list as it would have the same 
                                   TSSR value boundaries as `1-TSSR`.
    :returns: The same dataset but with each TargetText object containing the 
              TSSR subset error keys and associated list of 1's or 0's 
              denoting if the error key exists or not. The optional second 
              Tuple return are a list of the tssr boundaries.
    :raises NoSamplesError: If there are no samples within a subset.
    ''' 
    def error_func(target: str, 
                   train_target_sentiments: Dict[str, List[str]],
                   filtered_values: Dict[float, int],
                   target_sentiment: Union[str, int], 
                   target_data: TargetText) -> bool:
        tssr_value = tssr_target_value(target_data, target_sentiment, 
                                       subset_values=True)
        if tssr_value in filtered_values:
            return True
        return False

    # Possible values for the given dataset
    tssr_values_count = Counter()
    for target_data in dataset.values():
        tssr_values = []
        for target_sentiment in target_data['target_sentiments']:
            tssr_value = tssr_target_value(target_data, target_sentiment, 
                                           subset_values=True)
            tssr_values.append(tssr_value)
        tssr_values_count.update(tssr_values)
    # Split the TSSR values into low and high after removing the one values
    tssr_error_name_condition = {}
    tssr_error_name_condition['1-TSSR'] = {1: 1}
    tssr_error_name_condition['1-multi-TSSR'] = {1.1: 1}
    del tssr_values_count[1]
    del tssr_values_count[1.1]
    tssr_values_count = sorted(tssr_values_count.items(), key=lambda x: x[0], 
                               reverse=True)
    total_samples = sum([count for _, count in tssr_values_count])
    half_samples = int(total_samples / 2)
    high_count = 0
    high_dict = {}
    temp_tssr_values_count = dict(copy.deepcopy(tssr_values_count))
    for tssr_value, count in tssr_values_count:
        high_count += count
        if high_count > half_samples:
            break
        high_dict[tssr_value] = 1
        del temp_tssr_values_count[tssr_value]
    tssr_error_name_condition['high-TSSR'] = high_dict
    tssr_values_count = sorted(temp_tssr_values_count.items(), key=lambda x: x[0], 
                               reverse=True)
    low_dict = {tssr_value: 1 for tssr_value, _ in tssr_values_count}
    tssr_error_name_condition['low-TSSR'] = low_dict
    
    # Raise ValueError if not enough samples in any of the subsets.
    for values in tssr_error_name_condition.values():
        if len(values) == 0:
            enough_sample_err = ('Not enough samples in the '
                                 f'TargetTextCollection {dataset} to generate '
                                 'low, high, 1-TSSR, or 1-multi-TSSR subsets. '
                                 'The subsets within TSSR that were generated '
                                 f'{tssr_error_name_condition}')
            raise NoSamplesError(enough_sample_err)
    
    for error_name, tssr_values in tssr_error_name_condition.items():
        dataset = _pre_post_subsampling(dataset, dataset, True, 
                                        error_name, error_func, 
                                        train_dict={}, test_dict=tssr_values)
    
    if return_tssr_boundaries:
        high_tssr_values = sorted(high_dict.items(), key=lambda x: x[0], 
                                  reverse=True)
        high_values = (high_tssr_values[0][0], high_tssr_values[-1][0])
        low_tssr_values = sorted(low_dict.items(), key=lambda x: x[0], 
                                 reverse=True)
        low_values = (low_tssr_values[0][0], low_tssr_values[-1][0])
        return dataset, [(1.0, 1.0), high_values, low_values]
    return dataset

def tssr_raw(dataset: TargetTextCollection
             ) -> Tuple[TargetTextCollection, Dict[str, int]]:
    '''
    Given a dataset it will add a continuos number of error keys to each target text 
    object, where each key represents the TSSR value that the associated target 
    is within. Each value associated to the error keys are a list of 1's or 0's 
    the length of the number of samples where 1 denotes the error key is True 
    and 0 otherwise. See 
    :py:func`target_extraction.error_analysis.tssr_target_value` for an 
    explanation of how the TSSR value is calculated.

    :param dataset: The dataset to add the continuos TSSR error keys too.
    :returns: The same dataset but with each TargetText object containing the 
              continuos TSSR error keys and associated list of 1's or 0's 
              denoting if the error key exists or not. The dictionary contains 
              keys which are the TSSR values detected in the dataset and the 
              values are the number of targets that contain that TSSR value.
    ''' 
    def error_func(target: str, 
                   train_target_sentiments: Dict[str, List[str]],
                   filtered_values: Dict[float, int],
                   target_sentiment: Union[str, int], 
                   target_data: TargetText) -> bool:
        tssr_value = tssr_target_value(target_data, target_sentiment)
        if tssr_value in filtered_values:
            return True
        return False

    # Possible values for the given dataset
    tssr_values_count = Counter()
    for target_data in dataset.values():
        tssr_values = []
        for target_sentiment in target_data['target_sentiments']:
            tssr_values.append(tssr_target_value(target_data, target_sentiment))
        tssr_values_count.update(tssr_values)
    for tssr_value in tssr_values_count.keys():
        dataset = _pre_post_subsampling(dataset, dataset, True, 
                                        str(tssr_value), error_func, 
                                        train_dict={}, test_dict={tssr_value: 1})
    tssr_values_count = {str(tssr_value): count 
                         for tssr_value, count in tssr_values_count.items()}
    return dataset, tssr_values_count

def swap_list_dimensions(collection: TargetTextCollection, key: str
                                  ) -> TargetTextCollection:
    '''
    :param collection: The TargetTextCollection to change
    :param key: The key within the TargetText objects in the collection that 
                contain a List Value of shape (dim 1, dim 2)
    :returns: The collection but with the `key` values shape changed from 
              (dim 1, dim 2) to (dim 2, dim 1)

    :Note: This is a useful function when you need to change the predicted 
           values from shape (number runs, number targets) to 
           (number target, number runs) before using the following 
           function `reduce_collection_by_key_occurrence` where one of the 
           `associated_keys` are predicted values. It is required that the 
           sentiment predictions are of shape (number runs, number targets) 
           for the `sentiment_metrics` functions.
    '''
    new_target_objects = []
    anonymised = False
    for target_object in collection.values():
        target_object: TargetText
        target_object._key_error(key)
        new_target_object_dict = copy.deepcopy(dict(target_object))
        value_to_change = new_target_object_dict[key]
        new_value = []
        dim_1 = len(value_to_change)
        dim_2 = len(value_to_change[0])
        for index_1 in range(dim_1):
            for index_2 in range(dim_2):
                if index_1 == 0:
                    new_value.append([])
                new_value[index_2].append(value_to_change[index_1][index_2])
        new_target_object_dict[key] = new_value
        if 'text' not in new_target_object_dict:
            new_target_object_dict['text'] = None
        if new_target_object_dict['text'] is None:
            anonymised = True
            new_target_object_dict['anonymised'] = True
        new_target_objects.append(TargetText(**new_target_object_dict))
    return TargetTextCollection(new_target_objects, anonymised=anonymised)

def error_analysis_wrapper(error_function_name: str
                           ) -> Callable[[TargetTextCollection, 
                                          TargetTextCollection, 
                                          bool], TargetTextCollection]:
    '''
    To get a list of all possible function names easily use the `keys` of 
    `target_extraction.analysis.sentiment_error_analysis.ERROR_SPLIT_SUBSET_NAMES`
    dictionary.

    :param error_function_name: This can be either 1. `DS`, 2. `NT`, 3. `TSSR`, 
                                4. `n-shot`, 5. `TSR`
    :returns: The relevant error function where all error functions have the same 
                function signature where the input is:
                1. Train TargetTextCollection, 2. Test TargetTextCollection, and 
                3. Lower bool - whether to lower the targets.
                This then returns a the Test TargetTextCollection with the relevant 
                new keys. From the inputs only the Train and Lower are applicable 
                to `n-shot` and `TSR` error function due to them both being 
                global functions and relying on target text information.
    :raises ValueError: If the `error_function_name` is not one of the 5 listed. 
    '''
    def ds_wrapper(train_collection: TargetTextCollection, 
                    test_collection: TargetTextCollection, 
                    lower: bool) -> TargetTextCollection:
        '''
        :param train_collection: Not Applicable
        :param test_collection: The collection that is to be analysed
        :param lower: Lowering the target words Not Applicable
        :returns: A TargetTextCollection with the follwoing extra keys:
                `distinct_sentiment_1` `distinct_sentiment_2`, and
                `distinct_sentiment_3` 
        '''
        return distinct_sentiment(test_collection, separate_labels=True)

    def tssr_wrapper(train_collection: TargetTextCollection, 
                    test_collection: TargetTextCollection, 
                    lower: bool) -> TargetTextCollection:
        '''
        :param train_collection: Not Applicable
        :param test_collection: The collection that is to be analysed
        :param lower: Lowering the target words Not Applicable
        :returns: A TargetTextCollection with the follwoing extra keys:
                `1-TSSR` `1-multi-TSSR`, `low-TSSR`, and
                `high-TSSR` 
        '''
        return tssr_subset(test_collection, return_tssr_boundaries=False)

    def nt_wrapper(train_collection: TargetTextCollection, 
                    test_collection: TargetTextCollection, 
                    lower: bool) -> TargetTextCollection:
        '''
        :param train_collection: Not Applicable
        :param test_collection: The collection that is to be analysed
        :param lower: Lowering the target words Not Applicable
        :returns: A TargetTextCollection with the follwoing extra keys:
                `1-target` `low-targets`, `med-targets`, and
                `high-targets` 
        '''
        return num_targets_subset(test_collection, return_n_values=False)
    
    def tsr_wrapper(train_collection: TargetTextCollection, 
                    test_collection: TargetTextCollection, 
                    lower: bool) -> TargetTextCollection:
        '''
        :param train_collection: The collection to compare the Test collection with.
        :param test_collection: The collection that is to be analysed
        :param lower: Lowering the target words
        :returns: A TargetTextCollection with the follwoing extra keys:
                `unknown_sentiment_known_target` `unknown_targets`, and
                `known_sentiment_known_target` 
        '''
        subset_functions = [unknown_targets, unknown_sentiment_known_target, 
                            known_sentiment_known_target]
        for subset_function in subset_functions:
            test_collection = subset_function(test_collection, train_collection, 
                                              lower=lower)
        return test_collection
    
    def n_shot_wrapper(train_collection: TargetTextCollection, 
                        test_collection: TargetTextCollection, 
                        lower: bool) -> TargetTextCollection:
        '''
        :param train_collection: The collection to compare the Test collection with.
        :param test_collection: The collection that is to be analysed
        :param lower: Lowering the target words
        :returns: A TargetTextCollection with the follwoing extra keys:
                `zero-shot` `low-shot`, `med-shot`, and
                `high-shot` 
        '''
        return n_shot_subsets(test_collection, train_collection, lower=lower, 
                              return_n_values=False)
    
    acceptable_names = set(ERROR_SPLIT_SUBSET_NAMES.keys())
    if error_function_name not in acceptable_names:
        value_error = ('This error function name is not allowed '
                    f'{error_function_name}. These names are allowed '
                    f'{acceptable_names}')
        raise ValueError(value_error)
    
    if error_function_name == 'DS':
        return ds_wrapper
    elif error_function_name == 'TSSR':
        return tssr_wrapper
    elif error_function_name == 'NT':
        return nt_wrapper
    elif error_function_name == 'n-shot':
        return n_shot_wrapper
    elif error_function_name == 'TSR':
        return tsr_wrapper

def subset_metrics(target_collection: TargetTextCollection, 
                   subset_error_key: Union[str, List[str]],
                   metric_funcs: List[Callable[[TargetTextCollection, str, str, 
                                                bool, bool, Optional[int], bool], 
                                               Union[float, List[float]]]],
                   metric_names: List[str], 
                   metric_kwargs: Dict[str, Union[str,bool,int]],
                   include_dataset_size: bool = False
                   ) -> Dict[str, Union[List[float], float, int]]:
    '''
    This is most useful to find the metric score of an error subset
     
    :param target_collection: TargetTextCollection that contains the 
                              `subset_error_key` in each TargetText within the 
                              collection
    :param subset_error_key: The error key(s) to reduce the collection by. The samples 
                             left will only be those where the error key is True.
                             An example of a `subset_error_key` would be 
                             `zero-shot` from the :py:func:`n_shot_targets`. This 
                             can also be a list of keys e.g. 
                             [`zero-shot`, `low-shot`] from the 
                             :py:func:`n_shot_targets`.
    :param metric_funcs: A list of metric functions from 
                         `target_extraction.analysis.sentiment_metrics`. Example
                         metric function is 
                         :py:func:`target_extraction.analysis.sentiment_metrics.accuracy`
    :param metric_names: Names to give to each `metric_funcs`
    :param metric_kwargs: Keywords argument to give to the `metric_funcs` the only 
                          argument given is the first argument which will always 
                          be `target_collection`
    :param include_dataset_size: If True the returned dictionary will also include 
                                 a key `dataset size` that will contain an integer
                                 specifying the size of the dataset the metric(s) 
                                 was calculated on.
    :returns: A dictionary where the keys are the `metric_names` and the values 
              are the respective metric applied to the reduced/subsetted dataset.
              Thus if `average` in `metric_kwargs` is True then the return 
              will be Dict[str, float] where as if `array_scores` is True then 
              the return will be Dict[str, List[float]]. If no targets exist in 
              the collection through subsetting then the metric returned is 0.0
              or [0.0] if `array_scores` is true in `metric_kwargs`.
    '''
    true_sentiment_key = metric_kwargs['true_sentiment_key']
    predicted_sentiment_key = metric_kwargs['predicted_sentiment_key']
    target_collection = swap_and_reduce(target_collection, subset_error_key, 
                                        true_sentiment_key, 
                                        [predicted_sentiment_key])
    metric_name_score = {}
    if include_dataset_size:
        metric_name_score['dataset size'] = target_collection.number_targets()
    for metric_name, metric_func in zip(metric_names, metric_funcs):
        if not len(target_collection):
            if 'array_scores' in metric_kwargs:
                if metric_kwargs['array_scores']:
                    metric_score = [0.0]
                else:
                    metric_score = 0.0
            else:
                metric_score = 0.0
        else:
            metric_score = metric_func(target_collection, **metric_kwargs)
        metric_name_score[metric_name] = metric_score
    return metric_name_score


def _subset_and_score(arguments: Tuple[TargetTextCollection, str, 
                                       Callable[[TargetTextCollection, str, str, 
                                                 bool, bool, Optional[int], bool], 
                                                Union[float, List[float]]], 
                                       Dict[str, List[str]], bool]
                      ) -> Union[Tuple[List[float], List[int], List[str], List[str]],
                                 Tuple[List[float], List[int], List[str], List[str]], List[int]]:
    '''
    :param arguments: A tuple of 1. data, 2. subset name, 3. metric function,
                      4. Keyword arguments to give to the metric function, 5.
                      include_dataset_size whether or not to return the dataset
                      size that the metric was performed on. The metric function
                      should be one from 
                      `target_extraction.analysis.sentiment_metrics`. The only 
                      argument given to the metric function is the first argument 
                      (data collection) as that argument comes from `data`
                      argument. This function in affect subsets the `data` by 
                      the `subset name` and then provides the metric scores on
                      that subset.
    :returns: A tuple of 1. List of metric scores, 2. List of run numbers,
              3. List of subset names, 4. List of predictions keys, 5. IF
              include_dataset_size is True then the size of the subset dataset. The
              run number is essentially range(0,len(metric_scores)). The 
              list of subset names and prediction keys are the same value 
              as given just multipled by the number of runs. Thus all of the 
              list are of the same length.
    '''
    # Metirc name is only required if there will be more than one metric function 
    # which at the moment cannot happen but could be a useful future improvement
    metric_name = 'A Name'
    # un-pack arguments
    data_collection, subset_name, _metric_function, metric_kwargs, include_dataset_size = arguments
    prediction_key = metric_kwargs['predicted_sentiment_key']
    metric_values = subset_metrics(data_collection, subset_name, 
                                    [_metric_function], 
                                    [f'{metric_name}'], metric_kwargs, 
                                    include_dataset_size=include_dataset_size)
    metric_scores = metric_values[f'{metric_name}']
    pd_run_numbers = []
    pd_subset_names = []
    pd_prediction_keys = []
    for run_number, metric_score in enumerate(metric_scores):
        pd_run_numbers.append(run_number)
        pd_subset_names.append(subset_name)
        pd_prediction_keys.append(prediction_key)
    if include_dataset_size:
        subset_dataset_size = [metric_values['dataset size']] * len(metric_scores)
        return (metric_scores, pd_run_numbers, pd_subset_names, pd_prediction_keys,
                subset_dataset_size)
    else:
        return (metric_scores, pd_run_numbers, pd_subset_names, pd_prediction_keys)

def _subset_and_score_args_generator(target_collection: TargetTextCollection,
                                     prediction_keys: List[str],
                                     error_split_subset_names: Dict[str, List[str]],
                                     metric_func: Callable[[TargetTextCollection, str, str, 
                                                            bool, bool, Optional[int], bool], 
                                                           Union[float, List[float]]], 
                                     metric_kwargs: Dict[str, Union[str,int,bool]],
                                     include_dataset_size: bool = False
                                     ) -> Iterable[Tuple[TargetTextCollection, str, 
                                                         Callable[[TargetTextCollection, str, str, 
                                                                   bool, bool, Optional[int], bool], 
                                                                  Union[float, List[float]]], 
                                                         Dict[str, List[str]]]]:
    '''
    This is used to generate arguments to pass to the :py:func:`_subset_and_score`
    '''
    metric_kwargs_copy = copy.deepcopy(metric_kwargs)
    for prediction_key in prediction_keys:
        for error_split_name, subset_names in error_split_subset_names.items():
            for subset_name in subset_names:
                metric_kwargs_copy['predicted_sentiment_key'] = prediction_key
                yield (target_collection, subset_name, 
                       metric_func, metric_kwargs_copy,
                       include_dataset_size)

def _error_split_df(target_collection: TargetTextCollection, 
                    prediction_keys: List[str], true_sentiment_key: str, 
                    error_split_subset_names: Dict[str, List[str]],
                    metric_func: Callable[[TargetTextCollection, str, str, 
                                           bool, bool, Optional[int], bool], 
                                          Union[float, List[float]]],
                    metric_kwargs: Optional[Dict[str, Any]] = None,
                    num_cpus: Optional[int] = None,
                    collection_subsetting: Optional[List[List[str]]] = None,
                    include_dataset_size: bool = False,
                    table_format_return: bool = True
                    ) -> pd.DataFrame:
    '''
    This will require the `target_collection` having been pre-processed with the
    relevant error analysis functions within this module. A useful function to 
    perform the error analysis would be :py:func:`error_analysis_wrapper`

    :param target_collection: The collection where all TargetText's contain 
                              all `prediction_keys`, `true_sentiment_key`, and 
                              `subset_names` from the `error_split_subset_names`.
    :param prediction_keys: A list of keys that contain the predicted sentiment 
                            scores for each target in the TargetTextCollection
    :param true_sentiment_key: Key that contains the true sentiment scores 
                               for each target in the TargetTextCollection
    :param error_split_subset_names: The keys do not matter but the List values 
                                     must represent error subset names. An 
                                     example dictionary would be:
                                     `ERROR_SPLIT_SUBSET_NAMES`
    :param metric_func: A Metric function from
                        `target_extraction.analysis.sentiment_metrics`. Example
                         metric function is 
                         :py:func:`target_extraction.analysis.sentiment_metrics.accuracy`
    :param metric_kwargs: Keyword arguments to give to the `metric_func` the 
                          arguments given are: 1. `target_collection`, 2. `true_sentiment_key`,
                          3. `predicted_sentiment_key`, 4. `average`, and 
                          5. `array_scores`
    :param num_cpus: Number of cpus to use for multiprocessing. The task of 
                     subsetting and metric scoring is split down into one 
                     task and all tasks are then multiprocessed. This is also 
                     done in a Lazy fashion.   
    :param collection_subsetting: A list of lists where the outer list represents 
                                  the order of subsetting where as the inner list
                                  specifies the subset names to subset on. For example
                                  `[['1-TSSR', 'high-shot'], ['distinct_sentiment_2']]`
                                  would first subset the `test_collection` so that 
                                  only samples that are within ['1-TSSR', 'high-shot']
                                  subsets are in the collection and then it would 
                                  subset that collection further so that only 
                                  'distinct_sentiment_2' samples exist in the collection.
    :param include_dataset_size: The returned DataFrame will have two values the 
                                 metric associated with the error splits and the 
                                 size of the dataset from that subset.
    :param table_format_return: If this is True then the return will not be a 
                                pivot table but the raw dataframe. This can be 
                                more useful as a return format if `include_dataset_size`
                                is True. The columns for the DataFrame will be 
                                1. `prediction key`, 2. `run number`, 3. `subset names`
                                4. `Metric` and 5. Optional `Dataset Size`
    :returns: A dataframe that has a multi index of [`prediction key`, `run number`]
              and the columns are the error split subset names and the values are 
              the metric associated to those error splits given the prediction 
              key and the model run (run number). If any of the error subsets 
              do not have any targets that are relevant the accuracy will be 
              0.0
    '''
    pd_run_numbers = []
    pd_prediction_keys = []
    pd_subset_names = []
    pd_metric_values = []
    pd_dataset_size = []

    if metric_kwargs is None:
        metric_kwargs = {}
    metric_kwargs['average'] = False
    metric_kwargs['array_scores'] = True
    metric_kwargs['true_sentiment_key'] = true_sentiment_key

    if collection_subsetting is not None:
        for subset_names in collection_subsetting:
            target_collection = swap_and_reduce(target_collection, subset_names, 
                                                true_sentiment_key, prediction_keys)
    with Pool(num_cpus) as p:
        args_gen = _subset_and_score_args_generator(target_collection, 
                                                    prediction_keys, 
                                                    error_split_subset_names, 
                                                    metric_func, metric_kwargs,
                                                    include_dataset_size=include_dataset_size)
        results = p.imap(_subset_and_score, args_gen)

        for result in results:
            pd_metric_values.extend(result[0])
            pd_run_numbers.extend(result[1])
            pd_subset_names.extend(result[2])
            pd_prediction_keys.extend(result[3])
            if include_dataset_size:
                pd_dataset_size.extend(result[4])


    data_df = pd.DataFrame({'prediction key': pd_prediction_keys, 
                            'run number': pd_run_numbers, 
                            'subset names': pd_subset_names, 
                            'Metric': pd_metric_values})
    if include_dataset_size:
        data_df['Dataset Size'] = pd_dataset_size
        table = pd.pivot_table(data_df, values=['Metric', 'Dataset Size'], 
                               columns='subset names',
                               index=['prediction key', 'run number'])
    else:
        table = pd.pivot_table(data_df, values='Metric', columns='subset names',
                               index=['prediction key', 'run number'])
    if table_format_return:
        return table.fillna(value=0.0)
    return data_df

def error_split_df(train_collection: TargetTextCollection, 
                   test_collection: TargetTextCollection,
                   prediction_keys: List[str], true_sentiment_key: str, 
                   error_split_and_subset_names: Dict[str, List[str]],
                   metric_func: Callable[[TargetTextCollection, str, str, 
                                          bool, bool, Optional[int], bool], 
                                         Union[float, List[float]]],
                   metric_kwargs: Optional[Dict[str, Any]] = None,
                   num_cpus: Optional[int] = None,
                   lower_targets: bool = True,
                   collection_subsetting: Optional[List[List[str]]] = None,
                   include_dataset_size: bool = False,
                   table_format_return: bool = True
                   ) -> pd.DataFrame:
    '''
    This will perform `error_analysis_wrapper` over all `error_split_subset_names`
    keys and then returns the output from `_error_split_df`

    :param train_collection: The collection that was used to train the models 
                             that have made the predictions within 
                             `test_collection`
    :param test_collection: The collection where all TargetText's contain 
                            all `prediction_keys`, and `true_sentiment_key`.
    :param prediction_keys: A list of keys that contain the predicted sentiment 
                            scores for each target in the TargetTextCollection
    :param true_sentiment_key: Key that contains the true sentiment scores 
                               for each target in the TargetTextCollection
    :param error_split_and_subset_names: The keys do not matter but the List values 
                                         must represent error subset names. An 
                                         example dictionary would be:
                                         `ERROR_SPLIT_SUBSET_NAMES`
    :param metric_func: A Metric function from
                        `target_extraction.analysis.sentiment_metrics`. Example
                         metric function is 
                         :py:func:`target_extraction.analysis.sentiment_metrics.accuracy`
    :param metric_kwargs: Keyword arguments to give to the `metric_func` the 
                          arguments given are: 1. `target_collection`, 2. `true_sentiment_key`,
                          3. `predicted_sentiment_key`, 4. `average`, and 
                          5. `array_scores`
    :param num_cpus: Number of cpus to use for multiprocessing. The task of 
                     subsetting and metric scoring is split down into one 
                     task and all tasks are then multiprocessed. This is also 
                     done in a Lazy fashion.
    :param lower_targets: Whether or not the targets should be lowered during the 
                          `error_analysis_wrapper` function.   
    :param collection_subsetting: A list of lists where the outer list represents 
                                  the order of subsetting where as the inner list
                                  specifies the subset names to subset on. For example
                                  `[['1-TSSR', 'high-shot'], ['distinct_sentiment_2']]`
                                  would first subset the `test_collection` so that 
                                  only samples that are within ['1-TSSR', 'high-shot']
                                  subsets are in the collection and then it would 
                                  subset that collection further so that only 
                                  'distinct_sentiment_2' samples exist in the collection.
    :param include_dataset_size: The returned DataFrame will have two values the 
                                 metric associated with the error splits and the 
                                 size of the dataset from that subset.
    :param table_format_return: If this is True then the return will not be a 
                                pivot table but the raw dataframe. This can be 
                                more useful as a return format if `include_dataset_size`
                                is True. 
    :returns: A dataframe that has a multi index of [`prediction key`, `run number`]
              and the columns are the error split subset names and the values are 
              the metric associated to those error splits given the prediction 
              key and the model run (run number)
    '''
    for error_split, _ in error_split_and_subset_names.items():
        error_function = error_analysis_wrapper(error_split)
        test_collection = error_function(train_collection, test_collection, 
                                         lower=lower_targets)
    error_analysis_df = _error_split_df(test_collection, prediction_keys, 
                                        true_sentiment_key, 
                                        error_split_and_subset_names, 
                                        metric_func, metric_kwargs, num_cpus,
                                        collection_subsetting, 
                                        include_dataset_size=include_dataset_size,
                                        table_format_return=table_format_return)
    return error_analysis_df

def subset_name_to_error_split(subset_name: str) -> str:
    '''
    This in affect inverts the `ERROR_SPLIT_SUBSET_NAMES` dictionary and 
    returns the relevant error split name. It also initialises
    ERROR_SPLIT_SUBSET_NAMES.

    :param subset_name: Name of the subset you want to know which error split
                        it has come from.
    :returns: Associated error split name that the subset name has come from.
    '''
    if not SUBSET_NAMES_ERROR_SPLIT:
        for error_split, subset_names in ERROR_SPLIT_SUBSET_NAMES.items():
            for _subset_name in subset_names:
                SUBSET_NAMES_ERROR_SPLIT[_subset_name] = error_split
    return SUBSET_NAMES_ERROR_SPLIT[subset_name]