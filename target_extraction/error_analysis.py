'''
This module is dedicated to creating new TargetTextCollections that are 
subsamples of the original(s) that will allow the user to analysis the 
data with respect to some certain property.
'''
import copy
from collections import defaultdict
from typing import List, Callable, Dict, Union, Optional, Any, Tuple

from target_extraction.data_types import TargetTextCollection, TargetText

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
                                        error_key: str, 
                                        associated_keys: List[str]
                                        ) -> TargetTextCollection:
    '''
    :param dataset: The dataset that contains error analysis key which are 
                    one hot encoding of whether a target is in that 
                    error analysis class or not. Example function that 
                    produces these error keys are 
                    :func:`target_extraction.error_analysis.same_one_sentiment`
    :param error_key: Name of the error key e.g. `same_one_sentiment`
    :param associated_keys: The keys that are associated to the target that 
                            must be kept and are linked to that target. E.g. 
                            `target_sentiments`, `targets`, and `spans`.
    :returns: A new TargetTextCollection that contains only those targets and 
              relevant `associated_keys` within the TargetText's that the
              error analysis key were `True` (1 in the one hot encoding). 
              This could mean that some TargetText's will no longer exist.
    :raises KeyError: If the `error_key` or one or more of the `associated_keys` 
                      does not exist in one or more of the TargetText objects 
                      within the `dataset`
    '''
    reduced_collection = TargetTextCollection()
    key_check_list = [error_key, *associated_keys]
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
        for index, value in enumerate(target_data[error_key]):
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
        new_target_object = TargetText(**new_target_object)
        reduced_collection.add(new_target_object)
    return reduced_collection

def _pre_post_subsampling(test_dataset: TargetTextCollection, 
                          train_dataset: TargetTextCollection, 
                          lower: bool, error_key: str,
                          error_func: Callable[[TargetText, Dict[str, List[str]], 
                                                Dict[str, List[str]], 
                                                Union[int, str]], bool],
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
                          test_target_sentiments, target_sentiment):
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
    
    def error_func(target: TargetText, 
                   train_target_sentiments: Dict[str, List[str]],
                   test_target_sentiments: Dict[str, List[str]],
                   target_sentiment: Union[str, int]) -> bool:
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
    def error_func(target: TargetText, 
                   train_target_sentiments: Dict[str, List[str]],
                   test_target_sentiments: Dict[str, List[str]],
                   target_sentiment: Union[str, int]) -> bool:
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
    def error_func(target: TargetText, 
                   train_target_sentiments: Dict[str, List[str]],
                   test_target_sentiments: Dict[str, List[str]],
                   target_sentiment: Union[str, int]) -> bool:
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
    def error_func(target: TargetText, 
                   train_target_sentiments: Dict[str, List[str]],
                   test_target_sentiments: Dict[str, List[str]],
                   target_sentiment: Union[str, int]) -> bool:
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
    def error_func(target: TargetText, 
                   train_target_sentiments: Dict[str, List[str]],
                   test_target_sentiments: Dict[str, List[str]],
                   target_sentiment: Union[str, int]) -> bool:
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
    def error_func(target: TargetText, 
                   train_target_sentiments: Dict[str, List[str]],
                   test_target_sentiments: Dict[str, List[str]],
                   target_sentiment: Union[str, int]) -> bool:
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
    def error_func(target: TargetText, 
                   train_target_sentiments: Dict[str, List[str]],
                   test_target_sentiments: Dict[str, List[str]],
                   target_sentiment: Union[str, int]) -> bool:
        if (target in train_target_sentiments and 
            target in test_target_sentiments):
            train_sentiments = train_target_sentiments[target]
            if target_sentiment not in train_sentiments:
                return True
        return False

    return _pre_post_subsampling(test_dataset, train_dataset, lower, 
                                 'unknown_sentiment_known_target', error_func)

def distinct_sentiment(dataset: TargetTextCollection, 
                       separate_labels: bool = False) -> TargetTextCollection:
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
                            sentiment in the collection {2, 4} and the current 
                            TargetText contain 2 targets with 2 distinct 
                            sentiments then it will contain the following keys 
                            and values: `distinct_sentiment_2`: [1,1] and 
                            `distinct_sentiment_4`: [0,0].
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
        for unique_ds in dataset.unique_distinct_sentiments():
            ds_keys.append(f'distinct_sentiment_{unique_ds}')
        if len(ds_keys) == 0:
            raise ValueError('There are no Distinct sentiments/sentiments '
                             'in this collection')
        
    for target_data in dataset.values():
        target_sentiments = target_data['target_sentiments']
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
    def get_third_n(third_sample_count: int, n_relation_target: Dict[int, str],
                    target_sample_count: Dict[str, int]) -> Tuple[int, int]:
        start = True
        start_n = 0
        end_n = 0
        count = 0
        for n_relation, targets in n_relation_target.items():
            if start:
                start = False
                start_n = n_relation
            for target in targets:
                count += target_sample_count[target]
            if count >= third_sample_count:
                end_n = n_relation
                break
        else:
            end_n = n_relation
        if start_n == 0 or end_n == 0:
            raise ValueError('The start nor end can be zero')
        return (start_n, end_n) 

    def error_func(target: TargetText, 
                   ignore: Dict[str, int],
                   filtered_test: Dict[str, int],
                   target_sentiment: Union[str, int]) -> bool:
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
                               target in n_relation_test_target.keys()])
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
        return (return_n_values, n_ranges)
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
    def error_func(target: TargetText, 
                   ignore: Dict[str, int],
                   filtered_test: Dict[str, int],
                   target_sentiment: Union[str, int]) -> bool:
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
    for target_object in collection.values():
        target_object: TargetText
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
        new_target_objects.append(TargetText(**new_target_object_dict))
    return TargetTextCollection(new_target_objects)