'''
This module is dedictaed to creating new TargetTextCollections that are 
subsamples of the original(s) that will allow the user to analysis the 
data with respect to some certain property.
'''
from typing import List, Callable, Dict

from target_extraction.data_types import TargetTextCollection, TargetText

def count_error_key_occurence(dataset: TargetTextCollection, 
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
    '''
    count = 0
    for target_data in dataset.values():
        # Will raise a key error if the TargetText object does not have that 
        # error_key
        target_data._key_error(error_key)
        count += sum(target_data[error_key])
    return count

def _pre_post_subsampling(test_dataset: TargetTextCollection, 
                          train_dataset: TargetTextCollection, 
                          lower: bool, error_key: str,
                          error_func: Callable[[TargetText, Dict[str, List[str]], 
                                                Dict[str, List[str]]], bool]
                          ) -> TargetTextCollection:
    train_target_sentiments = train_dataset.target_sentiments(lower=lower, 
                                                              unique_sentiment=True)
    test_target_sentiments = test_dataset.target_sentiments(lower=lower, 
                                                            unique_sentiment=True)

    for target_data in test_dataset.values():
        test_targets = target_data['targets']
        error_values: List[int] = []
        if test_targets is None:
            target_data[error_key] = error_values
            continue
        for target in test_targets:
            if lower:
                target = target.lower()
            if error_func(target, train_target_sentiments, 
                          test_target_sentiments):
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

    :Note: If the target is None then an empty list is returned for that 
           `same_one_sentiment` key value.

    :param test_dataset: Test dataset to sub-sample
    :param train_dataset: Train dataset to reference
    :param lower: Whether to lower case the target words
    :returns: The test dataset but with each TargetText object containing a 
              `same_one_sentiment` key and associated list of values.
    '''
    
    def error_func(target: TargetText, 
                   train_target_sentiments: Dict[str, List[str]],
                   test_target_sentiments: Dict[str, List[str]]) -> bool:
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
    not just poisitive or not just negative) in the train and test 
    where as the 0 means it does not.

    :Note: If the target is None then an empty list is returned for that 
           `same_multi_sentiment` key value.

    :param test_dataset: Test dataset to sub-sample
    :param train_dataset: Train dataset to reference
    :param lower: Whether to lower case the target words
    :returns: The test dataset but with each TargetText object containing a 
              `same_multi_sentiment` key and associated list of values.
    '''
    def error_func(target: TargetText, 
                   train_target_sentiments: Dict[str, List[str]],
                   test_target_sentiments: Dict[str, List[str]]) -> bool:
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

    :Note: If the target is None then an empty list is returned for that 
           `similar_sentiment` key value.

    :param test_dataset: Test dataset to sub-sample
    :param train_dataset: Train dataset to reference
    :param lower: Whether to lower case the target words
    :returns: The test dataset but with each TargetText object containing a 
              `similar_sentiment` key and associated list of values.
    '''
    def error_func(target: TargetText, 
                   train_target_sentiments: Dict[str, List[str]],
                   test_target_sentiments: Dict[str, List[str]]) -> bool:
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

    :Note: If the target is None then an empty list is returned for that 
           `different_sentiment` key value.

    :param test_dataset: Test dataset to sub-sample
    :param train_dataset: Train dataset to reference
    :param lower: Whether to lower case the target words
    :returns: The test dataset but with each TargetText object containing a 
              `different_sentiment` key and associated list of values.
    '''
    def error_func(target: TargetText, 
                   train_target_sentiments: Dict[str, List[str]],
                   test_target_sentiments: Dict[str, List[str]]) -> bool:
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

    :Note: If the target is None then an empty list is returned for that 
           `unknown_targets` key value.

    :param test_dataset: Test dataset to sub-sample
    :param train_dataset: Train dataset to reference
    :param lower: Whether to lower case the target words
    :returns: The test dataset but with each TargetText object containing a 
              `unknown_targets` key and associated list of values.
    '''
    def error_func(target: TargetText, 
                   train_target_sentiments: Dict[str, List[str]],
                   test_target_sentiments: Dict[str, List[str]]) -> bool:
        if (target in train_target_sentiments and 
            target in test_target_sentiments):
            return False
        return True    
    
    return _pre_post_subsampling(test_dataset, train_dataset, lower, 
                                 'unknown_targets', error_func)

def distinct_sentiment(dataset: TargetTextCollection) -> TargetTextCollection:
    '''
    :param dataset: The dataset to add the distinct sentiment labels to
    :returns: The same dataset but with each TargetText object containing a 
              `distinct_sentiment` key and associated number of distinct 
              sentiments that are in that TargetText object per target.

    :Example: Given a TargetTextCollection that contains a single TargetText 
              object that has three targets where the first two have the label 
              positive and the last is negative it will add the 
              `distinct_sentiment` key to the TargetText object with the
              following value [2,2,2] as there are two unique/distinct 
              sentiments in that TargetText object.
    '''
    for target_data in dataset.values():
        target_sentiments = target_data['target_sentiments']
        num_unique_sentiments = len(set(target_sentiments))
        num_targets = len(target_sentiments)
        target_data['distinct_sentiment'] = [num_unique_sentiments 
                                             for _ in range(num_targets)]
    return dataset