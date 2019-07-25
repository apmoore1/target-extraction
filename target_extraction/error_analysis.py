'''
This module is dedictaed to creating new TargetTextCollections that are 
subsamples of the original(s) that will allow the user to analysis the 
data with respect to some certain property.
'''
from typing import List

from target_extraction.data_types import TargetTextCollection

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
    train_target_sentiments = train_dataset.target_sentiments(lower=lower, 
                                                              unique_sentiment=True)
    test_target_sentiments = test_dataset.target_sentiments(lower=lower, 
                                                            unique_sentiment=True)
    for target_data in test_dataset.values():
        test_targets = target_data['targets']
        same_one_values: List[int] = []
        if test_targets is None:
            target_data['same_one_sentiment'] = same_one_values
            continue
        for target in test_targets:
            if lower:
                target = target.lower()
            if (target in train_target_sentiments and 
                target in test_target_sentiments):
                train_sentiments = train_target_sentiments[target]
                test_sentiments = test_target_sentiments[target]
                if (len(train_sentiments) == 1 and 
                    len(test_sentiments) == 1):
                    if train_sentiments == test_sentiments:
                        same_one_values.append(1)
                        continue
            same_one_values.append(0)
        assert_err_msg = 'This should not occur as the number of targets in '\
                         f'this TargetText object {target_data} should equal '\
                         f'the number of same_one_value integer list {same_one_values}'
        assert len(test_targets) == len(same_one_values), assert_err_msg
        target_data['same_one_sentiment'] = same_one_values
    return test_dataset
