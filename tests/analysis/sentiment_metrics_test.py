from typing import List, Tuple

import pytest

from target_extraction.data_types import TargetTextCollection, TargetText
from target_extraction.analysis.sentiment_metrics import get_labels, accuracy, LabelError
from target_extraction.analysis.sentiment_metrics import macro_f1

def passable_example(true_sentiment_key: str, 
                     predicted_sentiment_key: str
                     ) -> Tuple[TargetTextCollection, List[str], List[List[str]]]:
    example_1 = TargetText(text_id='1', text='some text')
    example_1[true_sentiment_key] = ['pos', 'neg']
    example_1[predicted_sentiment_key] = [['neg', 'pos']]
    example_2 = TargetText(text_id='2', text='some text')
    example_2[true_sentiment_key] = ['pos', 'neg', 'neu']
    example_2[predicted_sentiment_key] = [['neu', 'neg', 'pos']]

    true_labels = ['pos', 'neg', 'pos', 'neg', 'neu']
    pred_labels = [['neg', 'pos', 'neu', 'neg', 'pos']]
    return TargetTextCollection([example_1, example_2]), true_labels, pred_labels

def passable_wrong_example(true_sentiment_key: str, 
                           predicted_sentiment_key: str
                           ) -> Tuple[TargetTextCollection, List[str], List[List[str]]]:
    example_1 = TargetText(text_id='1', text='some text')
    example_1[true_sentiment_key] = ['pos', 'neg']
    example_1[predicted_sentiment_key] = [['neg', 'pos']]
    example_2 = TargetText(text_id='2', text='some text')
    example_2[true_sentiment_key] = ['pos', 'neg', 'neu']
    example_2[predicted_sentiment_key] = [['neu', 'neu', 'neg']]

    true_labels = ['pos', 'neg', 'pos', 'neg', 'neu']
    pred_labels = [['neg', 'pos', 'neu', 'neu', 'neg']]
    return TargetTextCollection([example_1, example_2]), true_labels, pred_labels

def passable_example_multiple_preds(true_sentiment_key: str, 
                                    predicted_sentiment_key: str
                                    ) -> Tuple[TargetTextCollection, List[str], List[List[str]]]:
    example_1 = TargetText(text_id='1', text='some text')
    example_1[true_sentiment_key] = ['pos', 'neg']
    example_1[predicted_sentiment_key] = [['neg', 'pos'], ['pos', 'neg']]
    example_2 = TargetText(text_id='2', text='some text')
    example_2[true_sentiment_key] = ['pos', 'neg', 'neu']
    example_2[predicted_sentiment_key] = [['neu', 'neg', 'pos'], ['neg', 'neg', 'pos']]

    true_labels = ['pos', 'neg', 'pos', 'neg', 'neu']
    pred_labels = [['neg', 'pos', 'neu', 'neg', 'pos'], 
                   ['pos', 'neg', 'neg', 'neg', 'pos']]
    return TargetTextCollection([example_1, example_2]), true_labels, pred_labels

def passable_example_multiple_wrong_preds(true_sentiment_key: str, 
                                          predicted_sentiment_key: str
                                          ) -> Tuple[TargetTextCollection, List[str], List[List[str]]]:
    example_1 = TargetText(text_id='1', text='some text')
    example_1[true_sentiment_key] = ['pos', 'neg']
    example_1[predicted_sentiment_key] = [['neg', 'pos'], ['neu', 'neu']]
    example_2 = TargetText(text_id='2', text='some text')
    example_2[true_sentiment_key] = ['pos', 'neg', 'neu']
    example_2[predicted_sentiment_key] = [['neu', 'pos', 'pos'], ['neg', 'pos', 'pos']]

    true_labels = ['pos', 'neg', 'pos', 'neg', 'neu']
    pred_labels = [['neg', 'pos', 'neu', 'pos', 'pos'], 
                   ['neu', 'neu', 'neg', 'pos', 'pos']]
    return TargetTextCollection([example_1, example_2]), true_labels, pred_labels

def passable_diff_num_labels(true_sentiment_key: str, 
                             predicted_sentiment_key: str
                             ) -> Tuple[TargetTextCollection, List[str], List[List[str]]]:
    example_1 = TargetText(text_id='1', text='some text')
    example_1[true_sentiment_key] = ['pos', 'neg']
    example_1[predicted_sentiment_key] = [['neg', 'pos']]
    example_2 = TargetText(text_id='2', text='some text')
    example_2[true_sentiment_key] = ['pos', 'neg', 'neu']
    example_2[predicted_sentiment_key] = [['neg', 'neg', 'pos']]

    true_labels = ['pos', 'neg', 'pos', 'neg', 'neu']
    pred_labels = [['neg', 'pos', 'neg', 'neg', 'pos']]
    return TargetTextCollection([example_1, example_2]), true_labels, pred_labels

def passable_subset_multiple_preds(true_sentiment_key: str, 
                                   predicted_sentiment_key: str
                                   ) -> Tuple[TargetTextCollection, List[str], List[List[str]]]:
    example_1 = TargetText(text_id='1', text='some text')
    example_1[true_sentiment_key] = ['pos', 'neg']
    example_1[predicted_sentiment_key] = [['neg', 'neg'], ['neg', 'neg']]
    example_2 = TargetText(text_id='2', text='some text')
    example_2[true_sentiment_key] = ['pos', 'neg', 'neu']
    example_2[predicted_sentiment_key] = [['neu', 'neg', 'neu'], ['neg', 'neg', 'neu']]

    true_labels = ['pos', 'neg', 'pos', 'neg', 'neu']
    pred_labels = [['neg', 'neg', 'neu', 'neg', 'neu'], 
                   ['neg', 'neg', 'neg', 'neg', 'neu']]
    return TargetTextCollection([example_1, example_2]), true_labels, pred_labels

def passable_subset_multiple_preds_1(true_sentiment_key: str, 
                                     predicted_sentiment_key: str
                                     ) -> Tuple[TargetTextCollection, List[str], List[List[str]]]:
    example_1 = TargetText(text_id='1', text='some text')
    example_1[true_sentiment_key] = ['pos', 'neg']
    example_1[predicted_sentiment_key] = [['pos', 'neg'], ['neg', 'neg']]
    example_2 = TargetText(text_id='2', text='some text')
    example_2[true_sentiment_key] = ['pos', 'neg', 'neu']
    example_2[predicted_sentiment_key] = [['neu', 'neg', 'neu'], ['neg', 'neg', 'neu']]

    true_labels = ['pos', 'neg', 'pos', 'neg', 'neu']
    pred_labels = [['pos', 'neg', 'neu', 'neg', 'neu'], 
                   ['neg', 'neg', 'neg', 'neg', 'neu']]
    return TargetTextCollection([example_1, example_2]), true_labels, pred_labels

def passable_subset_multiple_preds_2(true_sentiment_key: str, 
                                     predicted_sentiment_key: str
                                     ) -> Tuple[TargetTextCollection, List[str], List[List[str]]]:
    example_1 = TargetText(text_id='1', text='some text')
    example_1[true_sentiment_key] = ['pos', 'neg']
    example_1[predicted_sentiment_key] = [['pos', 'pos'], ['neg', 'neg']]
    example_2 = TargetText(text_id='2', text='some text')
    example_2[true_sentiment_key] = ['pos', 'neg', 'neu']
    example_2[predicted_sentiment_key] = [['pos', 'pos', 'pos'], ['neg', 'neg', 'neu']]

    true_labels = ['pos', 'neg', 'pos', 'neg', 'neu']
    pred_labels = [['pos', 'pos', 'pos', 'pos', 'pos'], 
                   ['neg', 'neg', 'neg', 'neg', 'neu']]
    return TargetTextCollection([example_1, example_2]), true_labels, pred_labels

def wrong_labels_example(true_sentiment_key: str, 
                         predicted_sentiment_key: str
                         ) -> Tuple[TargetTextCollection, List[str], List[List[str]]]:
    example_1 = TargetText(text_id='1', text='some text')
    example_1[true_sentiment_key] = ['pos', 'neg']
    example_1[predicted_sentiment_key] = [['neg', 'pos']]
    example_2 = TargetText(text_id='2', text='some text')
    example_2[true_sentiment_key] = ['pos', 'neg', 'neu']
    example_2[predicted_sentiment_key] = [['neu', 'some', 'pos']]

    true_labels = ['pos', 'neg', 'pos', 'neg', 'neu']
    pred_labels = [['neg', 'pos', 'neu', 'some', 'pos']]
    return TargetTextCollection([example_1, example_2]), true_labels, pred_labels

def wrong_multiple_labels_example(true_sentiment_key: str, 
                                  predicted_sentiment_key: str
                                  ) -> Tuple[TargetTextCollection, List[str], List[List[str]]]:
    example_1 = TargetText(text_id='1', text='some text')
    example_1[true_sentiment_key] = ['pos', 'neg']
    example_1[predicted_sentiment_key] = [['neg', 'neg'], ['neg', 'neg']]
    example_2 = TargetText(text_id='2', text='some text')
    example_2[true_sentiment_key] = ['pos', 'neg', 'neu']
    example_2[predicted_sentiment_key] = [['neu', 'so', 'neu'], ['neg', 'neg', 'neu']]

    true_labels = ['pos', 'neg', 'pos', 'neg', 'neu']
    pred_labels = [['neg', 'neg', 'neu', 'so', 'neu'], 
                   ['neg', 'neg', 'neg', 'neg', 'neu']]
    return TargetTextCollection([example_1, example_2]), true_labels, pred_labels

def empty_preds_examples(true_sentiment_key: str, 
                         predicted_sentiment_key: str
                         ) -> Tuple[TargetTextCollection, List[str], List[List[str]]]:
    example_1 = TargetText(text_id='1', text='some text')
    example_1[true_sentiment_key] = ['pos', 'neg']
    example_1[predicted_sentiment_key] = [[]]
    example_2 = TargetText(text_id='2', text='some text')
    example_2[true_sentiment_key] = ['pos', 'neg', 'neu']
    example_2[predicted_sentiment_key] = [[]]

    true_labels = ['pos', 'neg', 'pos', 'neg', 'neu']
    pred_labels = [[]]
    return TargetTextCollection([example_1, example_2]), true_labels, pred_labels

def diff_label_pred_lengths(true_sentiment_key: str, 
                            predicted_sentiment_key: str
                            ) -> Tuple[TargetTextCollection, List[str], List[List[str]]]:
    example_1 = TargetText(text_id='1', text='some text')
    example_1[true_sentiment_key] = ['pos', 'neg']
    example_1[predicted_sentiment_key] = [['neg', 'pos'], ['pos', 'neg']]
    example_2 = TargetText(text_id='2', text='some text')
    example_2[true_sentiment_key] = ['pos', 'neg', 'neu']
    example_2[predicted_sentiment_key] = [['neu', 'neg'], ['neg', 'neg', 'pos']]

    true_labels = ['pos', 'neg', 'pos', 'neg', 'neu']
    pred_labels = [['neg', 'pos', 'neu', 'neg'], 
                   ['pos', 'neg', 'neg', 'neg', 'pos']]
    return TargetTextCollection([example_1, example_2]), true_labels, pred_labels

def all_diff_label_pred_lengths(true_sentiment_key: str, 
                                predicted_sentiment_key: str
                                ) -> Tuple[TargetTextCollection, List[str], List[List[str]]]:
    example_1 = TargetText(text_id='1', text='some text')
    example_1[true_sentiment_key] = ['pos', 'neg']
    example_1[predicted_sentiment_key] = [['neg', 'pos'], ['pos', 'neg']]
    example_2 = TargetText(text_id='2', text='some text')
    example_2[true_sentiment_key] = ['pos', 'neg', 'neu']
    example_2[predicted_sentiment_key] = [['neu', 'neg'], ['neg', 'neg']]

    true_labels = ['pos', 'neg', 'pos', 'neg', 'neu']
    pred_labels = [['neg', 'pos', 'neu', 'neg'], 
                   ['pos', 'neg', 'neg', 'neg']]
    return TargetTextCollection([example_1, example_2]), true_labels, pred_labels

def diff_label_pred_values(true_sentiment_key: str, 
                           predicted_sentiment_key: str
                           ) -> Tuple[TargetTextCollection, List[str], List[List[str]]]:
    example_1 = TargetText(text_id='1', text='some text')
    example_1[true_sentiment_key] = ['pos', 'neg']
    example_1[predicted_sentiment_key] = [['neg', 'po'], ['pos', 'neg']]
    example_2 = TargetText(text_id='2', text='some text')
    example_2[true_sentiment_key] = ['pos', 'neg', 'neu']
    example_2[predicted_sentiment_key] = [['neu', 'neg', 'po'], ['neg', 'neg', 'pos']]

    true_labels = ['pos', 'neg', 'pos', 'neg', 'neu']
    pred_labels = [['neg', 'po', 'neu', 'neg', 'po'], 
                   ['pos', 'neg', 'neg', 'neg', 'pos']]
    return TargetTextCollection([example_1, example_2]), true_labels, pred_labels

def all_diff_label_pred_values(true_sentiment_key: str, 
                               predicted_sentiment_key: str
                               ) -> Tuple[TargetTextCollection, List[str], List[List[str]]]:
    example_1 = TargetText(text_id='1', text='some text')
    example_1[true_sentiment_key] = ['pos', 'neg']
    example_1[predicted_sentiment_key] = [['neg', 'poss'], ['pos', 'neg']]
    example_2 = TargetText(text_id='2', text='some text')
    example_2[true_sentiment_key] = ['pos', 'neg', 'neu']
    example_2[predicted_sentiment_key] = [['neu', 'neg', 'pos'], ['neg', 'neg', 'poss']]

    true_labels = ['pos', 'neg', 'pos', 'neg', 'neu']
    pred_labels = [['neg', 'poss', 'neu', 'neg', 'pos'], 
                   ['pos', 'neg', 'neg', 'neg', 'poss']]
    return TargetTextCollection([example_1, example_2]), true_labels, pred_labels

def no_pred_values(true_sentiment_key: str, predicted_sentiment_key: str
                   ) -> Tuple[TargetTextCollection, List[str], List[List[str]]]:
    example_1 = TargetText(text_id='1', text='some text')
    example_1[true_sentiment_key] = []
    example_1[predicted_sentiment_key] = []
    example_2 = TargetText(text_id='2', text='some text')
    example_2[true_sentiment_key] = []
    example_2[predicted_sentiment_key] = []

    true_labels = []
    pred_labels = []
    return TargetTextCollection([example_1, example_2]), true_labels, pred_labels

def diff_num_preds(true_sentiment_key: str, 
                   predicted_sentiment_key: str
                   ) -> Tuple[TargetTextCollection, List[str], List[List[str]]]:
    example_1 = TargetText(text_id='1', text='some text')
    example_1[true_sentiment_key] = ['pos', 'neg']
    example_1[predicted_sentiment_key] = [['pos', 'neg']]
    example_2 = TargetText(text_id='2', text='some text')
    example_2[true_sentiment_key] = ['pos', 'neg', 'neu']
    example_2[predicted_sentiment_key] = [['neu', 'neg', 'pos'], ['neg', 'neg', 'poss']]

    true_labels = ['pos', 'neg', 'pos', 'neg', 'neu']
    pred_labels = [['neu', 'neg', 'pos'], 
                   ['pos', 'neg', 'neg', 'neg', 'poss']]
    return TargetTextCollection([example_1, example_2]), true_labels, pred_labels

@pytest.mark.parametrize("true_sentiment_key", ('target_sentiments', 'true values'))
@pytest.mark.parametrize("predicted_sentiment_key", ('predictions', 'another'))
def test_get_labels(true_sentiment_key: str, predicted_sentiment_key: str):
    normal_examples, true_labels, pred_labels = passable_example(true_sentiment_key, predicted_sentiment_key)
    labels = get_labels(normal_examples, true_sentiment_key, predicted_sentiment_key)
    assert true_labels == labels[0]
    assert pred_labels == labels[1]
    # Case where the predictions have more than one set of predictions per target
    normal_examples, true_labels, pred_labels = passable_example_multiple_preds(true_sentiment_key, predicted_sentiment_key)
    labels = get_labels(normal_examples, true_sentiment_key, predicted_sentiment_key)
    assert true_labels == labels[0]
    assert pred_labels == labels[1]
    # Return empty lists
    labels = get_labels(TargetTextCollection(), true_sentiment_key, predicted_sentiment_key)
    assert [] == labels[0]
    assert [] == labels[1]
    # Handles the case where the prediction labels are a subset of the True labels
    normal_examples, true_labels, pred_labels = passable_subset_multiple_preds(true_sentiment_key, predicted_sentiment_key)
    labels = get_labels(normal_examples, true_sentiment_key, predicted_sentiment_key)
    assert true_labels == labels[0]
    assert pred_labels == labels[1]
    # Raise an error as there are no predictions
    examples, true_labels, pred_labels = empty_preds_examples(true_sentiment_key, predicted_sentiment_key)
    with pytest.raises(ValueError):
        get_labels(examples, true_sentiment_key, predicted_sentiment_key)
    # Raises an error as one of the multiple predictions is not the same
    examples, true_labels, pred_labels = diff_label_pred_lengths(true_sentiment_key, predicted_sentiment_key)
    with pytest.raises(ValueError):
        get_labels(examples, true_sentiment_key, predicted_sentiment_key)
    # Raises an error as all of the multiple predictions do not have the same 
    # length as the True labels 
    examples, true_labels, pred_labels = all_diff_label_pred_lengths(true_sentiment_key, predicted_sentiment_key)
    with pytest.raises(ValueError):
        get_labels(examples, true_sentiment_key, predicted_sentiment_key)
    # Raises an error as the labels in one of the predictions is different to 
    # the True labels
    examples, true_labels, pred_labels = diff_label_pred_values(true_sentiment_key, predicted_sentiment_key)
    with pytest.raises(ValueError):
        get_labels(examples, true_sentiment_key, predicted_sentiment_key)
    # Raises an error as the label in all of the predictions are different to 
    # the True labels
    examples, true_labels, pred_labels = all_diff_label_pred_values(true_sentiment_key, predicted_sentiment_key)
    with pytest.raises(ValueError):
        get_labels(examples, true_sentiment_key, predicted_sentiment_key)

@pytest.mark.parametrize("true_sentiment_key", ('target_sentiments', 'true values'))
@pytest.mark.parametrize("predicted_sentiment_key", ('predictions', 'another'))
def test_metric_error_checks_and_accuracy(true_sentiment_key: str, 
                                          predicted_sentiment_key: str):
    # Test accuracy works as should on one set of predictions
    example, _, _ = passable_example(true_sentiment_key, predicted_sentiment_key)
    score = accuracy(example, true_sentiment_key, predicted_sentiment_key, 
                     False, False, None)
    assert 0.2 == score
    # Test that accuracy fails when only one set of predictions when asking for 
    # average or array_score or both
    with pytest.raises(ValueError):
        accuracy(example, true_sentiment_key, predicted_sentiment_key, 
                 True, False, None)
    with pytest.raises(ValueError):
        accuracy(example, true_sentiment_key, predicted_sentiment_key, 
                 False, True, None)
    with pytest.raises(ValueError):
        accuracy(example, true_sentiment_key, predicted_sentiment_key, 
                 True, True, None)
    # Test key error for true and predicted sentiment key
    with pytest.raises(KeyError):
        accuracy(example, 'different', predicted_sentiment_key, 
                 False, False, None)
    with pytest.raises(KeyError):
        accuracy(example, true_sentiment_key, 'different', 
                 False, False, None)
    # Test the assert number of labels when predicted and true labels have the 
    # same number of labels
    score = accuracy(example, true_sentiment_key, predicted_sentiment_key, 
                     False, False, 3)
    assert 0.2 == score
    # Test when the number of labels that are True are different to predicted
    example, _, _ = passable_diff_num_labels(true_sentiment_key, predicted_sentiment_key)
    score = accuracy(example, true_sentiment_key, predicted_sentiment_key, 
                     False, False, 3)
    assert 0.2 == score
    # Test assert labels for the multiple sentiment case
    example, _, _ = passable_example_multiple_preds(true_sentiment_key, predicted_sentiment_key)
    score = accuracy(example, true_sentiment_key, predicted_sentiment_key, 
                     True, False, 3)
    assert 0.4 == score
    # Test assert label when the number of labels in True is different to all 
    # preds
    example, _, _ = passable_subset_multiple_preds(true_sentiment_key, predicted_sentiment_key)
    score = accuracy(example, true_sentiment_key, predicted_sentiment_key, 
                     True, False, 3)
    assert 0.6 == score
    # Test assert label when the number of labels in True is different to all
    # but one of the preds
    example, _, _ = passable_subset_multiple_preds_1(true_sentiment_key, predicted_sentiment_key)
    score = accuracy(example, true_sentiment_key, predicted_sentiment_key, 
                     True, False, 3)
    assert 0.7 == score
    # Test assert label when the number of labels in True is different to all
    # of the preds
    example, _, _ = passable_subset_multiple_preds_2(true_sentiment_key, predicted_sentiment_key)
    score = accuracy(example, true_sentiment_key, predicted_sentiment_key, 
                     True, False, 3)
    assert 0.5 == score
    # Test assert label when it is not the same
    with pytest.raises(LabelError):
        score = accuracy(example, true_sentiment_key, predicted_sentiment_key, 
                         True, False, 2)
    with pytest.raises(LabelError):
        score = accuracy(example, true_sentiment_key, predicted_sentiment_key, 
                         True, False, 4)
    with pytest.raises(LabelError):
        score = accuracy(example, true_sentiment_key, predicted_sentiment_key, 
                         True, False, 0)
    # Test the array scores case of the multiple preds
    example, _, _ = passable_subset_multiple_preds_1(true_sentiment_key, predicted_sentiment_key)
    score = accuracy(example, true_sentiment_key, predicted_sentiment_key, 
                     False, True, None)
    assert [0.8, 0.6] == score
    # Test label error when the prediction labels do not match the true labels
    example, _, _ = wrong_labels_example(true_sentiment_key, predicted_sentiment_key)
    with pytest.raises(ValueError):
        accuracy(example, true_sentiment_key, predicted_sentiment_key, 
                 False, False, None)
    # Test label error when the prediction labels in one set of predictions do 
    # not match the true labels
    example, _, _ = wrong_multiple_labels_example(true_sentiment_key, predicted_sentiment_key)
    with pytest.raises(ValueError):
        accuracy(example, true_sentiment_key, predicted_sentiment_key, 
                 True, False, None)
    # Test the case for multiple predictions where the average and array_scores 
    # are both True or both False
    # Test assert labels for the multiple sentiment case
    with pytest.raises(ValueError):
        accuracy(example, true_sentiment_key, predicted_sentiment_key, 
                 False, False, 3)
    with pytest.raises(ValueError):
        accuracy(example, true_sentiment_key, predicted_sentiment_key, 
                 True, True, 3)
    # Test the case when accuracy is 0 or 1
    example, _, _ = passable_example_multiple_wrong_preds(true_sentiment_key, predicted_sentiment_key)
    score = accuracy(example, true_sentiment_key, predicted_sentiment_key, 
                     True, False, 3)
    assert 0.0 == score
    example, _, _ = passable_example_multiple_wrong_preds(true_sentiment_key, predicted_sentiment_key)
    score = accuracy(example, true_sentiment_key, predicted_sentiment_key, 
                     False, True, None)
    assert [0.0, 0.0] == score

    example, _, _ = passable_wrong_example(true_sentiment_key, predicted_sentiment_key)
    score = accuracy(example, true_sentiment_key, predicted_sentiment_key, 
                     False, False, 3)
    assert 0.0 == score
    # Test when there are no predicted scores
    example, _, _ = no_pred_values(true_sentiment_key, predicted_sentiment_key)
    with pytest.raises(ValueError):
        accuracy(example, true_sentiment_key, predicted_sentiment_key, 
                 False, False, None)
    # Test when there are different number of predictions per target
    example, _, _ = diff_num_preds(true_sentiment_key, predicted_sentiment_key)
    with pytest.raises(ValueError):
        accuracy(example, true_sentiment_key, predicted_sentiment_key, 
                 False, False, None)

@pytest.mark.parametrize("true_sentiment_key", ('target_sentiments', 'true values'))
@pytest.mark.parametrize("predicted_sentiment_key", ('predictions', 'another'))
def test_macro_f1(true_sentiment_key: str, predicted_sentiment_key: str):
    # Test macro F1 works as should on one set of predictions
    example, _, _ = passable_example(true_sentiment_key, predicted_sentiment_key)
    score = macro_f1(example, true_sentiment_key, predicted_sentiment_key, 
                     False, False, None)
    assert 0.5 / 3.0 == score
    # Test it works on multiple predictions
    example, _, _ = passable_example_multiple_preds(true_sentiment_key, predicted_sentiment_key)
    score = macro_f1(example, true_sentiment_key, predicted_sentiment_key, 
                     True, False, None)
    assert ((0.5 / 3.0) + (1.3 / 3)) / 2 == score
    # Test it works on multiple predictions
    example, _, _ = passable_example_multiple_preds(true_sentiment_key, predicted_sentiment_key)
    score = macro_f1(example, true_sentiment_key, predicted_sentiment_key, 
                     False, True, None)
    assert [(0.5 / 3.0), (1.3 / 3)] == score
    

    


