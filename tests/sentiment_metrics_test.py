from typing import List, Tuple

import pytest

from target_extraction.data_types import TargetTextCollection, TargetText
from target_extraction.sentiment_metrics import get_labels

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


