from typing import List, Tuple

import pytest

from target_extraction.analysis import util, sentiment_metrics
from target_extraction.data_types import TargetTextCollection, TargetText

def passable_example_multiple_preds(true_sentiment_key: str, 
                                    predicted_sentiment_key: str
                                    ) -> TargetTextCollection:
    example_1 = TargetText(text_id='1', text='some text')
    example_1[true_sentiment_key] = ['pos', 'neg']
    example_1[predicted_sentiment_key] = [['pos', 'neg'], ['pos', 'neg']]
    example_2 = TargetText(text_id='2', text='some text')
    example_2[true_sentiment_key] = ['pos', 'neg', 'neu']
    if predicted_sentiment_key == 'model_2':
        example_2[predicted_sentiment_key] = [['pos', 'neg', 'neu'], ['neg', 'neg', 'pos']]
    else:
        example_2[predicted_sentiment_key] = [['neu', 'neg', 'pos'], ['neg', 'neg', 'pos']]
    return TargetTextCollection([example_1, example_2])

@pytest.mark.parametrize('metric_name', (None, 'scores'))
@pytest.mark.parametrize('metric_function_name', ('macro_f1', 'accuracy'))
def test_metric_df(metric_function_name: str, metric_name: str):
    model_1_collection = passable_example_multiple_preds('true_sentiments', 'model_1')
    model_2_collection = passable_example_multiple_preds('true_sentiments', 'model_2')
    combined_collection = TargetTextCollection()
    for key, value in model_1_collection.items():
        combined_collection.add(value)
        combined_collection[key]['model_2'] = model_2_collection[key]['model_2']
    metric_function = getattr(sentiment_metrics, metric_function_name)
    # Test the array score version first
    model_1_scores = metric_function(model_1_collection, 'true_sentiments', 'model_1', 
                                     average=False, array_scores=True)
    model_2_scores = metric_function(model_2_collection, 'true_sentiments', 'model_2', 
                                     average=False, array_scores=True)
    test_df = util.metric_df(combined_collection, metric_function, 'true_sentiments', 
                             predicted_sentiment_keys=['model_1', 'model_2'], 
                             average=False, array_scores=True, metric_name=metric_name)
    get_metric_name = 'metric' if None else metric_name
    assert (4, 2) == test_df.shape
    for model_name, true_model_scores in [('model_1', model_1_scores), 
                                          ('model_2', model_2_scores)]:
        test_model_scores = test_df.loc[test_df['prediction key']==f'{model_name}'][f'{get_metric_name}']
        assert true_model_scores == test_model_scores.to_list()
    # Test the average version
    model_1_scores = metric_function(model_1_collection, 'true_sentiments', 'model_1', 
                                     average=True, array_scores=False)
    model_2_scores = metric_function(model_2_collection, 'true_sentiments', 'model_2', 
                                     average=True, array_scores=False)
    test_df = util.metric_df(combined_collection, metric_function, 'true_sentiments', 
                             predicted_sentiment_keys=['model_1', 'model_2'], 
                             average=True, array_scores=False, metric_name=metric_name)
    get_metric_name = 'metric' if None else metric_name
    assert (2,2) == test_df.shape
    for model_name, true_model_scores in [('model_1', model_1_scores), 
                                          ('model_2', model_2_scores)]:
        test_model_scores = test_df.loc[test_df['prediction key']==f'{model_name}'][f'{get_metric_name}']
        test_model_scores = test_model_scores.to_list()
        assert 1 == len(test_model_scores)
        assert true_model_scores == test_model_scores[0]