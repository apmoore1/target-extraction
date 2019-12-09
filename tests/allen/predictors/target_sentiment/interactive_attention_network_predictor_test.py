from pathlib import Path

from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

import target_extraction

class TestInteractiveAttentionNetworkPredictor():

    
    archive_dir = Path(__file__, '..', '..', '..', '..', 'data', 'allen', 
                       'predictors', 'target_sentiment', 
                       'IAN').resolve()
    ian_archive = load_archive(str(Path(archive_dir, 'ian', 'model.tar.gz')))
    ian_predictor = Predictor.from_archive(ian_archive, 'target-sentiment')

    inter_ian_archive = load_archive(str(Path(archive_dir, 'inter_ian', 'model.tar.gz')))
    inter_ian_predictor = Predictor.from_archive(inter_ian_archive, 'target-sentiment')

    ian_sequences_archive = load_archive(str(Path(archive_dir, 'ian_target_sequences', 
                                                  'model.tar.gz')))
    ian_sequences_predictor = Predictor.from_archive(ian_sequences_archive, 'target-sentiment')

    name_predictors = [('ian', ian_predictor), ('inter_ian', inter_ian_predictor),
                       ('ian_target_sequences', ian_sequences_predictor)]

    def test_outputs(self):
        '''
        This also in affect tests the :meth:`decode` method within 
        :class:`target_extraction.allen.models.target_sentiment.
                interactive_attention_network.InteractiveAttentionNetworkClassifier`
        '''
        text = "The food was lousy - too sweet or too salty and the portions tiny."
        tokens = ["The", "food", "was", "lousy", "-", "too", "sweet", "or", 
                  "too", "salty", "and", "the", "portions", "tiny" ,"."]
        example_input = {"text": text, "spans": [[4, 18]], "targets": ["food was lousy"]}
        # The last two do not matter.
        output_keys = ['words', 'text', 'targets', 'target words', 
                       'class_probabilities', 'sentiments', 'targets_mask',
                       'word_attention', 'targets_attention', 'targets_word_mask',
                       'context_mask']

        for name, predictor in self.name_predictors:
            result = predictor.predict_json(example_input)
            for output_key in output_keys:
                assert output_key in result
            assert len(output_keys) == len(result)
            assert tokens == result['words']
            assert text == result['text']
            assert ['food was lousy'] == result['targets']
            assert [['food', 'was', 'lousy']] == result['target words']
            assert 1 == len(result['sentiments'])
            assert result['sentiments'][0] in ['positive', 'negative', 'neutral']
            assert 1 == len(result['class_probabilities'])
            assert len(result['class_probabilities']) == len(result['sentiments'])
            assert 3 == len(result['class_probabilities'][0])
            for class_probability in result['class_probabilities']:
                for probability in class_probability:
                    assert probability > 0 and probability < 1
            assert 3 == len(result['targets_attention'][0])
            for word_attention in result['word_attention'][0]:
                assert word_attention > 0.0
            assert 15 == len(result['word_attention'][0])
    
    def test_batch_inputs(self):
        '''
        The difference between this and non-batch is that in the batch case 
        it has to handle case where some samples have different number of 
        targets
        '''
        text_1 = "The food was lousy - too sweet or too salty and the portions tiny."
        text_2 = "We, there were four of us, arrived at noon - the place was "\
                 "empty - and the staff"
        texts = [text_1, text_2]

        tokens_1 = ["The", "food", "was", "lousy", "-", "too", "sweet", "or", 
                    "too", "salty", "and", "the", "portions", "tiny" ,"."]
        tokens_2 = ["We", ",",  "there", "were", "four", "of", "us", ",", 
                    "arrived", "at", "noon", "-", "the", "place", "was", 
                    "empty", "-", "and", "the", "staff"]
        tokens = [tokens_1, tokens_2]

        targets = [["food was lousy", "portions"], ["staff"]]
        target_words = [[["food", "was", "lousy"], ["portions"]], [["staff"]]]

        example_1 = {'text': text_1, 'spans': [[4, 18], [52, 60]], 
                     'targets': targets[0]}
        example_2 = {'text': text_2, "spans": [[75, 80]], 'targets': targets[1]}
        example_input = [example_1, example_2]

        target_attention_0 = [[0.33, 0.33, 0.33], [1.0]]
        target_attention_1 = [[1.0]]
        target_attention_input = [target_attention_0, target_attention_1]

        word_attention_0 = [[1] * 15] * 2
        word_attention_1 = [[1] * 20]
        word_attention_input = [word_attention_0, word_attention_1]

        number_sentiments = [2, 1]

        for name, predictor in self.name_predictors:
            results = predictor.predict_batch_json(example_input)
            for i, result in enumerate(results):
                assert tokens[i] == result['words']
                assert texts[i] == result['text']
                assert targets[i] == result['targets']
                assert target_words[i] == result['target words']
                assert number_sentiments[i] == len(result['sentiments'])
                assert len(result['sentiments']) == len(result['class_probabilities'])
                assert 3 == len(result['class_probabilities'][0])

                for class_probability in result['class_probabilities']:
                    for probability in class_probability:
                        assert probability > 0 and probability < 1
                for sentiment in result['sentiments']:
                    assert sentiment in ['positive', 'negative', 'neutral']

                if i != 0:
                    assert target_attention_input[i] == result['targets_attention']
                for target_attention_index, target_attention in enumerate(target_attention_input[i]):
                    assert len(target_attention) == len(result['targets_attention'][target_attention_index])

                assert len(word_attention_input[i]) == len(result["word_attention"])
                for w_i in range(len(word_attention_input[i])):
                    assert len(word_attention_input[i][w_i]) == len(result["word_attention"][w_i])
            # ensure that the attention weights for the sentences are not the same 
            # for the same sentence
            first_result = results[0]
            assert first_result["word_attention"][0] != first_result["word_attention"][1]
