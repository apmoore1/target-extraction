from pathlib import Path

from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

import target_extraction

class TestSplitContextsPredictor():

    
    archive_dir = Path(__file__, '..', '..', '..', '..', 'data', 'allen', 
                        'predictors', 'target_sentiment', 
                        'split_contexts').resolve()
    tdlstm_archive = load_archive(str(Path(archive_dir, 'tdlstm', 'model.tar.gz')))
    tdlstm_predictor = Predictor.from_archive(tdlstm_archive, 'target-sentiment')

    tclstm_archive = load_archive(str(Path(archive_dir, 'tclstm', 'model.tar.gz')))
    tclstm_predictor = Predictor.from_archive(tclstm_archive, 'target-sentiment')

    inter_tdlstm_archive = load_archive(str(Path(archive_dir, 'inter_tdlstm', 
                                                 'model.tar.gz')))
    inter_tdlstm_predictor = Predictor.from_archive(inter_tdlstm_archive, 
                                                    'target-sentiment')
    inter_tclstm_archive = load_archive(str(Path(archive_dir, 'inter_tclstm', 
                                                 'model.tar.gz')))
    inter_tclstm_predictor = Predictor.from_archive(inter_tclstm_archive, 
                                                    'target-sentiment')

    predictors = [('tdlstm', tdlstm_predictor), ('tclstm', tclstm_predictor), 
                  ('inter_tdlstm', inter_tclstm_predictor), 
                  ('inter_tclstm', inter_tclstm_predictor)]


    def test_outputs(self):
        '''
        This also in affect tests the :meth:`decode` method within 
        :class:`target_extraction.allen.models.target_sentiment.
                split_contexts.SplitContextsClassifier`
        '''
        text = "The food was lousy - too sweet or too salty and the portions tiny."
        tokens = ["The", "food", "was", "lousy", "-", "too", "sweet", "or", 
                  "too", "salty", "and", "the", "portions", "tiny" ,"."]
        example_input = {"text": text, "spans": [[4, 8]], "targets": ["food"]}
       
        output_keys = ['words', 'text', 'targets', 'target words', 
                       'class_probabilities', 'sentiments']

        for name, predictor in self.predictors:
            result = predictor.predict_json(example_input)
            for output_key in output_keys:
                assert output_key in result
            assert tokens == result['words']
            assert text == result['text']
            assert ['food'] == result['targets']
            assert [['food']] == result['target words']
            assert 1 == len(result['sentiments'])
            assert result['sentiments'][0] in ['positive', 'negative', 'neutral']
            assert 1 == len(result['class_probabilities'])
            assert len(result['class_probabilities']) == len(result['sentiments'])
            assert 3 == len(result['class_probabilities'][0])
            for class_probability in result['class_probabilities']:
                for probability in class_probability:
                    assert probability > 0 and probability < 1
    
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

        targets = [["food", "portions"], ["staff"]]
        target_words = [[["food"], ["portions"]], [["staff"]]]

        example_1 = {'text': text_1, 'spans': [[4, 8], [52, 60]], 
                     'targets': targets[0]}
        example_2 = {'text': text_2, "spans": [[75, 80]], 'targets': targets[1]}
        example_input = [example_1, example_2]

        number_sentiments = [2, 1]

        for name, predictor in self.predictors:
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
