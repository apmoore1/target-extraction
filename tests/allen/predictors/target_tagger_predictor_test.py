from pathlib import Path
from itertools import product

from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from allennlp.common.params import Params
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.checks import ConfigurationError
import pytest

import target_extraction
from target_extraction.allen.predictors import TargetTaggerPredictor

class TestTargetTaggerPredictor():

    def test_non_constructor_params(self):
        example_input = {'text': "The laptop"}
        example_token_input = {'tokens': ["The", "laptop", "case", "was", "great", "and", "cover", "was", "rubbish"],
                               'text': "The laptop case was great and cover was rubbish"}
        example_token_pos_input = {**example_token_input, 
                                   'pos_tags': ["DET", "NOUN", "NOUN", "AUX", "ADJ", "CCONJ", "NOUN", "AUX", "ADJ"]}

        archive_dir = Path(__file__, '..', '..', '..', 'data', 'allen', 
                           'predictors', 'target_tagger').resolve()
        non_pos_archive = load_archive(str(Path(archive_dir, 'non_pos_model', 
                                                'model.tar.gz')))
        pos_archive = load_archive(str(Path(archive_dir, 'pos_model', 'model.tar.gz')))

        non_pos_predictor = Predictor.from_archive(non_pos_archive, 'target-tagger')
        pos_predictor = Predictor.from_archive(pos_archive, 'target-tagger')

        predictors = [('non_pos', non_pos_predictor), ('pos', pos_predictor)]
        output_keys = ['logits', 'mask', 'tags', 'class_probabilities', 'words', 
                       'text']
        for name, predictor in predictors:
            # Example where the input only provides the text and the predictor 
            # has to generate the tokens and pos tags
            result = predictor.predict_json(example_input)
            for output_key in output_keys:
                    assert output_key in result
            correct_tokens = ["The", "laptop"]
            assert correct_tokens == result['words']
            correct_text = example_input['text']
            assert correct_text == result['text']
            # Example where the input only provides the tokens
            if name == 'pos':
                # Raised as there will be no POS tags.
                with pytest.raises(ConfigurationError):
                    result = predictor.predict_json(example_token_input)
            else:
                result = predictor.predict_json(example_token_input)
                for output_key in output_keys:
                    assert output_key in result
                correct_tokens = example_token_input['tokens']
                assert correct_tokens == result['words']
                correct_text = example_token_input['text']
                assert correct_text == result['text']
            # Example where the input provides the tokens and pos tags
            result = predictor.predict_json(example_token_pos_input)
            for output_key in output_keys:
                assert output_key in result
            correct_tokens = example_token_input['tokens']
            assert correct_tokens == result['words']
            correct_text = example_token_input['text']
            assert correct_text == result['text']

    @pytest.mark.parametrize("pos_tags", (True, False))
    @pytest.mark.parametrize("fine_grained_tags", (True, False))
    def test_constructor_parameters(self, pos_tags: bool, 
                                    fine_grained_tags: bool):
        example_input = {'text': "The laptop"}
        
        archive_dir = Path(__file__, '..', '..', '..', 'data', 'allen', 
                           'predictors', 'target_tagger').resolve()
        non_pos_archive = load_archive(str(Path(archive_dir, 'non_pos_model', 
                                                'model.tar.gz')))
        # For the Non-POS based model changing the POS tagger or the granuality 
        # should have no affect.
        non_pos_model = non_pos_archive.model
        non_pos_params = non_pos_archive.config
        non_pos_dataset_reader = DatasetReader.from_params(non_pos_params.pop('dataset_reader'))
        predictor = TargetTaggerPredictor(non_pos_model, non_pos_dataset_reader, 
                                          pos_tags=pos_tags, 
                                          fine_grained_tags=fine_grained_tags)
        result = predictor.predict_json(example_input)
        output_keys = ['logits', 'mask', 'tags', 'class_probabilities', 'words', 
                       'text']
        for output_key in output_keys:
            assert output_key in result
        
        # The POS based model should fail when the 
        pos_archive = load_archive(str(Path(archive_dir, 'pos_model', 'model.tar.gz')))
        pos_model = pos_archive.model
        pos_params = pos_archive.config
        pos_dataset_reader = DatasetReader.from_params(pos_params.pop('dataset_reader'))
        predictor = TargetTaggerPredictor(pos_model, pos_dataset_reader, 
                                          pos_tags=pos_tags, 
                                          fine_grained_tags=fine_grained_tags)
        # KeyError happens as the POS tags vocab is learnt on coarse and not 
        # fine labels.
        # When the no POS tags are given it should return a configuration error.
        if pos_tags == True and fine_grained_tags == False:
            result = predictor.predict_json(example_input)
            for output_key in output_keys:
                assert output_key in result
        elif pos_tags == True:
            with pytest.raises(KeyError):
                result = predictor.predict_json(example_input)
        else:
            with pytest.raises(ConfigurationError):
                result = predictor.predict_json(example_input)

        # To ensure the fine grained tags work we load the fine POS model
        pos_archive = load_archive(str(Path(archive_dir, 'fine_pos_model', 'model.tar.gz')))
        pos_model = pos_archive.model
        pos_params = pos_archive.config
        pos_dataset_reader = DatasetReader.from_params(pos_params.pop('dataset_reader'))
        predictor = TargetTaggerPredictor(pos_model, pos_dataset_reader, 
                                          pos_tags=pos_tags, 
                                          fine_grained_tags=fine_grained_tags)
        # KeyError happens as the POS tags vocab is learnt on fine and not 
        # coarse labels.
        # When the no POS tags are given it should return a configuration error.
        if pos_tags == True and fine_grained_tags == True:
            result = predictor.predict_json(example_input)
            for output_key in output_keys:
                        assert output_key in result
        elif pos_tags == True:
            with pytest.raises(KeyError):
                result = predictor.predict_json(example_input)
        else:
            with pytest.raises(ConfigurationError):
                result = predictor.predict_json(example_input)