from pathlib import Path

from allennlp.common.testing import ModelTestCase
from allennlp.models import Model
from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
import pytest

import target_extraction

class TargetTaggerTest(ModelTestCase):

    def setUp(self):
        test_dir = Path(__file__, '..', '..', '..','data', 'allen')
        test_data_dir = Path(test_dir, 'dataset_readers', 'target_extraction')
        non_pos_data = str(Path(test_data_dir, 'non_pos_sequence.json').resolve())
        pos_data = str(Path(test_data_dir, 'pos_sequence.json').resolve())
        
        model_dir = Path(test_dir, 'models', 'target_extraction')
        self.crf_param_file = str(Path(model_dir, 'crf_config.json').resolve())
        self.softmax_param_file = str(Path(model_dir, 'softmax_config.json').resolve())
        self.no_encoder_file = str(Path(model_dir, 'no_encoder_config.json').resolve())

        self.data_files = [non_pos_data, pos_data]
        self.param_files = [self.crf_param_file, self.softmax_param_file,
                            self.no_encoder_file]

        self.set_up_model(self.crf_param_file, non_pos_data)
        super().setUp()

    def test_batch_predictions_are_consistent(self):
        # This only uses the CRF tagger with no POS data.
        self.ensure_batch_predictions_are_consistent()

    def test_crf_tagger_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.crf_param_file)

    def test_softmax_tagger_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.softmax_param_file)

    def test_forward_pass_runs_correctly(self):
        for param_file in self.param_files:
            params = Params.from_file(param_file).duplicate()
            model = Model.from_params(vocab=self.vocab, params=params.get('model'))
            training_tensors = self.dataset.as_tensor_dict()
            output_dict = model(**training_tensors)
            for key in output_dict.keys():
                assert key in {'logits', 'mask', 'tags', 'class_probabilities', 
                            'loss', 'words'}
            tags = output_dict['tags']
            assert len(tags) == 3
            assert len(tags[0]) == 9
            assert len(tags[1]) == 5
            for example_tags in tags:
                for tag_id in example_tags:
                    tag = self.model.vocab.get_token_from_index(tag_id, namespace="labels")
                    assert tag in {'O', 'B', 'I'}
            words = output_dict['words']
            assert words[0] == ["The", "laptop", "case", "was", "great", "and", 
                                "cover", "was", "rubbish"]

    def test_mismatching_dimensions_throws_configuration_error(self):
        params = Params.from_file(self.param_file).duplicate()
        # Make the encoder wrong - it should be 2 to match
        # the embedding dimension from the text_field_embedder.
        params["model"]["encoder"]["input_size"] = 10
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=params.pop("model"))
        # Make the encoder output wrong
        params = Params.from_file(self.param_file).duplicate()
        params["model"]["encoder"]["hidden_size"] = 70
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=params.pop("model"))
        # Remove the encoder and force an error with the text field embedder
        params = Params.from_file(self.param_file).duplicate()
        params["model"].pop("encoder")
        # Text embedder is output 55 and feed forward expects 160
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=params.pop("model"))
