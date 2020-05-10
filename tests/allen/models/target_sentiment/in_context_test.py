import copy
from pathlib import Path
import tempfile

from allennlp.common.testing import ModelTestCase
from allennlp.common.params import Params
from allennlp.data import Batch
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.nn import InitializerApplicator, Initializer
from flaky import flaky
from allennlp.common.checks import ConfigurationError
import pytest

import target_extraction
from .util import loss_weights

class InContextClassifierTest(ModelTestCase):
    def setUp(self):
        super().setUp()

        test_dir = Path(__file__, '..', '..', '..', '..','data', 'allen',  
                        'models', 'target_sentiment').resolve()
        test_data = str(Path(test_dir, 'multi_target_category_sentiments.json'))
        config_dir = Path(test_dir, 'In_Context')
        self.in_context_config = str(Path(config_dir, 'in_context_config.jsonnet'))
        self.max_pool_config = str(Path(config_dir, 'max_pool_config.jsonnet'))
        self.feedforward_config = str(Path(config_dir, 'feedforward.jsonnet'))
        self.set_up_model(self.in_context_config, test_data)

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_max_pool_train_save(self):
        params = Params.from_file(self.max_pool_config).duplicate()
        params_copy = copy.deepcopy(params)
        Model.from_params(vocab=self.vocab, params=params_copy.get('model'))
        with tempfile.NamedTemporaryFile(mode='w+') as temp_file:
            params.to_file(temp_file.name)
            self.ensure_model_can_train_save_and_load(temp_file.name)

    def test_feedforward_train_save(self):
        params = Params.from_file(self.feedforward_config).duplicate()
        params_copy = copy.deepcopy(params)
        Model.from_params(vocab=self.vocab, params=params_copy.get('model'))
        with tempfile.NamedTemporaryFile(mode='w+') as temp_file:
            params.to_file(temp_file.name)
            self.ensure_model_can_train_save_and_load(temp_file.name)

    @flaky
    def test_batch_predictions_are_consistent(self):
        ignore = ['words', 'text', 'targets', 'target words']
        self.ensure_batch_predictions_are_consistent(keys_to_ignore=ignore)

    def test_wrong_target_encoding_pooling_function(self):
        params = Params.from_file(self.in_context_config)
        params['model']['target_encoding_pooling_function'] = 'error'
        with pytest.raises(ValueError):
            Model.from_params(vocab=self.vocab, params=params.get('model'))

    def test_embedding_encoder_dim_match(self):
        params = Params.from_file(self.in_context_config)
        params['model']['context_field_embedder']["token_embedders"]['tokens']["embedding_dim"] = 5
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=params.get('model'))

    def test_encoder_feedforward_dim_match(self):
        params = Params.from_file(self.feedforward_config)
        params['model']['context_encoder']['hidden_size'] = 5
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=params.get('model'))
        
    def test_forward_pass_runs_correctly(self):
        params = Params.from_file(self.in_context_config)   
        model = Model.from_params(vocab=self.vocab, 
                                  params=params.get('model'))
        training_tensors = self.dataset.as_tensor_dict()

        output_dict = model(**training_tensors)
        for key in output_dict.keys():
            assert key in {'class_probabilities', 'targets_mask',
                           'loss', 'words', 'text', 'targets', 'target words'}
        words = output_dict['words']
        assert words[1] == ["The", "food", "was", "lousy", "-", "too", "sweet", 
                            "or", "too", "salty", "and", "the", "portions", 
                            "tiny", "."]
        text = output_dict['text']
        assert text[1] == "The food was lousy - too sweet or too salty and "\
                          "the portions tiny."
        targets = output_dict['targets']
        assert targets[0] == ["staff acted"]
        target_words = output_dict['target words']
        assert target_words[1] == [["food", "was", "lousy"], ["portions"]]
        target_mask = output_dict['targets_mask']
        assert target_mask.cpu().data.numpy().tolist() == [[1,0], [1,1]]
        class_probs = output_dict['class_probabilities']
        class_probs = class_probs[0].cpu().data.numpy().tolist()
        for prob in class_probs[0]:
            assert prob < 1
            assert prob > 0
        for prob in class_probs[1]:
            assert prob == 0

    def test_loss_weights(self):
        loss_weights(self.param_file, self.vocab, self.dataset) 
    