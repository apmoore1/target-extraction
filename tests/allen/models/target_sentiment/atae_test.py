import copy
from pathlib import Path
import tempfile

from allennlp.common.testing import ModelTestCase
from allennlp.common.params import Params
from allennlp.data.dataset import Batch
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.nn import InitializerApplicator, Initializer
from flaky import flaky
from allennlp.common.checks import ConfigurationError
import pytest
import numpy

import target_extraction
from .util import loss_weights

class ATAEClassifierTest(ModelTestCase):
    def setUp(self):
        super().setUp()

        test_dir = Path(__file__, '..', '..', '..', '..','data', 'allen',  
                        'models', 'target_sentiment').resolve()
        test_data = str(Path(test_dir, 'multi_target_category_sentiments.json'))
        config_dir = Path(test_dir, 'ATAE_configs')
        self.atae_config = str(Path(config_dir, 'atae_config.jsonnet'))
        self.ae_config = str(Path(config_dir, 'ae_config.jsonnet'))
        self.at_config = str(Path(config_dir, 'at_config.jsonnet'))
        self.inter_atae_config = str(Path(config_dir, 'inter_atae_config.jsonnet'))
        self.position_weight_atae_config = str(Path(config_dir, 'position_weight_atae_config.jsonnet'))
        self.atae_elmo_config = str(Path(config_dir, 'atae_elmo_config.jsonnet'))
        self.atae_elmo_wordvector_config = str(Path(config_dir, 'atae_elmo_wordvector_config.jsonnet'))
        self.atae_elmo_target_sequences_config = str(Path(config_dir, 'atae_elmo_target_sequences_config.jsonnet'))

        self.set_up_model(self.atae_config, test_data)

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_ae_train_save(self):
        params = Params.from_file(self.ae_config).duplicate()
        params_copy = copy.deepcopy(params)
        Model.from_params(vocab=self.vocab, params=params_copy.get('model'))
        with tempfile.NamedTemporaryFile(mode='w+') as temp_file:
            params.to_file(temp_file.name)
            self.ensure_model_can_train_save_and_load(temp_file.name)
    
    def test_at_train_save(self):
        params = Params.from_file(self.at_config).duplicate()
        params_copy = copy.deepcopy(params)
        Model.from_params(vocab=self.vocab, params=params_copy.get('model'))
        with tempfile.NamedTemporaryFile(mode='w+') as temp_file:
            params.to_file(temp_file.name)
            self.ensure_model_can_train_save_and_load(temp_file.name)

    def test_inter_atae_train_save(self):
        params = Params.from_file(self.inter_atae_config).duplicate()
        params_copy = copy.deepcopy(params)
        Model.from_params(vocab=self.vocab, params=params_copy.get('model'))
        with tempfile.NamedTemporaryFile(mode='w+') as temp_file:
            params.to_file(temp_file.name)
            self.ensure_model_can_train_save_and_load(temp_file.name)

    def test_position_weight_atae_train_save(self):
        params = Params.from_file(self.position_weight_atae_config).duplicate()
        params_copy = copy.deepcopy(params)
        Model.from_params(vocab=self.vocab, params=params_copy.get('model'))
        with tempfile.NamedTemporaryFile(mode='w+') as temp_file:
            params.to_file(temp_file.name)
            self.ensure_model_can_train_save_and_load(temp_file.name)

    def test_inter_ae_train_save(self):
        params = Params.from_file(self.ae_config).duplicate()
        inter_target_encoder = {"sequence_encoder": {"type": "gru", "input_size": 20,
                                "hidden_size": 6, "bidirectional": False,
                                "num_layers": 1},
                                "type": "sequence_inter_target"}
        params['model']['inter_target_encoding'] = inter_target_encoder
        params_copy = copy.deepcopy(params)
        Model.from_params(vocab=self.vocab, params=params_copy.get('model'))
        with tempfile.NamedTemporaryFile(mode='w+') as temp_file:
            params.to_file(temp_file.name)
            self.ensure_model_can_train_save_and_load(temp_file.name)
    
    def test_inter_at_train_save(self):
        params = Params.from_file(self.at_config).duplicate()
        inter_target_encoder = {"sequence_encoder": {"type": "gru", "input_size": 20,
                                "hidden_size": 6, "bidirectional": False,
                                "num_layers": 1}, 
                                "type": "sequence_inter_target"}
        params['model']['inter_target_encoding'] = inter_target_encoder
        params_copy = copy.deepcopy(params)
        Model.from_params(vocab=self.vocab, params=params_copy.get('model'))
        with tempfile.NamedTemporaryFile(mode='w+') as temp_file:
            params.to_file(temp_file.name)
            self.ensure_model_can_train_save_and_load(temp_file.name)

    def test_elmo_atae_train_save(self):
        params = Params.from_file(self.atae_elmo_config).duplicate()
        params_copy = copy.deepcopy(params)
        Model.from_params(vocab=self.vocab, params=params_copy.get('model'))
        with tempfile.NamedTemporaryFile(mode='w+') as temp_file:
            params.to_file(temp_file.name)
            self.ensure_model_can_train_save_and_load(temp_file.name)

    def test_elmo_atae_wordvector_train_save(self):
        params = Params.from_file(self.atae_elmo_wordvector_config).duplicate()
        params_copy = copy.deepcopy(params)
        Model.from_params(vocab=self.vocab, params=params_copy.get('model'))
        with tempfile.NamedTemporaryFile(mode='w+') as temp_file:
            params.to_file(temp_file.name)
            self.ensure_model_can_train_save_and_load(temp_file.name)

    def test_elmo_atae_target_sequences_train_save(self):
        params = Params.from_file(self.atae_elmo_target_sequences_config).duplicate()
        params_copy = copy.deepcopy(params)
        Model.from_params(vocab=self.vocab, params=params_copy.get('model'))
        with tempfile.NamedTemporaryFile(mode='w+') as temp_file:
            params.to_file(temp_file.name)
            self.ensure_model_can_train_save_and_load(temp_file.name)

    @flaky
    def test_batch_predictions_are_consistent(self):
        ignore = ['words', 'text', 'targets', 'target words',
                  'word_attention', 'targets_attention', 'context_mask', 
                  'targets_word_mask']
        self.ensure_batch_predictions_are_consistent(keys_to_ignore=ignore)

    def test_atae_flags(self):
        # Should raise a config error if AE and AttentionAE are both False
        params = Params.from_file(self.atae_config).duplicate()
        params['model']['AE'] = False
        params['model']['AttentionAE'] = False
        params_copy = copy.deepcopy(params)
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=params_copy.get('model'))

    def test_atae_bidirectional_context_encoder(self):
        # Ensure that it works when it is not bi-directional
        params = Params.from_file(self.atae_config).duplicate()
        params['model']['context_encoder']['bidirectional'] = False
        params['model']['feedforward']['input_dim'] = 10
        params_copy = copy.deepcopy(params)
        Model.from_params(vocab=self.vocab, params=params_copy.get('model'))
        with tempfile.NamedTemporaryFile(mode='w+') as temp_file:
            params.to_file(temp_file.name)
            self.ensure_model_can_train_save_and_load(temp_file.name)

    def test_target_field_embedder(self):
        # Test that can handle having a target embedder as well as a text
        # embedder
        params = Params.from_file(self.atae_config).duplicate()
        target_embedder = {"token_embedders": {"tokens": {"type": "embedding",
                                                          "embedding_dim": 4,
                                                          "trainable": False}}}
        params['model']['target_field_embedder'] = target_embedder
        params_copy = copy.deepcopy(params)
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=params_copy.get('model'))
        params['model']['target_encoder']['input_size'] = 4
        params_copy = copy.deepcopy(params)
        Model.from_params(vocab=self.vocab, params=params_copy.get('model'))
        with tempfile.NamedTemporaryFile(mode='w+') as temp_file:
            params.to_file(temp_file.name)
            self.ensure_model_can_train_save_and_load(temp_file.name)
    
    def test_target_field_embedder_with_target_sequences(self):
        # Test that the target field embedder cannot be used with the 
        # Target Embedder.
        params = Params.from_file(self.atae_elmo_target_sequences_config).duplicate()
        target_embedder = {"token_embedders": {"tokens": {"type": "embedding",
                                                          "embedding_dim": 4,
                                                          "trainable": False}}}
        params['model']['target_field_embedder'] = target_embedder
        params_copy = copy.deepcopy(params)
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=params_copy.get('model'))
    
    def test_target_encoder(self):
        # Ensure raise a configuration error if the target encoder input dim 
        # is not equal to the context_field_embedder output dim.
        params = Params.from_file(self.atae_config).duplicate()
        params['model']['target_encoder']['input_size'] = 4
        params_copy = copy.deepcopy(params)
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=params_copy.get('model'))

    def test_context_encoder(self):
        # Ensure raise a configuration error if the context encoder input dim 
        # is not equal to the context_field_embedder output dim + target encoder
        # output.
        params = Params.from_file(self.atae_config).duplicate()
        params['model']['context_encoder']['input_size'] = 5
        params_copy = copy.deepcopy(params)
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=params_copy.get('model'))
        # This should be the same result for the AE config
        params = Params.from_file(self.ae_config).duplicate()
        params['model']['context_encoder']['input_size'] = 5
        params_copy = copy.deepcopy(params)
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=params_copy.get('model'))
        # Ensure that for the AT config that configuration error is raised 
        # if the context encoder input dim is not equal to the context_field_embdder
        params = Params.from_file(self.at_config).duplicate()
        params['model']['context_encoder']['input_size'] = 17
        params_copy = copy.deepcopy(params)
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=params_copy.get('model'))

    def test_inter_context(self):
        # Raises a config error if the output context encoder does not match 
        # the input of the inter target encoder
        params = Params.from_file(self.inter_atae_config).duplicate()
        params['model']['context_encoder']['hidden_size'] = 6
        params_copy = copy.deepcopy(params)
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=params_copy.get('model'))

    def test_requires_position_weights(self):
        # Raises a ValueError id the position_weights are not in the forward 
        # pass when the model config requires them
        params = Params.from_file(self.position_weight_atae_config).duplicate()
        del params['dataset_reader']['position_weights']
        params_copy = copy.deepcopy(params)
        Model.from_params(vocab=self.vocab, params=params_copy.get('model'))
        with tempfile.NamedTemporaryFile(mode='w+') as temp_file:
            params.to_file(temp_file.name)
            with pytest.raises(ValueError):
                self.ensure_model_can_train_save_and_load(temp_file.name)

    def test_inter_feedforward(self):
        # Raises a config error if the inter target encoder does not match 
        # the input of the feedforward
        params = Params.from_file(self.inter_atae_config).duplicate()
        params['model']['feedforward']['input_dim'] = 20
        params_copy = copy.deepcopy(params)
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=params_copy.get('model'))
        
    def test_forward_pass_runs_correctly(self):
        params = Params.from_file(self.param_file)   
        model = Model.from_params(vocab=self.vocab, 
                                  params=params.get('model'))
        training_tensors = self.dataset.as_tensor_dict()

        output_dict = model(**training_tensors)
        for key in output_dict.keys():
            assert key in {'class_probabilities', 'targets_mask',
                           'loss', 'words', 'text', 'targets', 'target words',
                           'word_attention', 'targets_attention','targets_word_mask',
                           'context_mask'}
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
    