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


def loss_weights(param_file: str, vocab: Vocabulary, dataset: Batch):
    '''
    Check that when using the loss weights a different loss does appear. 
    Furthermore when the loss weights are 1., 1., 1. it should 
    be very close if not the same as not using the loss weights.
    '''
    def get_loss(params: Params) -> float:
        # Required to remove the random initialization
        constant_init = Initializer.from_params(Params({"type": "constant", "val": 0.5}))
        initializer = InitializerApplicator([(".*", constant_init)])
        model = Model.from_params(vocab=vocab, 
                                  params=params.get('model'))
        initializer(model)
        training_tensors = dataset.as_tensor_dict()
        output_dict = model(**training_tensors)
        return output_dict['loss'].cpu().data.numpy()

    params = Params.from_file(param_file)   
    normal_loss = get_loss(params)

    params = Params.from_file(param_file).duplicate()
    params['model']['loss_weights'] = [0.2, 0.5, 0.1]
    weighted_loss = get_loss(params)
    with pytest.raises(AssertionError):
        numpy.testing.assert_array_almost_equal(normal_loss, weighted_loss, 2)

    params = Params.from_file(param_file).duplicate()
    params['model']['loss_weights'] = [1., 1., 1.]
    same_loss = get_loss(params)
    numpy.testing.assert_array_almost_equal(normal_loss, same_loss, 1)

class InteractivateAttentionNetworkClassifierTest(ModelTestCase):
    def setUp(self):
        super().setUp()

        test_dir = Path(__file__, '..', '..', '..', '..','data', 'allen',  
                        'models', 'target_sentiment').resolve()
        test_data = str(Path(test_dir, 'multi_target_category_sentiments.json'))
        config_dir = Path(test_dir, 'IAN')
        self.ian_config = str(Path(config_dir, 'ian_config.jsonnet'))
        self.ian_elmo_wordvector_config = str(Path(config_dir, 'ian_elmo_wordvector_config.jsonnet'))
        self.ian_elmo_config = str(Path(config_dir, 'ian_elmo_config.jsonnet'))

        self.set_up_model(self.ian_config,
                          test_data)

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_ian_elmo_train_save(self):
        params = Params.from_file(self.ian_elmo_config).duplicate()
        params_copy = copy.deepcopy(params)
        Model.from_params(vocab=self.vocab, params=params_copy.get('model'))
        with tempfile.NamedTemporaryFile(mode='w+') as temp_file:
            params.to_file(temp_file.name)
            self.ensure_model_can_train_save_and_load(temp_file.name)
    
    def test_ian_elmo_wordvector_train_save(self):
        params = Params.from_file(self.ian_elmo_wordvector_config).duplicate()
        params_copy = copy.deepcopy(params)
        Model.from_params(vocab=self.vocab, params=params_copy.get('model'))
        with tempfile.NamedTemporaryFile(mode='w+') as temp_file:
            params.to_file(temp_file.name)
            self.ensure_model_can_train_save_and_load(temp_file.name)

    @flaky
    def test_batch_predictions_are_consistent(self):
        ignore = ['words', 'text', 'targets', 'target words',
                  'word_attention', 'targets_attention']
        self.ensure_batch_predictions_are_consistent(keys_to_ignore=ignore)

    def test_target_field_embedder(self):
        # Test that can handle having a target embedder as well as a text
        # embedder
        params = Params.from_file(self.ian_config).duplicate()
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
    
    def test_target_encoder(self):
        # Ensure raise a configuration error if the target encoder input dim 
        # is not equal to the context_field_embedder output dim.
        params = Params.from_file(self.ian_config).duplicate()
        params['model']['target_encoder']['input_size'] = 4
        params_copy = copy.deepcopy(params)
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=params_copy.get('model'))

    def test_context_encoder(self):
        # Ensure raise a configuration error if the context encoder input dim 
        # is not equal to the context_field_embedder output dim.
        params = Params.from_file(self.ian_config).duplicate()
        params['model']['context_encoder']['input_size'] = 4
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
    