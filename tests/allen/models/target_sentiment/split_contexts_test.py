from pathlib import Path

from allennlp.common.testing import ModelTestCase
from allennlp.models import Model
from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.data import DatasetReader, Vocabulary
from allennlp.data.dataset import Batch
import pytest

import target_extraction

import copy
from pathlib import Path

import pytest

from allennlp.common.testing import ModelTestCase
from allennlp.common.params import Params
from allennlp.models import Model
from flaky import flaky
from allennlp.common.checks import ConfigurationError

class SplitContextsClassifierTest(ModelTestCase):
    def setUp(self):
        super().setUp()

        test_dir = Path(__file__, '..', '..', '..', '..','data', 'allen',  
                        'models', 'target_sentiment').resolve()
        test_data = str(Path(test_dir, 'target_category_sentiments.json'))
        self.tdlstm_config = str(Path(test_dir, 'tdlstm_config.jsonnet'))
        self.tclstm_config = str(Path(test_dir, 'tclstm_config.jsonnet'))

        self.set_up_model(self.tdlstm_config,
                          test_data)

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    @flaky
    def test_batch_predictions_are_consistent(self):
        self.ensure_batch_predictions_are_consistent()

    def test_tclstm_version(self):
        # Test the normal case
        self.ensure_model_can_train_save_and_load(self.tclstm_config)
        # Test that an error raises if the left text encoder does not have an 
        # input dimension that is equal to the context text word embeddings + 
        # the output dimension of the target encoder.
        params = Params.from_file(self.tclstm_config)
        params["model"]["left_text_encoder"]["embedding_dim"] = 10
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=params.get('model'))
        # Test the right text encoder
        params = Params.from_file(self.tclstm_config)
        params["model"]["right_text_encoder"]["embedding_dim"] = 10
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=params.get('model'))
        # Test the target encoder
        params = Params.from_file(self.tclstm_config)
        params["model"]["target_encoder"]["embedding_dim"] = 5
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=params.get('model'))

    def test_target_field_embedder(self):
        # Test that can handle having a target embedder as well as a text
        # embedder
        params = Params.from_file(self.tclstm_config).duplicate()
        target_embedder = {"token_embedders": {"tokens": {"type": "embedding",
                                                          "embedding_dim": 15,
                                                          "trainable": False}}}
        params['model']['target_field_embedder'] = target_embedder
        params_copy = copy.deepcopy(params)
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=params_copy.get('model'))
        params['model']['target_encoder']['embedding_dim'] = 15
        params['model']['left_text_encoder']['embedding_dim'] = 25
        params['model']['right_text_encoder']['embedding_dim'] = 25
        params_copy = copy.deepcopy(params)
        Model.from_params(vocab=self.vocab, params=params_copy.get('model'))


