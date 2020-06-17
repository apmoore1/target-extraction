from pathlib import Path

from allennlp.common.testing import ModelTestCase
from allennlp.models import Model
from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.data import DatasetReader, Vocabulary
from allennlp.data import Batch
import pytest

import target_extraction

class TargetTaggerTest(ModelTestCase):

    def setUp(self):
        test_dir = Path(__file__, '..', '..', '..','data', 'allen')
        test_data_dir = Path(test_dir, 'dataset_readers', 'target_extraction')
        self.non_pos_data = str(Path(test_data_dir, 'non_pos_sequence.json').resolve())
        self.pos_data = str(Path(test_data_dir, 'pos_sequence.json').resolve())

        model_dir = Path(test_dir, 'models', 'target_extraction')
        self.crf_param_file = str(Path(model_dir, 'crf_config.json').resolve())
        self.softmax_param_file = str(Path(model_dir, 'softmax_config.json').resolve())
        self.no_encoder_file = str(Path(model_dir, 'no_encoder_config.json').resolve())

        self.non_pos_param_files = [self.crf_param_file, self.softmax_param_file,
                                    self.no_encoder_file]
        
        self.pos_crf_param_file = str(Path(model_dir, 'pos_crf_config.json').resolve())
        self.pos_softmax_param_file = str(Path(model_dir, 'pos_softmax_config.json').resolve())
        self.pos_embedding_param_file = str(Path(model_dir, 'pos_embedding_softmax_config.json').resolve())
        self.pos_param_files = [self.pos_crf_param_file, 
                                self.pos_softmax_param_file,
                                self.pos_embedding_param_file]

        self.set_up_model(self.crf_param_file, self.non_pos_data)
        super().setUp()

    def test_batch_predictions_are_consistent(self):
        # This only uses the CRF tagger with no POS data.
        self.ensure_batch_predictions_are_consistent()

    def test_crf_tagger_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.crf_param_file)

    def test_softmax_tagger_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.softmax_param_file)

    def test_crf_pos_tagger_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.pos_crf_param_file)

    def test_softmax_pos_tagger_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.pos_softmax_param_file)
    
    def test_softmax_pos_embedding_tagger_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.pos_embedding_param_file)

    def load_pos_dataset_vocab(self, pos_param_file):
        reader = DatasetReader.from_params(pos_param_file['dataset_reader'])
        instances = list(reader.read(str(self.pos_data)))
        vocab = Vocabulary.from_instances(instances)
        dataset = Batch(instances)
        dataset.index_instances(vocab)
        return dataset, vocab

    def test_forward_pass_runs_correctly(self):
        def test_param_file(param_file, pos_data):
            params = Params.from_file(param_file).duplicate()
            if pos_data:
                dataset, vocab = self.load_pos_dataset_vocab(params)
                model = Model.from_params(vocab=vocab, params=params.get('model'))
                training_tensors = dataset.as_tensor_dict()
            else:
                model = Model.from_params(vocab=self.vocab, params=params.get('model'))
                training_tensors = self.dataset.as_tensor_dict()

            output_dict = model(**training_tensors)
            for key in output_dict.keys():
                assert key in {'logits', 'mask', 'tags', 'class_probabilities', 
                               'loss', 'words', 'text'}
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
            text = output_dict['text']
            assert text[0] == "The laptop case was great and cover was rubbish"

        for param_file in self.non_pos_param_files:
            test_param_file(param_file, False)
        for param_file in self.pos_param_files:
            test_param_file(param_file, True)
            
    def test_label_encoding_required(self):
        params = Params.from_file(self.param_file).duplicate()
        params["model"].pop("label_encoding")
        params["model"].pop("calculate_span_f1")
        # constrain_crf_decoding requires label encoding
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=params.pop("model"))
        # calculate_span_f1 requires label encoding
        params = Params.from_file(self.param_file).duplicate()
        params["model"].pop("label_encoding")
        params["model"].pop("constrain_crf_decoding")
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=params.pop("model"))

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

    def test_pos_embedding_mismatching_dimensions_throws_configuration_error(self):
        params = Params.from_file(self.pos_embedding_param_file).duplicate()
        params["model"]["encoder"]["input_size"] = 55
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=params.pop("model"))
        params = Params.from_file(self.param_file).duplicate()
        params["model"].pop("encoder")
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=params.pop("model"))

    def test_no_pos_tags_in_vocab(self):
        for param_file in self.pos_param_files:
            params = Params.from_file(param_file).duplicate()
            with pytest.raises(ConfigurationError):
                Model.from_params(vocab=self.vocab, params=params.get('model'))
    
    def test_no_pos_tags_forward_training(self):
        # Data that does not contain POS tags
        training_tensors = self.dataset.as_tensor_dict()
        for param_file in self.pos_param_files:
            params = Params.from_file(param_file).duplicate()
            _, vocab = self.load_pos_dataset_vocab(params)
            model = Model.from_params(vocab=vocab, params=params.get('model'))
            with pytest.raises(ConfigurationError):
                model(**training_tensors)