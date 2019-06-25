import copy
from pathlib import Path
import tempfile

from allennlp.models.model import Model
from allennlp.common.params import Params
import pytest

from target_extraction.allen import AllenNLPModel
from target_extraction.data_types import TargetTextCollection

class TestAllenNLPModel():

    test_data_dir = Path(__file__, '..', '..','data', 'allen', 'dataset_readers',
                         'target_extraction').resolve()
    TARGET_EXTRACTION_TRAIN_DATA = Path(test_data_dir, 'non_pos_sequence.json')
    TARGET_EXTRACTION_TEST_DATA = Path(test_data_dir, 'non_pos_test_sequence.json')
    
    model_dir = Path(__file__, '..', '..','data', 'allen', 'predictors',
                     'target_tagger').resolve()
    TARGET_EXTRACTION_MODEL = Path(model_dir, 'non_pos_model', 'model.tar.gz')
    
    CONFIG_FILE = Path(model_dir, 'config_char.json')


    def test_repr_(self):
        model = AllenNLPModel('ML', self.CONFIG_FILE)
        model_repr = model.__repr__()
        assert model_repr == 'ML'

    def test_fitted_attr(self):
        model = AllenNLPModel('ML', self.CONFIG_FILE)
        assert not model.fitted

        model.fitted = True
        assert model.fitted 

    @pytest.mark.parametrize("test_data", (True, False))
    def test_target_extraction_fit(self, test_data: bool):
        
        model = AllenNLPModel('TE', self.CONFIG_FILE)
        assert not model.fitted
        assert model.model is None 
        
        train_data = TargetTextCollection.load_json(self.TARGET_EXTRACTION_TRAIN_DATA)
        val_data = TargetTextCollection.load_json(self.TARGET_EXTRACTION_TRAIN_DATA)
        
        tokens_in_vocab = ['at', 'case', 'was', 'the', 'day', 'great', 'cover', 
                           'office', 'another', 'and', 'rubbish', 'laptop',
                           '@@PADDING@@', '@@UNKNOWN@@']
        if test_data:
            tokens_in_vocab = tokens_in_vocab + ['better']
            test_data = TargetTextCollection.load_json(self.TARGET_EXTRACTION_TEST_DATA)
            model.fit(train_data, val_data, test_data)
        else:
            model.fit(train_data, val_data)
        
        token_index = model.model.vocab.get_token_to_index_vocabulary('tokens')
        assert len(token_index) == len(tokens_in_vocab)
        for token in tokens_in_vocab:
            assert token in token_index
        
        # Check attributes have changed.
        assert model.model is not None
        assert isinstance(model.model, Model)
        assert model.fitted

        # Check that it will save to a directory of our choosing
        with tempfile.TemporaryDirectory() as save_dir:
            saved_model_fp = Path(save_dir, 'model.tar.gz')
            assert not saved_model_fp.exists()
            model = AllenNLPModel('TE', self.CONFIG_FILE, save_dir=save_dir)
            model.fit(train_data, val_data)
            assert saved_model_fp.exists()

    def test_set_random_seeds(self):
        # test the case where the params is empty
        empty_params = Params({})
        assert len(empty_params) == 0
        AllenNLPModel._set_random_seeds(empty_params)
        assert len(empty_params) == 3
        seed_keys = ["random_seed", "numpy_seed", "pytorch_seed"]
        for key in seed_keys:
            assert isinstance(empty_params[key], int)
            assert empty_params[key] in range(1,99999)
        
        # test the case where the param is not empty and contain the seed keys
        original_values = {"random_seed": 599999, "numpy_seed": 599999,
                           "pytorch_seed": 799999}
        seed_params = Params(copy.deepcopy(original_values))
        assert len(seed_params) == 3
        AllenNLPModel._set_random_seeds(seed_params)
        for key, value in original_values.items():
            assert value != seed_params[key]
            assert seed_params[key] in range(1,99999)
    
    def test_preprocess_and_load_param_file(self):
        # Test that it does nothing to an empty params object
        with tempfile.TemporaryDirectory() as temp_dir:
            empty_param_fp = Path(temp_dir, 'empty')
            empty_params = Params({})
            assert len(empty_params) == 0
            empty_params.to_file(str(empty_param_fp))
            empty_params = AllenNLPModel._preprocess_and_load_param_file(empty_param_fp)
            assert len(empty_params) == 0
            assert isinstance(empty_params, Params)

            full_params_fp = Path(temp_dir, 'full')
            full_params = Params({'train_data_path': 1, 'validation_data_path': 1, 
                                'test_data_path': 1, 'evaluate_on_test': 1,
                                'anything': 2})
            assert len(full_params) == 5
            full_params.to_file(str(full_params_fp))
            full_params = AllenNLPModel._preprocess_and_load_param_file(full_params_fp)
            assert len(full_params) == 1
            assert full_params['anything'] == 2

    @pytest.mark.parametrize("test_data", (True, False))
    def test_add_dataset_paths(self, test_data: bool):
        # Test the case where the params are empty and it should populate the 
        # params
        empty_params = Params({})
        
        train_fp = Path(__file__, '..', 'models', 'target_tagger_test.py')
        train_str = str(train_fp.resolve())
        
        val_fp = Path(__file__, '..', 'dataset_readers', 'target_extraction_test.py')
        val_str = str(val_fp.resolve())
        
        test_fp = Path(__file__, '..', 'predictors', 'target_tagger_predictor_test.py')
        test_str = str(test_fp.resolve())

        assert len(empty_params) == 0
        if test_data:
            AllenNLPModel._add_dataset_paths(empty_params, train_fp, val_fp, test_fp)
            assert len(empty_params) == 3
            assert empty_params['train_data_path'] == train_str
            assert empty_params['validation_data_path'] == val_str
            assert empty_params['test_data_path'] == test_str
        else:
            AllenNLPModel._add_dataset_paths(empty_params, train_fp, val_fp)
            assert len(empty_params) == 2
            assert empty_params['train_data_path'] == train_str
            assert empty_params['validation_data_path'] == val_str
        

        # Test when the params were not empty
        full_params = Params({'train_data_path': 'something', 'another': 1})
        assert len(full_params) == 2
        assert full_params['train_data_path'] == 'something'
        if test_data:
            AllenNLPModel._add_dataset_paths(full_params, train_fp, val_fp, test_fp)
            assert len(full_params) == 4
            assert full_params['train_data_path'] == train_str
            assert full_params['validation_data_path'] == val_str
            assert full_params['test_data_path'] == test_str 
            assert full_params['another'] == 1
        else:
            AllenNLPModel._add_dataset_paths(full_params, train_fp, val_fp)
            assert len(full_params) == 3
            assert full_params['train_data_path'] == train_str
            assert full_params['validation_data_path'] == val_str 
            assert full_params['another'] == 1