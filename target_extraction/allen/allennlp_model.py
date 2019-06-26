import collections
from typing import Optional, List, Any, Iterable, Dict, Union
import json
import tempfile
from pathlib import Path
import random

from allennlp.common.params import Params
from allennlp.commands.train import train_model_from_file
from allennlp.commands.find_learning_rate import find_learning_rate_model
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
#from bella.data_types import TargetCollection
import numpy as np

import target_extraction
from target_extraction.data_types import TargetTextCollection
#from bella_allen_nlp.predictors.target_predictor import TargetPredictor

class AllenNLPModel():
    '''
    This is a wrapper for the AllenNLP dataset readers, models, and predictors 
    so that the input to functions can be 
    :class:`target_extraction.data_types.TargetTextCollection` objects
    and the return a metric or metrics as well as predicitons within the 
    :class:`target_extraction.data_types.TargetTextCollection` objects. This 
    is instead of running everything through multiple bash files calling 
    ``allennlp train`` etc.
    '''

    def __init__(self, name: str, model_param_fp: Path, predictor_name: str, 
                 save_dir: Optional[Path] = None) -> None:
        '''
        :param name: Name of the model e.g. ELMO-Target-Extraction
        :param model_params_fp: File path to the model parameters that will 
                                define the AllenNLP model and how to train it.
        :param predictor_name: Name of the predictor to be used with the 
                               AllenNLP model e.g. for a target_tagger model 
                               the predictor should prbably be `target-tagger`
        :param save_dir: Directory to save the model to. This has to be set
                         up front as the fit function saves the model each 
                         epoch.
        '''

        self.name = name
        self.model = None
        self.save_dir = save_dir
        self._fitted = False
        self._param_fp = model_param_fp.resolve()
        self._predictor_name = predictor_name
        #self.labels = None

    def fit(self, train_data: TargetTextCollection, 
            val_data: TargetTextCollection,
            test_data: Optional[TargetTextCollection] = None) -> None:
        '''
        Given the training, validation, and optionally the test data it will 
        train the model that is defined in the model params file provided as 
        argument to the constructor of the class. Once trained the model can 
        be accessed through the `model` attribute.

        NOTE: If the test data is given the model only uses it to fit to the 
        vocabularly that is within the test data, the model NEVER trains on 
        the test data.
        
        :param train_data: Training data.
        :param val_data: Validation data.
        :param test_data: Optional, test data.
        '''

        model_params = self._preprocess_and_load_param_file(self._param_fp)
        # Ensures that a different random seed is used each time
        self._set_random_seeds(model_params)
        with tempfile.TemporaryDirectory() as temp_dir:
            train_fp = Path(temp_dir, 'train_data.json')
            val_fp = Path(temp_dir, 'val_data.json')

            # Write the training and validation data to json Optionally test as 
            # well
            train_data.to_json_file(train_fp)
            val_data.to_json_file(val_fp)
            if test_data:
                test_fp = Path(temp_dir, 'test_data.json')
                test_data.to_json_file(test_fp)
                self._add_dataset_paths(model_params, train_fp, val_fp, test_fp)
                model_params["evaluate_on_test"] = True
            else:
                self._add_dataset_paths(model_params, train_fp, val_fp)

            save_dir = self.save_dir
            if save_dir is None:
                save_dir = Path(temp_dir, 'temp_save_dir')
            
            temp_param_fp = Path(temp_dir, 'temp_param_file.json')
            model_params.to_file(temp_param_fp.resolve())
            trained_model = train_model_from_file(temp_param_fp, save_dir)
            self.model = trained_model
            #self.labels = self._get_labels()
        self.fitted = True

    def _predict_iter(self, data: Union[Iterable[Dict[str, Any]], 
                                        List[Dict[str, Any]]]
                      ) -> Iterable[Dict[str, Any]]:
        '''
        Iterates over the predictions and yields one prediction at a time.
        This is a useful wrapper as it performs the data pre-processing and 
        assertion checks.

        The predictions are predicted in batchs so that the model does not 
        load in lots of data at once and thus have memory issues.

        :param data: Iterable or list of dictionaries that the predictor can 
                     take as input e.g. `target-tagger` predictor expects at 
                     most a `text` key and value.
        :yields: A dictionary containing all the values the model outputs e.g.
                 For the `target_tagger` model it would return `logits`, 
                 `class_probabilities`, `mask`, and `tags`.
        :raises AssertionError: If the `model` attribute is None. This can be 
                                overcome by either fitting or loading a model.
        :raises TypeError: If the data given is not of Type List or Iterable.
        '''
        no_model_error = 'There is no model to make predictions, either fit '\
                         'or load a model to resolve this.'
        assert self.model, no_model_error
        self.model.eval()

        all_model_params = Params.from_file(self._param_fp)

        reader_params = all_model_params.get("dataset_reader")
        dataset_reader = DatasetReader.from_params(reader_params)
        predictor = Predictor.by_name(self._predictor_name)(self.model, dataset_reader)

        batch_size = 64
        if 'iterator' in all_model_params:
            iter_params = all_model_params.get("iterator")
            if 'batch_size' in iter_params:
                batch_size = iter_params['batch_size']
        
        # Data has to be an iterator
        if isinstance(data, list) or isinstance(data, collections.Iterable):
            data = iter(data)
        else:
            raise TypeError(f'Data given has to be of type {collections.Iterable}'
                            f' and not {type(data)}')
        data_exists = True
        while data_exists:
            data_batch = []
            for _ in range(batch_size):
                try:
                    data_batch.append(next(data))
                except StopIteration:
                    data_exists = False
            if data_batch:
                predictions = predictor.predict_batch_json(data_batch)
                for prediction in predictions:
                    yield prediction

    def predict_sequences(self, data: Union[Iterable[Dict[str, Any]], 
                                            List[Dict[str, Any]]]
                          ) -> Iterable[Dict[str, Any]]:
        '''
        Given the data it will predict the sequence labels and return the 
        confidence socres in those labels as well.

        :param data: Iterable or list of dictionaries that contains at least 
                     `text` key and value and if you do not want the 
                     predictor to do the tokenization then provide `tokens` 
                     as well. Some model may also expect `pos_tags` which the 
                     predictor will provide if the `text` key is only provided.
        :yields: A dictionary containing all the following keys and values:
                 1. `sequence_labels`: A list of predicted sequence labels. 
                    This will be a List of Strings.
                 2. `confidence`: The confidence the model had in predicting 
                    each sequence label, this comes from the softmax score.
                    This will be a List of floats.
        '''
        self.model: Model
        label_to_index = self.model.vocab.get_token_to_index_vocabulary('labels')
        for prediction in self._predict_iter(data):
            output_dict = {}
            # Length of the text
            sequence_length = sum(prediction['mask'])
            
            # Sequence labels
            sequence_labels = prediction['tags'][:sequence_length]
            output_dict['sequence_labels'] = sequence_labels
            
            # Confidence scores
            # First get the index of predicted lables
            confidence_indexs = [label_to_index[label] for label in sequence_labels]
            confidence_scores = prediction['class_probabilities'][:sequence_length]
            label_confidence_scores = [] 
            for scores, index in zip(confidence_scores, confidence_indexs):
                print(index)
                print(sequence_labels)
                print(scores)
                print(scores[index])
                label_confidence_scores.append(scores[index])
            output_dict['confidence'] = label_confidence_scores

            yield output_dict
    #def predict(self, data: TargetCollection) -> np.ndarray:
    #    '''
    #    Given the data to predict with return a matrix of shape 
    #    [n_samples, n_classes] where the predict class will be one and all 
    #    others 0.
    #    To get the class label for these predictions use the `labels` attribute.
    #    The index of the predicted class is associated to the index within the 
    #    `labels` attribute.
    #    :param data: Data to predict on.
    #    :returns: A matrix of shape [n_samples, n_classes]
    #    '''
    #    predictions = self._predict_iter(data)

    #    n_samples = len(data)
    #    n_classes = len(self.labels)
    #    predictions_matrix = np.zeros((n_samples, n_classes))
    #    for index, prediction in enumerate(predictions):
    #        class_probabilities = prediction['class_probabilities']
    #        class_label = np.argmax(class_probabilities)
    #        predictions_matrix[index][class_label] = 1
    #    return predictions_matrix

    #def predict_label(self, data: TargetCollection, 
    #                  mapper: Optional[Dict[str, Any]] = None) -> np.ndarray:
    #    '''
    #    Given the data to predict with return a vector of class labels.
    #    Optionally a mapper dictionary can be given to map the class labels 
    #    to a different label e.g. {'positive': 1, 'neutral': 0, 'negative': -1}.
        
    #    :param data: Data to predict on.
    #    :returns: A vector of shape [n_samples]
    #    '''
    #    predictions = self._predict_iter(data)

    #    predictions_list = []
    #    for prediction in predictions:
    #        label = prediction['label']
    #        if mapper:
    #            label = mapper[label]
    #        predictions_list.append(label)
    #    return np.array(predictions_list)

    #def probabilities(self, data: TargetCollection) -> np.ndarray:
    #    '''
    #    Returns the probability for each class for every sample in the data. 
    #    The returned matrix is of shape [n_samples, n_classes]
    #    :param data: Data to predict on
    #    :returns: probabilities that a class is true for each class for each 
    #              sample. 
    #    '''
        
    #    predictions = self._predict_iter(data)

    #    n_samples = len(data)
    #    n_classes = len(self.labels)
    #    probability_matrix = np.zeros((n_samples, n_classes))
    #    for index, prediction in enumerate(predictions):
    #        class_probabilities = prediction['class_probabilities']
    #        probability_matrix[index] = class_probabilities
    #    return probability_matrix

    def load(self, cuda_device: int = -1) -> Model:
        '''
        Loads the model. This does not require you to train the model if the 
        `save_dir` attribute is pointing to a folder containing a trained model.
        This is just a wrapper around the `load_archive` function.

        :param cuda_device: Whether the loaded model should be loaded on to the 
                            CPU (-1) or the GPU (0). Default CPU.
        :returns: The model that was saved at `self.save_dir` 
        :raises AssertionError: If the `save_dir` argument is None
        :raises FileNotFoundError: If the save directory does not exist.
        '''

        save_dir_err = 'Save directory was not set in the constructor of the class'
        assert self.save_dir, save_dir_err
        if self.save_dir.exists():
            archive = load_archive(self.save_dir / "model.tar.gz", 
                                   cuda_device=cuda_device)
            self.model = archive.model
            #self.labels = self._get_labels()
            return self.model
        raise FileNotFoundError('There is nothing at the save dir:\n'
                                f'{self.save_dir.resolve()}')

    #def find_learning_rate(self, train_data: TargetCollection,
    #                       results_dir: Path, 
    #                       find_lr_kwargs: Optional[Dict[str, Any]] = None
    #                       ) -> None:
    #    '''
    #    Given the training data it will plot learning rate against loss to allow 
    #    you to find the best learning rate.
    #    This is just a wrapper around 
    #    allennlp.commands.find_learning_rate.find_learning_rate_model method.
    #    :param train_data: Training data.
    #    :param results_dir: Directory to store the results of the learning rate
    #                        findings.
    #    :param find_lr_kwargs: Dictionary of keyword arguments to give to the 
    #                           allennlp.commands.find_learning_rate.find_learning_rate_model
    #                           method.
    #    '''
    #    model_params = self._preprocess_and_load_param_file(self._param_fp)
    #    with tempfile.TemporaryDirectory() as temp_dir:
    #        train_fp = Path(temp_dir, 'train_data.json')
    #        self._data_to_json(train_data, train_fp)
    #        model_params['train_data_path'] = str(train_fp.resolve())
    #        if find_lr_kwargs is None:
    #            find_learning_rate_model(model_params, results_dir)
    #        else:
    #            find_learning_rate_model(model_params, results_dir, **find_lr_kwargs)

    #def _get_labels(self) -> List[Any]:
    #    '''
    #    Will return all the possible class labels that the model attribute 
    #    can generate.
    #    :returns: List of possible labels the model can generate
    #    '''
    #    vocab = self.model.vocab
    #    label_dict = vocab.get_index_to_token_vocabulary('labels')
    #    return [label_dict[i] for i in range(len(label_dict))]
        
    #@staticmethod
    #def _data_to_json(data: TargetTextCollection, file_path: Path) -> None:
    #    '''
    #    Converts the data into json format and saves it to the given file path. 
    #    The AllenNLP models read the data from json formatted files.
    #    :param data: data to be saved into json format.
    #    :param file_path: file location to save the data to.
    #    '''
    #    target_data = data.data_dict()
    #    TargetTextCollection.to_json_file(file)
    #    with file_path.open('w+') as json_file:
    #        for index, data in enumerate(target_data):
    #            json_encoded_data = json.dumps(data)
    #            if index != 0:
    #                json_encoded_data = f'\n{json_encoded_data}'
    #            json_file.write(json_encoded_data)

    @staticmethod
    def _preprocess_and_load_param_file(model_param_fp: Path) -> Params:
        '''
        Given a model parameter file it will load it as a Params object and 
        remove all data fields for the Param object so that these keys can be 
        added with different values associated to them.
        fields (keys) that are removed:
        1. train_data_path
        2. validation_data_path
        3. test_data_path
        4. evaluate_on_test
        :param model_param_fp: File path to the model parameters that will 
                               define the AllenNLP model and how to train it.
        :returns: The model parameter file as a Params object with the data 
                  fields removed if they exisited.
        '''

        model_param_fp = str(model_param_fp)
        fields_to_remove = ['train_data_path', 'validation_data_path', 
                            'test_data_path', 'evaluate_on_test']
        model_params = Params.from_file(model_param_fp)
        for field in fields_to_remove:
            if field in model_params:
                model_params.pop(field)
        return model_params

    @staticmethod
    def _add_dataset_paths(model_params: Params, train_fp: Path, val_fp: Path, 
                           test_fp: Optional[Path] = None) -> None:
        '''
        Give model parameters it will add the given train, validation and 
        optional test dataset paths to the model parameters.
        Does not return anything as the model parameters object is mutable
        :param model_params: model parameters to add the dataset paths to
        :param train_fp: Path to the training dataset
        :param val_fp: Path to the validation dataset
        :param test_fp: Optional path to the test dataset
        '''

        model_params['train_data_path'] = str(train_fp.resolve())
        model_params['validation_data_path'] = str(val_fp.resolve())
        if test_fp:
            model_params['test_data_path'] = str(test_fp.resolve())

    @staticmethod
    def _set_random_seeds(model_params: Params) -> None:
        '''
        This ensures to some extent that the experiments are NOT reproducible 
        so that we can take into account the random seed problem.
        Returns nothing as the model_params will be modified as they are a 
        mutable object.
        :param model_params: The parameters of the model
        '''

        seed, numpy_seed, torch_seed = [random.randint(1,99999) 
                                        for i in range(3)]
        model_params["random_seed"] = seed
        model_params["numpy_seed"] = numpy_seed
        model_params["pytorch_seed"] = torch_seed     

    def __repr__(self) -> str:
        '''
        :returns: the name of the model e.g. TDLSTM or IAN
        '''
        return self.name

    @property
    def fitted(self) -> bool:
        '''
        If the model has been fitted (default False)
        :return: True or False
        '''

        return self._fitted

    @fitted.setter
    def fitted(self, value: bool) -> None:
        '''
        Sets the fitted attribute
        :param value: The value to assign to the fitted attribute
        '''

        self._fitted = value