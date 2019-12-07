from allennlp.common.params import Params
from allennlp.data.dataset import Batch
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn import InitializerApplicator, Initializer
from allennlp.models import Model
import numpy
import pytest

def loss_weights(param_file: str, vocab: Vocabulary, dataset: Batch) -> None:
    '''
    Check that when using the loss weights a different loss does appear. 
    Furthermore when the loss weights are 1., 1., 1. it should 
    be very close if not the same as not using the loss weights.

    :param param_file: A file path to a model configuration file
    :param vocab: A vocab of the model that will be loaded from the 
                  configuration file.
    :param dataset: A batch of the dataset to test the model on.
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