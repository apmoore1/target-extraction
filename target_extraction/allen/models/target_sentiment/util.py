from typing import List, Union, Optional, Dict

from allennlp.models import Model
import torch

def loss_weight_order(model: Model, loss_weights: Optional[List[float]], 
                      label_name: str) -> Union[None, List[float]]:
    '''
    :param model: The model that you want to know the loss weights for. Requires 
                  a vocab.
    :param loss_weights: The loss weights to give to the labels. Can be None and 
                         if so returns None.
    :param label_name: The name of the vocab for the label that is to be 
                       predicted.
    :returns: None if `loss weights` is None. Else return a list of weights to 
              give to each label, where the original loss weights are ordered 
              by `['negative', 'neutral', 'positive']` and the returned are 
              ordered by occurrence in the models vocab for that `label_name` 
    '''
    if loss_weights is not None:
        label_name_index = model.vocab.get_token_to_index_vocabulary(namespace=label_name)
        label_name_index = sorted(label_name_index.items(), key=lambda x: x[1])
        
        temp_loss_weights = []
        loss_weight_labels = ['negative', 'neutral', 'positive']
        loss_weights = {label_name: weight for label_name, weight 
                        in zip(loss_weight_labels, loss_weights)}
        for label_name, index in label_name_index:
            temp_loss_weights.append(loss_weights[label_name])
        return temp_loss_weights
    else:
        return None

def elmo_input_reshape(inputs: Dict[str, torch.Tensor], batch_size: int,
                       number_targets: int, batch_size_num_targets: int
                       ) -> Dict[str, torch.Tensor]:
    '''
    :param inputs: The token indexer dictionary where the keys state the token 
                   indexer and the values are the Tensors that are of shape 
                   (Batch Size, Number Targets, Sequence Length)
    :param batch_size: The Batch Size
    :param number_targets: The max number of targets in the batch
    :param batch_size_num_targets: Batch Size * number of targets
    :returns: If the inputs contains a `elmo` key it will reshape all the keys 
              values into shape (Batch Size * Number Targets, Sequence Length)
              so that it can be processed by the ELMO embedder. 
    '''
    if 'elmo' not in inputs:
        return inputs
    temp_inputs: Dict[str, torch.Tensor] = {}
    for key, value in inputs.items():
        temp_value = value.view(batch_size_num_targets, *value.shape[2:])
        temp_inputs[key] = temp_value
    return temp_inputs

def elmo_input_reverse(embedded_input: torch.Tensor, 
                       inputs: Dict[str, torch.Tensor], batch_size: int,
                       number_targets: int, batch_size_num_targets: int
                       ) -> torch.Tensor:
    '''
    :param embedded_input: The embedding generated after the embedder has been 
                           forwarded over the `inputs`
    :param inputs: The token indexer dictionary where the keys state the token 
                   indexer and the values are the Tensors that are of shape 
                   (Batch Size, Number Targets, Sequence Length)
    :param batch_size: The Batch Size
    :param number_targets: The max number of targets in the batch
    :param batch_size_num_targets: Batch Size * number of targets
    :returns: If the inputs contains a `elmo` key it will reshape the 
              `embedded_input` into the original shape of 
              (Batch Size, Number Targets, Sequence Length, embedding dim)
    '''
    if 'elmo' not in inputs:
        return embedded_input
    return embedded_input.view(batch_size, number_targets,
                               *embedded_input.shape[1:])