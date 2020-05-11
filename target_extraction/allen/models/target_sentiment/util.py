from collections import defaultdict
from typing import List, Union, Optional, Dict

from allennlp.data import TextFieldTensors
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder
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
        loss_name_weights = {label_name: weight for label_name, weight 
                             in zip(loss_weight_labels, loss_weights)}
        for label_name, index in label_name_index:
            temp_loss_weights.append(loss_name_weights[label_name])
        return temp_loss_weights
    else:
        return None

def elmo_input_reshape(inputs: TextFieldTensors, batch_size: int,
                       number_targets: int, batch_size_num_targets: int
                       ) -> TextFieldTensors:
    '''
    NOTE: This does not work for the hugginface transformers as when they are 
    processed by the token indexers they produce additional key other than 
    token ids such as mask ids and segment ids that also need handling, of 
    which we have not had time to handle this yet. A way around this, which 
    would be more appropriate, would be to use `target_sequences` like in the 
    `InContext` model, to generate contextualised targets from the context rather 
    than using the target words as is without context.

    :param inputs: The token indexer dictionary where the keys state the token 
                   indexer and the values are the Tensors that are of shape 
                   (Batch Size, Number Targets, Sequence Length)
    :param batch_size: The Batch Size
    :param number_targets: The max number of targets in the batch
    :param batch_size_num_targets: Batch Size * number of targets
    :returns: If the inputs contains a `elmo` or 'token_characters' key it will 
              reshape all the keys values into shape 
              (Batch Size * Number Targets, Sequence Length) so that it can be 
              processed by the ELMO or character embedder/encoder. 
    '''
    if 'elmo' in inputs or 'token_characters' in inputs:
        temp_inputs: TextFieldTensors = defaultdict(dict)
        for key, inner_key_value in inputs.items():
            for inner_key, value in inner_key_value.items():
                temp_value = value.view(batch_size_num_targets, *value.shape[2:])
                temp_inputs[key][inner_key] = temp_value
        return dict(temp_inputs)
    else:
        return inputs

def elmo_input_reverse(embedded_input: torch.Tensor, 
                       inputs: TextFieldTensors, batch_size: int,
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
    :returns: If the inputs contains a `elmo` or 'token_characters' key it will 
              reshape the `embedded_input` into the original shape of 
              (Batch Size, Number Targets, Sequence Length, embedding dim)
    '''
    if 'elmo' in inputs or 'token_characters' in inputs:
        return embedded_input.view(batch_size, number_targets,
                                   *embedded_input.shape[1:])
    return embedded_input

def concat_position_embeddings(embedding_context: torch.Tensor, 
                               position_indexes: Optional[Dict[str, torch.LongTensor]] = None,
                               target_position_embedding: Optional[TextFieldEmbedder] = None
                               ) -> torch.Tensor:
    '''
    :param embedding_context: Tensor of shape (batch size * number targets, 
                              context sequence length, context dim)
    :param position_indexes: Dictionary of token indexer name to a 
                             Tensor of shape (batch size, number targets, 
                             text sequence length)
    :param target_position_embedding: An embedding function for the position 
                                      indexes, where the dimension of the position 
                                      embedding is `position dim`.
    :returns: If `position_indexes` and `target_position_embedding` are None then
              the `embedding_context` is returned without any change. Else the 
              relevant position embeddings are concatenated onto the relevant 
              token embeddings within the `embedding_context` to create a 
              Tensor of shape (batch size * number targets, text sequence length, 
              context dim + position dim)
    :raises ValueError: If `target_position_embedding` is not None when 
                        `position_indexes` is None.
    '''
    if target_position_embedding:
        if position_indexes is None:
            raise ValueError('This model requires `position_indexes` as '
                             'input to the `target_position_embedding` '
                             'forward function to get the position embeddings')
        target_position_embeddings = target_position_embedding(position_indexes)
        batch_size_num_targets, context_sequence_length, context_dim = embedding_context.shape
        position_embedding_dim = target_position_embeddings.shape[-1]
        # re-shape position_embeddings
        target_position_embeddings = target_position_embeddings.view(batch_size_num_targets, 
                                                                     context_sequence_length, 
                                                                     position_embedding_dim)
        embedding_context = torch.cat((embedding_context, 
                                       target_position_embeddings), -1)
    return embedding_context