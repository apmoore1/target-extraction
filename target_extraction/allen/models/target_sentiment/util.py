from typing import List, Union, Optional

from allennlp.models import Model

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