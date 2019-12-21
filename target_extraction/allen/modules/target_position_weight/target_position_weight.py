from typing import Tuple

from allennlp.common import Registrable
import torch

class TargetPositionWeight(torch.nn.Module, Registrable):
    '''
    A ``TargetPositionWeight`` is a ``Module`` that represents different 
    methods that can weight a target sample's encoded text by the position 
    the tokens take in the text with respect to the target tokens.
    '''

    def forward(self, targets_features: torch.Tensor, 
                relative_target_positions: torch.Tensor, 
                sequence_mask: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        :param targets_features: A tensor of shape (batch * num_targets, 
                                 text_sequence_length, dim). This tensor 
                                 will be returned weighted by the position 
                                 of the tokens in the sequence with respect to
                                 the target tokens.
        :param relative_target_positions: A tensor of shape (batch, num_targets, 
                                          text_sequence_length). This will be 
                                          a tensor that contains the position 
                                          of each token to its associated 
                                          target tokens in the sample. 
        :param sequence_mask: A tensor of shape (batch * num_targets, 
                              text_sequence_length). The mask  
                              determines which tokens are to be weighted 
                              based on their position in the sequence.
        :returns: A tuple of two tensors 1. tensor of shape (batch * num_targets, 
                  text_sequence_length, dim), where the `target_features` have been weighted based 
                  on each tokens position to its sample's respective target 
                  token position. 2. tensor of shape (batch * num_targets, 
                  text_sequence_length) representing the weights that the 
                  `target_features` have been multipled by to get the first 
                  tensor in this tuple.
        :raises ConfigurationError: If the `targets_features` first dimension 
                                    is not `batch size * num targets` size.
        '''
        raise NotImplementedError