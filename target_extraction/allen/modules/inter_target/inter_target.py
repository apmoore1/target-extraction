from allennlp.common import Registrable
import torch

class InterTarget(torch.nn.Module, Registrable):
    '''
    A ``InterTarget`` is a ``Module`` that takes as input a 
    tensor of shape (batch, num_targets, dim), where the tensor represents 
    the features for each target within a text. The output is the same shape 
    tensor (batch, num_targets, dim) but where each target has been encoded 
    with some information from its surrounding targets within the same 
    text.
    '''

    def forward(self, targets_features: torch.Tensor, mask: torch.Tensor
                ) -> torch.Tensor:
        '''
        :param targets_features: A tensor of shape (batch, num_targets, dim)
        :param mask: A tensor of shape (batch, num_targets). The mask determines 
                     which targets are padding and which are not `0` indicates 
                     padding.
        :returns: A tensor of shape (batch, num_targets, dim), where the 
                  features from the others targets have been encoded within
                  each other through this model.
        '''
        raise NotImplementedError

    def get_input_dim(self) -> int:
        '''
        :returns: The dim size of the input to forward which is of shape 
                  (batch, num_target, dim)
        '''
        raise NotImplementedError

    def get_output_dim(self) -> int:
        '''
        :returns: The dim size of the return from forward which is of shape 
                  (batch, num_target, dim)
        '''
        raise NotImplementedError