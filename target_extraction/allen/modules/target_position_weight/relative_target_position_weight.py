from typing import Tuple

from allennlp.common.checks import check_dimensions_match
import torch

from target_extraction.allen.modules.target_position_weight import TargetPositionWeight

@TargetPositionWeight.register("relative_target_position_weight")
class RelativeTargetPositionWeight(TargetPositionWeight):
    '''
    The weighting performed here is the following:
        
    $$1 - \frac{|\theta - i|}{n}$$
    
    Where $i$ is the location of the token, $\theta$ is the location of 
    the nearest target token (can be more than one taregt token in the sentence
    if the target is a multi-word target), and $n$ is the token length of 
    the text. The weight of the target tokens by default is $1$ thus target 
    tokens are not down weighted. This is the same weighting as equation 7 
    within `Chen et al. 2017 <https://www.aclweb.org/anthology/D17-1047/>`_
    and equation 2 in `Zhao et al. 2019 <https://arxiv.org/abs/1906.04501>`_

    :param zero_target_word_weighting: If True it will apply a weight of `0`
                                       to all target words (same as masking the 
                                       target words). This would be the same 
                                       weighting function as `Zhang et al. 2019 
                                       <https://www.aclweb.org/anthology/D19-1464.pdf>`_
    '''

    def __init__(self, zero_target_word_weighting: bool = False):
        super().__init__()
        self.zero_target_word_weighting = zero_target_word_weighting

    def forward(self, targets_features: torch.Tensor, 
                relative_target_positions: torch.Tensor, 
                sequence_mask: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:

        seq_batch_targets, _, _ = targets_features.shape
        batch_size, num_targets, sequence_length = relative_target_positions.shape
        batch_targets = batch_size * num_targets

        check_dimensions_match(batch_targets, seq_batch_targets, 
                               'target_features first dimension', 
                               'relative_target_positions new first dimension')

        relative_target_positions = relative_target_positions.type(torch.float32)
        # Want to make the target word positions to be zero rather than one
        relative_target_positions = relative_target_positions.view(batch_targets , -1)
        relative_target_positions = relative_target_positions - 1

        text_lengths = sequence_mask.sum(1)
        text_lengths = text_lengths.type(torch.float32)
        
        expanded_text_lengths = text_lengths.unsqueeze(-1).repeat(1,sequence_length)
        weighted_target_position = 1 - (relative_target_positions / expanded_text_lengths)
        if self.zero_target_word_weighting:
            not_target_token_positions = (relative_target_positions != 0).type(torch.float32)
            weighted_target_position = weighted_target_position * not_target_token_positions
        weighted_target_position = weighted_target_position * sequence_mask
        weighted_target_position[torch.isnan(weighted_target_position)] = 0.0
        weighted_target_features = targets_features * weighted_target_position.unsqueeze(-1)
        return weighted_target_features, weighted_target_position