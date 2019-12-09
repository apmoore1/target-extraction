from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from overrides import overrides
import torch

from ..inter_target.inter_target import InterTarget


@InterTarget.register("sequence_inter_target")
class SequenceInterTarget(InterTarget):
    def __init__(self, sequence_encoder: Seq2SeqEncoder) -> None:
        '''
        :param sequence_encoder: The sequence encoder to be used within forward.
        '''
        super().__init__()
        self.sequence_encoder = sequence_encoder

    @overrides
    def forward(self, targets_features: torch.Tensor, mask: torch.Tensor
                ) -> torch.Tensor:
        '''
        :param targets_features: A tensor of shape (batch, num_targets, dim)
        :param mask: A tensor of shape (batch, num_targets). The mask determines 
                     which targets are padding and which are not `0` indicates 
                     padding.
        :returns: A tensor of shape (batch, num_targets, dim), where the 
                  features from the others targets are encoded into each other 
                  through the `sequence_encoder` e.g. LSTM, where in the case 
                  of an LSTM it encodes each target starting 
                  from the first (left most) target to the last (right most) 
                  target in the text. If Bi-Directional then the LSTM will also 
                  encode from the last to the first target in the text. This 
                  type of encoding is a generalisation of `Modeling Inter-Aspect 
                  Dependencies for Aspect-Based Sentiment 
                  Analysis <https://www.aclweb.org/anthology/N18-2043/>`_, from 
                  that paper it would model equation 4. 
        '''
        return self.sequence_encoder(targets_features, mask)

    @overrides
    def get_input_dim(self) -> int:
        '''
        :returns: The dim size of the input to forward which is of shape 
                  (batch, num_target, dim)
        '''
        return self.sequence_encoder.get_input_dim()

    @overrides
    def get_output_dim(self) -> int:
        '''
        :returns: The dim size of the return from forward which is of shape 
                  (batch, num_target, dim)
        '''
        return self.sequence_encoder.get_output_dim()