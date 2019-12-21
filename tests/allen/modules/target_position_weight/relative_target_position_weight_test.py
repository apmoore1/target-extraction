from allennlp.common.checks import ConfigurationError
import pytest
import torch

from target_extraction.allen.modules.target_position_weight import RelativeTargetPositionWeight

@pytest.mark.parametrize('zero_target_word_weighting', (True, False))
def test_forward(zero_target_word_weighting: bool):
    batch_size = 2
    number_targets = 2
    text_seq_length = 3
    encoded_dim = 4
    
    encoded_text_tensor = [[[0.5,0.3,0.2,0.6], [0.2,0.3,0.4,0.7], [0.5,0.4,0.6,0.2]], 
                           [[0.4,0.5,0.3,0.7], [0.3,0.1,0.2,0.0], [0.0,0.0,0.0,0.0]],
                           [[0.5,0.3,0.2,0.3], [0.0,0.0,0.0,0.0], [0.0,0.0,0.0,0.0]], 
                           [[0.0,0.0,0.0,0.0], [0.0,0.0,0.0,0.0], [0.0,0.0,0.0,0.0]]]
    encoded_text_tensor = torch.Tensor(encoded_text_tensor)
    assert (batch_size * number_targets, text_seq_length, encoded_dim) == encoded_text_tensor.shape
    relative_target_positions = torch.Tensor([[[2,1,2], [1,2,0]],
                                              [[1,0,0], [0,0,0]]])
    assert (batch_size, number_targets, text_seq_length) == relative_target_positions.shape
    sequence_mask = torch.Tensor([[1,1,1], [1,1,0], [1,0,0], [0,0,0]])
    assert (batch_size * number_targets, text_seq_length) == sequence_mask.shape

    weighting = RelativeTargetPositionWeight(zero_target_word_weighting)
    result = weighting.forward(encoded_text_tensor, relative_target_positions, sequence_mask)
    weighted_encoding, position_weights = result

    assert (batch_size * number_targets, text_seq_length, encoded_dim) == weighted_encoding.shape
    assert (batch_size * number_targets, text_seq_length) == position_weights.shape

    true_position_weights = torch.Tensor([[2/3,1.0,2/3], [1.0,0.5,0.0], 
                                          [1.0,0.0,0.0], [0.0,0.0,0.0]])
    true_weighted_encoded = [[[1/3,0.2,(2/3 * 0.2),0.4], [0.2,0.3,0.4,0.7], [1/3,(2/3 * 0.4),0.4,(2/3 * 0.2)]], 
                             [[0.4,0.5,0.3,0.7], [0.15,0.05,0.1,0.0], [0.0,0.0,0.0,0.0]],
                             [[0.5,0.3,0.2,0.3], [0.0,0.0,0.0,0.0], [0.0,0.0,0.0,0.0]], 
                             [[0.0,0.0,0.0,0.0], [0.0,0.0,0.0,0.0], [0.0,0.0,0.0,0.0]]]
    true_weighted_encoded = torch.Tensor(true_weighted_encoded)
    if not zero_target_word_weighting:
        assert torch.allclose(true_position_weights, position_weights)
        assert torch.allclose(true_weighted_encoded, weighted_encoding)
    
    true_position_weights = torch.Tensor([[2/3,0.0,2/3], [0.0,0.5,0.0], 
                                          [0.0,0.0,0.0], [0.0,0.0,0.0]])
    true_weighted_encoded = [[[1/3,0.2,(2/3 * 0.2),0.4], [0.0,0.0,0.0,0.0], [1/3,(2/3 * 0.4),0.4,(2/3 * 0.2)]], 
                             [[0.0,0.0,0.0,0.0], [0.15,0.05,0.1,0.0], [0.0,0.0,0.0,0.0]],
                             [[0.0,0.0,0.0,0.0], [0.0,0.0,0.0,0.0], [0.0,0.0,0.0,0.0]], 
                             [[0.0,0.0,0.0,0.0], [0.0,0.0,0.0,0.0], [0.0,0.0,0.0,0.0]]]
    true_weighted_encoded = torch.Tensor(true_weighted_encoded)
    if zero_target_word_weighting:
        assert torch.allclose(true_position_weights, position_weights)
        assert torch.allclose(true_weighted_encoded, weighted_encoding)

    # Test that it raises the configuration error if the dimensions of the 
    # position and the encoded text do not match
    wrong_relative_target_positions = torch.zeros((batch_size, 3, text_seq_length))
    with pytest.raises(ConfigurationError):
        weighting.forward(encoded_text_tensor, wrong_relative_target_positions, 
                          sequence_mask)

    
