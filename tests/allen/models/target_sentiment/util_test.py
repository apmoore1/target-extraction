from allennlp.modules.text_field_embedders.basic_text_field_embedder import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.embedding import Embedding
import torch
import pytest

from target_extraction.allen.models.target_sentiment import util

def test_concat_position_embeddings():
    # Test the normal case
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

    position_indexes = torch.Tensor([[[2,1,2], [1,2,0]],
                                     [[1,0,0], [0,0,0]]])
    position_indexes = position_indexes.type(torch.long)
    assert (batch_size, number_targets, text_seq_length) == position_indexes.shape
    position_indexes = {'position_tokens': {'tokens': position_indexes}}

    embedding = Embedding(num_embeddings=3, embedding_dim=5, trainable=False)
    target_position_embedding = BasicTextFieldEmbedder({'position_tokens': embedding})
    assert (batch_size, number_targets, text_seq_length, 5) == target_position_embedding(position_indexes).shape

    test_encoded_text_tensor = util.concat_position_embeddings(encoded_text_tensor, 
                                                               position_indexes, 
                                                               target_position_embedding)
    assert (batch_size * number_targets, text_seq_length, encoded_dim + 5) == test_encoded_text_tensor.shape

    # Test the case where it should return just the original encoded_text_tensor
    test_encoded_text_tensor = util.concat_position_embeddings(encoded_text_tensor, 
                                                               None, None)
    assert torch.all(torch.eq(test_encoded_text_tensor, encoded_text_tensor))

    # Test ValueError when the `target_position_embedding` is not None but 
    # position_indexes is None
    with pytest.raises(ValueError):
        util.concat_position_embeddings(encoded_text_tensor, None, 
                                        target_position_embedding)