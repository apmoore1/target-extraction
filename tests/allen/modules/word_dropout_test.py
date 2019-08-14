import torch
import numpy
from flaky import flaky

from target_extraction.allen.modules.word_dropout import WordDrouput

@flaky
def test_word_dropout():
    def number_zeros(tensor_to_test: torch.Tensor, tensor_dim: int, 
                     batch_size: int, number_words: int) -> int:
        zero_array = numpy.zeros(tensor_dim)
        zero_count = 0
        for i in range(batch_size):
            for j in range(number_words):
                if numpy.array_equal(tensor_to_test.data[i, j, :].numpy(), zero_array):
                    zero_count += 1
        return zero_count

    tensor = torch.rand([5, 7, 10])
    assert 0 == number_zeros(tensor, 10, 5, 7)
    # Around half of the words in the 5 * 7 words should have zero vectors
    word_dropout = WordDrouput(0.5)
    half_dropout = word_dropout(tensor)
    min_dropout = int((5 * 7) / 2) - 4
    max_dropout = int((5 * 7) / 2) + 4
    zero_count = number_zeros(half_dropout, 10, 5, 7)
    assert zero_count <= max_dropout and zero_count >= min_dropout

    # Zero Dropout
    word_dropout = WordDrouput(0.0)
    zero_dropout = word_dropout(tensor)
    assert 0 == number_zeros(zero_dropout, 10, 5, 7)

    # Full Dropout
    word_dropout = WordDrouput(1.0)
    full_dropout = word_dropout(tensor)
    assert 35 == number_zeros(full_dropout, 10, 5, 7)
