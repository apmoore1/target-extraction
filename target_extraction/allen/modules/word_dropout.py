import torch
from torch.nn.modules import Dropout2d

class WordDrouput(torch.nn.Module):
    '''
    Word Dropout will randomly drop whole words/timesteps.
    This is equivalent to `1D Spatial Dropout`_. 
    
    .. _1D Spatial 
        Dropout:https://keras.io/layers/core/#spatialdropout1d 
    '''

    def __init__(self, p: float) -> None:
        '''
        :param p: probability of a whole word/timestep to be zeroed/dropped.
        '''
        super().__init__()
        self._word_dropout = Dropout2d(p)

    def forward(self, embedded_text: torch.FloatTensor) -> torch.FloatTensor:
        '''
        :param embedded_text: A tensor of shape: 
                              [batch_size, timestep, embedding_dim] of which 
                              the dropout will drop entire timestep which is 
                              the equivalent to words.
        :returns: The given tensor but with timesteps/words dropped.
        '''
        embedded_text = embedded_text.unsqueeze(2)
        embedded_text = self._word_dropout(embedded_text)
        return embedded_text.squeeze(2)