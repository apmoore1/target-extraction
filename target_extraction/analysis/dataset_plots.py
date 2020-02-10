'''
This module contains plot functions that use the statistics produced from 
:py:mod:`target_extraction.analysis.dataset_statistics`
'''
from typing import Callable, List, Optional

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from target_extraction.data_types import TargetTextCollection
from target_extraction.analysis.dataset_statistics import tokens_per_target
from target_extraction.analysis.dataset_statistics import tokens_per_sentence

def target_length_plot(collections: List[TargetTextCollection],
                       target_key: str, 
                       tokeniser: Callable[[str], List[str]],
                       max_target_length: Optional[int] = None,
                       cumulative_percentage: bool = False,
                       ax: Optional[plt.Axes] = None) -> plt.Axes:
    '''
    :param collections: A list of collections to generate target length plots 
                        for.
    :param target_key: The key within each sample in the collection that contains 
                       the list of targets to be analysed. This can also be the 
                       predicted target key, which might be useful for error 
                       analysis.
    :param tokenizer: The tokenizer to use to split the target(s) into tokens. See 
                      for a module of comptabile tokenisers 
                      :py:mod:`target_extraction.tokenizers`
    :param max_target_length: The maximum target length to plot on the X-axis.
    :param cumulative_percentage: If the return should not be percentage of 
                                  the number of tokens in each target but rather 
                                  the cumulative percentage of targets with 
                                  that number of tokens.
    :param ax: Optional Axes to plot on too.
    :returns: A point plot where the X-axis represents the target length, Y-axis 
              percentage of samples with that target length, and the hue 
              represents the collection.
    '''
    dataset_names = []
    token_lengths = []
    length_percentage = []
    normalise = False if cumulative_percentage else True
    for collection in collections:
        name = collection.name
        token_length_percent = tokens_per_target(collection, target_key, 
                                                 tokeniser, normalise=normalise, 
                                                 cumulative_percentage=cumulative_percentage)
        for token_length, length_percent in token_length_percent.items():
            dataset_names.append(name)
            token_lengths.append(token_length)
            length_percentage.append(length_percent)
    
    y_axis_label = 'Percentage'
    if cumulative_percentage:
        y_axis_label = 'Cumulative %'
    else:
        length_percentage = [length_percent * 100 
                             for length_percent in length_percentage]
    target_length_df = pd.DataFrame({'Dataset': dataset_names, 
                                     'Target Length': token_lengths, 
                                     y_axis_label: length_percentage})
    if max_target_length is not None:
        target_length_df = target_length_df[target_length_df['Target Length'] <= max_target_length]
    return sns.pointplot(data=target_length_df, hue='Dataset', x='Target Length', 
                         y=y_axis_label, dodge=0.4, ax=ax)


def sentence_length_plot(collections: List[TargetTextCollection], 
                         tokeniser: Callable[[str], List[str]],
                         as_percentage: bool = True,
                         sentences_with_targets_only: bool = True,
                         ax: Optional[plt.Axes] = None) -> plt.Axes:
    '''
    :param collections: A list of collections to generate sentence length plots
                        for.
    :param tokenizer: The tokenizer to use to split the sentences into tokens. See 
                      for a module of comptabile tokenisers 
                      :py:mod:`target_extraction.tokenizers`
    :param as_percentage: The frequency of the sentence lengths should be 
                          normalised with respect to the number of sentences in 
                          the relevent dataset and then as a percentage.
    :param sentences_with_targets_only: Only use the sentences that have targets 
                                        within them.
    :param ax: Optional Axes to plot on too.
    :returns: A line plot where the X-axis represents that sentence length, 
              Y-axis the frequency of the sentence length, and the color 
              represents the collection.
    '''
    if sentences_with_targets_only:
        temp_collections = []
        for collection in collections:
            name = collection.name
            target_collection = collection.samples_with_targets()
            target_collection.name = name
            temp_collections.append(target_collection)
        collections = temp_collections
    
    dataset_names = []
    sentence_length = []
    length_frequencies = []

    for collection in collections:
        length_frequency = tokens_per_sentence(collection, tokeniser=tokeniser)
        if as_percentage:
            num_sentences = float(sum(length_frequency.values()))
            length_frequency = {length: 100 * (frequency / num_sentences) 
                                for length, frequency in length_frequency.items()}
        for length, frequency in length_frequency.items():
            sentence_length.append(length)
            length_frequencies.append(frequency)
            dataset_names.append(collection.name)
    
    y_axis_name = 'Frequency'
    if as_percentage:
        y_axis_name = 'Percentage'
    
    df = pd.DataFrame({'Dataset': dataset_names, 'Sentence Length': sentence_length, 
                        y_axis_name: length_frequencies})
    return sns.lineplot(data=df, x='Sentence Length', hue='Dataset', 
                        y=y_axis_name, ax=ax)