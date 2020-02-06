'''
All of the tests in this module will only test that no errors are produced 
and the graphs do get created.
'''
import copy
from pathlib import Path

import matplotlib.pyplot as plt

from target_extraction.data_types import TargetTextCollection, TargetText
from target_extraction.tokenizers import whitespace
from target_extraction.analysis.dataset_plots import target_length_plot

DATA_DIR = Path(__file__, '..', '..', 'data', 'analysis', 'sentiment_error_analysis').resolve()
TRAIN_COLLECTION =  TargetTextCollection.load_json(Path(DATA_DIR, 'train_with_blank.json'))
TRAIN_COLLECTION.name = 'train'

def test_target_length_plot():
    # standard/normal case
    ax = target_length_plot([TRAIN_COLLECTION], 'targets', whitespace())
    del ax
    # cumulative percentage True
    ax = target_length_plot([TRAIN_COLLECTION], 'targets', whitespace(), 
                            cumulative_percentage=True)
    del ax
    # Max target length
    ax = target_length_plot([TRAIN_COLLECTION], 'targets', whitespace(), 
                            cumulative_percentage=True, max_target_length=1)
    del ax
    # Can take consume an axes
    fig, alt_ax = plt.subplots(1,1)
    ax = target_length_plot([TRAIN_COLLECTION], 'targets', whitespace(), 
                            cumulative_percentage=True, max_target_length=1,
                            ax=alt_ax)
    assert alt_ax == ax
    del ax
    plt.close(fig)

    # Can take more than one collection
    alt_collection = copy.deepcopy(list(TRAIN_COLLECTION.dict_iterator()))
    alt_collection = [TargetText(**v) for v in alt_collection]
    alt_collection = TargetTextCollection(alt_collection)
    alt_collection.name = 'Another'
    ax = target_length_plot([TRAIN_COLLECTION, alt_collection], 
                            'targets', whitespace(), 
                            cumulative_percentage=True, max_target_length=1)
    assert alt_ax != ax
    del alt_ax
    del ax