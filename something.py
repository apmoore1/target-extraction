import tempfile
from pathlib import Path
from time import time

from target_extraction.dataset_parsers import wang_2017_election_twitter_train, wang_2017_election_twitter_test

with tempfile.TemporaryDirectory() as temp_dir:
    another = Path(temp_dir, 'first')
    t = time()
    a=wang_2017_election_twitter_train(another)
    a_t = a.number_targets()
    b=wang_2017_election_twitter_test(another)
    b_t = b.number_targets()
    print('done')

