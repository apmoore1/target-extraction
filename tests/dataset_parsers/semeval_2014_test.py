from pathlib import Path
from typing import Dict, List, Any

import pytest

from target_extraction.dataset_parsers import semeval_2014
from target_extraction.data_types_util import Span

class Semeval2014:

    DATA_PATH_DIR = Path(__file__, '..', '..', 'data', 'parsing', 'semeval_14')

    def _target_answer(self, conflict: bool) -> Dict[str, Dict[str, Any]]:
        answer_0 = {'text_id': '3121', 'text': "The staff were ok", 
                    'targets': ['staff'], 'spans': [Span(4, 9)],
                    'sentiments': ['neutral']}

    @pytest.mark.parametrize("conflict", (True, False))
    def test_read_from_file(self, conflict: bool):
        data_fp = Path(self.DATA_PATH_DIR, 'semeval_14_example.xml')
        target_text_collection = semeval_2014(data_fp, conflict)

        assert len(target_text_collection) == 5

        _ids = ['3121', '2777', '2534', '2634', '1793']
        assert list(target_text_collection.keys()) == _ids



    @pytest.mark.parametrize("conflict", (True, False))
    def test_unreadable_file(self, conflict: bool):
        # Test that it will raise a SyntaxError if a file is not of correct
        # format

        unreadable_fp = Path(self.DATA_PATH_DIR, 
                             'unpassable_semeval_14_example.xml')
        with pytest.raises(SyntaxError):
            semeval_2014(unreadable_fp, conflict)

    