from pathlib import Path

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import ensure_list
import pytest

from target_extraction.allen.dataset_readers import TargetConllDatasetReader

class TestTargetConllDatasetReader():
    @pytest.mark.parametrize("coding_scheme", ('BIO', 'BIOUL'))
    @pytest.mark.parametrize("lazy", (True, False))
    def test_read_from_file(self, lazy: bool, coding_scheme: str):
        reader = TargetConllDatasetReader(lazy=lazy, 
                                          coding_scheme=coding_scheme)
        
        data_dir = Path(__file__, '..', '..', '..', 'data', 'allen', 
                        'dataset_readers', 'target_conll').resolve()

        # Handle normal case and data that has extra predicted labels
        normal_fp = Path(data_dir, 'conll_format.conll')
        predicted_fp = Path(data_dir, 'predicted_conll_format.conll')
        for fp in [normal_fp, predicted_fp]:

            instance1 = {"tokens": ["The", "laptop", "case", "was", "great", "and", 
                                    "cover", "was", "rubbish"],
                        "tags": ["O", "B-0", "I-0", "I-0", "O", "O", "O", "O", "O"],
                        "bioul_tags": ["O", "B-0", "I-0", "L-0", "O", "O", "O", "O", "O"]}
            instance2 = {"tokens": ["The", "laptop", "case", "was", "great", "and", 
                                    "cover", "was", "rubbish"],
                        "tags": ["O", "B-0", "I-0", "O", "O", "O", "B-1", "O", "O"],
                        "bioul_tags": ["O", "B-0", "L-0", "O", "O", "O", "U-1", "O", "O"]}
            instance3 = {"tokens": ["The", "laptop", "case", "was", "great", "and", 
                                    "cover", "was", "rubbish"],
                        "tags": ["O", "O", "O", "O", "O", "O", "B-1", "O", "O"],
                        "bioul_tags": ["O", "O", "O", "O", "O", "O", "U-1", "O", "O"]}
            
            instances = ensure_list(reader.read(str(fp)))

            assert len(instances) == 3
            fields = instances[0].fields
            assert [t.text for t in fields["tokens"]] == instance1["tokens"]
            if coding_scheme == 'BIOUL':
                assert fields['tags'].labels == instance1["bioul_tags"]
            else:
                assert fields['tags'].labels == instance1["tags"]
            assert fields['metadata']['words'] == instance1["tokens"]
            
            fields = instances[1].fields
            assert [t.text for t in fields["tokens"]] == instance2["tokens"]
            if coding_scheme == 'BIOUL':
                assert fields['tags'].labels == instance2["bioul_tags"]
            else:
                assert fields['tags'].labels == instance2["tags"]
            assert fields['metadata']['words'] == instance2["tokens"]

            fields = instances[2].fields
            assert [t.text for t in fields["tokens"]] == instance3["tokens"]
            if coding_scheme == 'BIOUL':
                assert fields['tags'].labels == instance3["bioul_tags"]
            else:
                assert fields['tags'].labels == instance3["tags"]
            assert fields['metadata']['words'] == instance3["tokens"]

    @pytest.mark.parametrize("lazy", (True, False))
    def raise_config_error_on_coding_scheme(self, lazy: bool):
        with pytest.raises(ConfigurationError):
            TargetConllDatasetReader(lazy=lazy, coding_scheme='error')