from pathlib import Path

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import ensure_list
import pytest

from target_extraction.allen.dataset_readers import TargetExtractionDatasetReader

class TestTargetExtractionDatasetReader():
    @pytest.mark.parametrize("lazy", (True, False))
    @pytest.mark.parametrize("pos_tags", (True, False))
    def test_read_from_file(self, lazy: bool, pos_tags: bool):
        reader = TargetExtractionDatasetReader(lazy=lazy, pos_tags=pos_tags)
        
        data_dir = Path(__file__, '..', '..', '..', 'data', 'allen', 
                        'dataset_readers', 'target_extraction').resolve()

        instance1 = {"tokens": ["The", "laptop", "case", "was", "great", "and", 
                                "cover", "was", "rubbish"],
                     "pos_tags": ["DET", "NOUN", "NOUN", "AUX", "ADJ", "CCONJ", 
                                  "NOUN", "AUX", "ADJ"],
                     "tags": ["O", "B", "I", "O", "O", "O", "O", "O", "O"],
                     "text": "The laptop case was great and cover was rubbish"}
        instance2 = {"tokens": ["Another", "day", "at", "the", "office"],
                     "pos_tags": ["DET", "NOUN", "ADP", "DET", "NOUN"],
                     "tags": ["O", "O", "O", "O", "O"],
                     "text": "Another day at the office"}
        instance3 = {"tokens": ["The", "laptop", "case", "was", "great", "and", 
                                "cover", "was", "rubbish"],
                     "pos_tags": ["DET", "NOUN", "NOUN", "AUX", "ADJ", "CCONJ", 
                                  "NOUN", "AUX", "ADJ"],
                     "tags": ["O", "B", "I", "O", "O", "O", "B", "O", "O"],
                     "text": "The laptop case was great and cover was rubbish"}

        # POS tagged data 
        pos_tagged_fp = Path(data_dir, 'pos_sequence.json')
        instances = ensure_list(reader.read(str(pos_tagged_fp)))

        assert len(instances) == 3
        fields = instances[0].fields
        assert [t.text for t in fields["tokens"]] == instance1["tokens"]
        if pos_tags:
            assert fields['pos_tags'].labels == instance1["pos_tags"]
        else:
            assert 'pos_tags' not in fields
        assert fields['tags'].labels == instance1["tags"]
        assert fields['metadata']['words'] == instance1["tokens"]
        assert fields['metadata']['text'] == instance1["text"]
        
        fields = instances[1].fields
        assert [t.text for t in fields["tokens"]] == instance2["tokens"]
        if pos_tags:
            assert fields['pos_tags'].labels == instance2["pos_tags"]
        else:
            assert 'pos_tags' not in fields
        assert fields['tags'].labels == instance2["tags"]
        assert fields['metadata']['words'] == instance2["tokens"]
        assert fields['metadata']['text'] == instance2["text"]

        fields = instances[2].fields
        assert [t.text for t in fields["tokens"]] == instance3["tokens"]
        if pos_tags:
            assert fields['pos_tags'].labels == instance3["pos_tags"]
        else:
            assert 'pos_tags' not in fields
        assert fields['tags'].labels == instance3["tags"]
        assert fields['metadata']['words'] == instance3["tokens"]
        assert fields['metadata']['text'] == instance3["text"]

        # Non-POS tagged version
        non_pos_tagged_fp = Path(data_dir, 'non_pos_sequence.json')

        if not pos_tags:
            instances = ensure_list(reader.read(str(non_pos_tagged_fp)))

            assert len(instances) == 3
            fields = instances[0].fields
            assert [t.text for t in fields["tokens"]] == instance1["tokens"]
            assert 'pos_tags' not in fields
            assert fields['tags'].labels == instance1["tags"]
            assert fields['metadata']['words'] == instance1["tokens"]
            assert fields['metadata']['text'] == instance1["text"]
            
            fields = instances[1].fields
            assert [t.text for t in fields["tokens"]] == instance2["tokens"]
            assert 'pos_tags' not in fields
            assert fields['tags'].labels == instance2["tags"]
            assert fields['metadata']['words'] == instance2["tokens"]
            assert fields['metadata']['text'] == instance2["text"]

            fields = instances[2].fields
            assert [t.text for t in fields["tokens"]] == instance3["tokens"]
            assert 'pos_tags' not in fields
            assert fields['tags'].labels == instance3["tags"]
            assert fields['metadata']['words'] == instance3["tokens"]
            assert fields['metadata']['text'] == instance3["text"]
        else:
            with pytest.raises(ConfigurationError):
                ensure_list(reader.read(str(non_pos_tagged_fp)))