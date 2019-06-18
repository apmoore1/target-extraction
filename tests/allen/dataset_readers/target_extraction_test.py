from pathlib import Path

from allennlp.common.util import ensure_list
import pytest

from target_extraction.allen.dataset_readers import TargetExtractionDatasetReader

class TestTargetExtractionDatasetReader():
    @pytest.mark.parametrize("lazy", (True, False))
    @pytest.mark.parametrize("pos_tags", (True, False))
    def test_read_from_file(self, lazy: bool, pos_tags: bool):
        reader = TargetExtractionDatasetReader(lazy=lazy, pos_tags=pos_tags)
        
        data_dir = Path(__file__, '..', '..', 'data', 'allen', 
                        'dataset_readers', 'target_extraction').resolve()
        pos_tagged_fp = Path(data_dir, 'pos_sequence.json')
        
        instances = ensure_list(reader.read(str(pos_tagged_fp)))

        instance1 = {"tokens": ["The", "laptop", "case", "was", "great", "and", 
                                "cover", "was", "rubbish"],
                     "pos_tags": ["DET", "NOUN", "NOUN", "AUX", "ADJ", "CCONJ", 
                                  "NOUN", "AUX", "ADJ"],
                     "labels": ["O", "B", "I", "O", "O", "O", "O", "O", "O"]}
        instance2 = {"text": ["The", "So", "called", "desktop", "Runs", "to",
                              "badly"],
                     "targets": [["Runs"]],
                     "sentiments": [sentiment_mapper[-1]]}
        instance3 = {"text": ["errrr", "when", "I", "order", "I", "did", "go",
                              "full", "scale", "for", "the", "noting", "or",
                              "full", "trackpad", "I", "wanted", "something",
                              "for", "basics", "of", "being", "easy", "to",
                              "move", "It", "."],
                     "targets": [["noting"], ["move", "It"], ["trackpad"]],
                     "sentiments": [sentiment_mapper[0], sentiment_mapper[1], 
                                    sentiment_mapper[0]]}

        assert len(instances) == 3
        fields = instances[0].fields
        assert [t.text for t in fields["text"]] == instance1["text"]
        for index, target_field in enumerate(fields['targets']):
            assert [t.text for t in target_field] == instance1["targets"][index]
        assert fields['labels'].labels == instance1["sentiments"]
        
        fields = instances[1].fields
        assert [t.text for t in fields["text"]] == instance2["text"]
        for index, target_field in enumerate(fields['targets']):
            assert [t.text for t in target_field] == instance2["targets"][index]
        assert fields['labels'].labels == instance2["sentiments"]

        fields = instances[2].fields
        assert [t.text for t in fields["text"]] == instance3["text"]
        for index, target_field in enumerate(fields['targets']):
            assert [t.text for t in target_field] == instance3["targets"][index]
        assert fields['labels'].labels == instance3["sentiments"]