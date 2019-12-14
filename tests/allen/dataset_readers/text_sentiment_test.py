import pytest
from pathlib import Path

from allennlp.common.util import ensure_list
from allennlp.common.testing import AllenNlpTestCase

from target_extraction.allen.dataset_readers import TextSentimentReader

DATA_DIR = Path(__file__, '..', '..', '..', 'data', 'allen', 
                'dataset_readers', 'text_sentiment').resolve()
# All of these tests have been taken from the Allennlp github
# https://github.com/allenai/allennlp/blob/master/allennlp/tests/data/dataset_readers/text_classification_json_test.py
# here the only add on is the `label_name` argument in the constructor
class TestTextClassificationJsonReader:
    @pytest.mark.parametrize("label_name", ('label', 'text_sentiment'))
    @pytest.mark.parametrize("lazy", (True, False))
    def test_set_skip_indexing_true(self, lazy: bool, label_name: str):
        reader = TextSentimentReader(lazy=lazy, skip_label_indexing=True, 
                                     label_name=label_name)
        ag_path = (
            AllenNlpTestCase.FIXTURES_ROOT
            / "data"
            / "text_classification_json"
            / "integer_labels.jsonl"
        )
        if label_name == 'text_sentiment':
            ag_path = Path(DATA_DIR,"integer_labels.jsonl" ).resolve()
        instances = reader.read(ag_path)
        instances = ensure_list(instances)

        instance1 = {"tokens": ["This", "text", "has", "label", "0"], "label": 0}
        instance2 = {"tokens": ["This", "text", "has", "label", "1"], "label": 1}

        assert len(instances) == 2
        fields = instances[0].fields
        assert [t.text for t in fields["tokens"].tokens] == instance1["tokens"]
        assert fields["label"].label == instance1["label"]
        fields = instances[1].fields
        assert [t.text for t in fields["tokens"].tokens] == instance2["tokens"]
        assert fields["label"].label == instance2["label"]

        ag_path = (
                AllenNlpTestCase.FIXTURES_ROOT
                / "data"
                / "text_classification_json"
                / "imdb_corpus.jsonl"
            )
        if label_name == 'text_sentiment':
            ag_path = Path(DATA_DIR, "imdb_corpus.jsonl" ).resolve()
        with pytest.raises(ValueError) as exec_info:
            ensure_list(reader.read(ag_path))
        assert str(exec_info.value) == "Labels must be integers if skip_label_indexing is True."

    @pytest.mark.parametrize("label_name", ('label', 'text_sentiment'))
    @pytest.mark.parametrize("lazy", (True, False))
    def test_read_from_file_ag_news_corpus(self, lazy: bool, label_name: str):
        reader = TextSentimentReader(lazy=lazy, label_name=label_name)
        ag_path = (
            AllenNlpTestCase.FIXTURES_ROOT
            / "data"
            / "text_classification_json"
            / "ag_news_corpus.jsonl"
        )
        if label_name == 'text_sentiment':
            ag_path = Path(DATA_DIR, 'ag_news_corpus.jsonl')
        instances = reader.read(ag_path)
        instances = ensure_list(instances)

        instance1 = {
            "tokens": [
                "Memphis",
                "Rout",
                "Still",
                "Stings",
                "for",
                "No",
                ".",
                "14",
                "Louisville",
                ";",
                "Coach",
                "Petrino",
                "Vows",
                "to",
                "Have",
                "Team",
                "Better",
                "Prepared",
                ".",
                "NASHVILLE",
                ",",
                "Tenn.",
                "Nov",
                "3",
                ",",
                "2004",
                "-",
                "Louisville",
                "#",
                "39;s",
                "30-point",
                "loss",
                "at",
                "home",
                "to",
                "Memphis",
                "last",
                "season",
                "is",
                "still",
                "a",
                "painful",
                "memory",
                "for",
                "the",
                "Cardinals",
                ".",
            ],
            "label": "2",
        }
        instance2 = {
            "tokens": [
                "AP",
                "-",
                "Eli",
                "Manning",
                "has",
                "replaced",
                "Kurt",
                "Warner",
                "as",
                "the",
                "New",
                "York",
                "Giants",
                "'",
                "starting",
                "quarterback",
                ".",
            ],
            "label": "2",
        }
        instance3 = {
            "tokens": [
                "A",
                "conference",
                "dedicated",
                "to",
                "online",
                "journalism",
                "explores",
                "the",
                "effect",
                "blogs",
                "have",
                "on",
                "news",
                "reporting",
                ".",
                "Some",
                "say",
                "they",
                "draw",
                "attention",
                "to",
                "under",
                "-",
                "reported",
                "stories",
                ".",
                "Others",
                "struggle",
                "to",
                "establish",
                "the",
                "credibility",
                "enjoyed",
                "by",
                "professionals",
                ".",
            ],
            "label": "4",
        }

        assert len(instances) == 3
        fields = instances[0].fields
        assert [t.text for t in fields["tokens"].tokens] == instance1["tokens"]
        assert fields["label"].label == instance1["label"]
        fields = instances[1].fields
        assert [t.text for t in fields["tokens"].tokens] == instance2["tokens"]
        assert fields["label"].label == instance2["label"]
        fields = instances[2].fields
        assert [t.text for t in fields["tokens"].tokens] == instance3["tokens"]
        assert fields["label"].label == instance3["label"]

    @pytest.mark.parametrize("label_name", ('label', 'text_sentiment'))
    @pytest.mark.parametrize("lazy", (True, False))
    def test_read_from_file_ag_news_corpus_and_truncates_properly(self, lazy: bool, 
                                                                  label_name: str):
        reader = TextSentimentReader(lazy=lazy, max_sequence_length=5, 
                                     label_name=label_name)
        ag_path = (
            AllenNlpTestCase.FIXTURES_ROOT
            / "data"
            / "text_classification_json"
            / "ag_news_corpus.jsonl"
        )
        if label_name == 'text_sentiment':
            ag_path = Path(DATA_DIR, 'ag_news_corpus.jsonl')
        instances = reader.read(ag_path)
        instances = ensure_list(instances)

        instance1 = {"tokens": ["Memphis", "Rout", "Still", "Stings", "for"], "label": "2"}
        instance2 = {"tokens": ["AP", "-", "Eli", "Manning", "has"], "label": "2"}
        instance3 = {"tokens": ["A", "conference", "dedicated", "to", "online"], "label": "4"}

        assert len(instances) == 3
        fields = instances[0].fields
        assert [t.text for t in fields["tokens"].tokens] == instance1["tokens"]
        assert fields["label"].label == instance1["label"]
        fields = instances[1].fields
        assert [t.text for t in fields["tokens"].tokens] == instance2["tokens"]
        assert fields["label"].label == instance2["label"]
        fields = instances[2].fields
        assert [t.text for t in fields["tokens"].tokens] == instance3["tokens"]
        assert fields["label"].label == instance3["label"]

    @pytest.mark.parametrize("label_name", ('label', 'text_sentiment'))
    @pytest.mark.parametrize("lazy", (True, False))
    def test_read_from_file_ag_news_corpus_and_segments_sentences_properly(self, lazy: bool, label_name: str):
        reader = TextSentimentReader(lazy=lazy, segment_sentences=True, 
                                     label_name=label_name)
        ag_path = (
            AllenNlpTestCase.FIXTURES_ROOT
            / "data"
            / "text_classification_json"
            / "ag_news_corpus.jsonl"
        )
        if label_name == 'text_sentiment':
            ag_path = Path(DATA_DIR, 'ag_news_corpus.jsonl')
        instances = reader.read(ag_path)
        instances = ensure_list(instances)

        instance1 = {
            "tokens": [
                ["Memphis", "Rout", "Still", "Stings", "for", "No", ".", "14", "Louisville", ";"],
                ["Coach", "Petrino", "Vows", "to", "Have"],
                ["Team", "Better", "Prepared", "."],
                ["NASHVILLE", ","],
                ["Tenn.", "Nov", "3", ",", "2004", "-", "Louisville", "#", "39;s",
                 "30-point", "loss", "at", "home", 'to', 'Memphis', 'last', 
                 'season', 'is', 'still', 'a', 'painful', 'memory', 'for', 'the', 
                 'Cardinals', '.'],
            ],
            "label": "2",
        }
        instance2 = {
            "tokens": [['AP', '-', 'Eli', 'Manning', 'has', 'replaced', 'Kurt', 
                        'Warner', 'as', 'the', 'New', 'York', 'Giants', "'", 
                        'starting', 'quarterback', '.']],
            "label": "2",
        }
        instance3 = {
            "tokens": [
                [
                    "A",
                    "conference",
                    "dedicated",
                    "to",
                    "online",
                    "journalism",
                    "explores",
                    "the",
                    "effect",
                    "blogs",
                    "have",
                    "on",
                    "news",
                    "reporting",
                    ".",
                ],
                [
                    "Some",
                    "say",
                    "they",
                    "draw",
                    "attention",
                    "to",
                    "under",
                    "-",
                    "reported",
                    "stories",
                    ".",
                ],
                [
                    "Others",
                    "struggle",
                    "to",
                    "establish",
                    "the",
                    "credibility",
                    "enjoyed",
                    "by",
                    "professionals",
                    ".",
                ],
            ],
            "label": "4",
        }

        assert len(instances) == 3
        fields = instances[0].fields
        text = [[token.text for token in sentence.tokens] for sentence in fields["tokens"]]
        assert text == instance1["tokens"]
        assert fields["label"].label == instance1["label"]
        fields = instances[1].fields
        text = [[token.text for token in sentence.tokens] for sentence in fields["tokens"]]
        assert text == instance2["tokens"]
        assert fields["label"].label == instance2["label"]
        fields = instances[2].fields
        text = [[token.text for token in sentence.tokens] for sentence in fields["tokens"]]
        assert text == instance3["tokens"]
        assert fields["label"].label == instance3["label"]