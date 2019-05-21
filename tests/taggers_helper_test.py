import pytest

from target_extraction.taggers_helper import stanford_downloader, spacy_downloader

class TestTaggersHelper:

    @pytest.mark.parametrize("lang", ('en', 'de', 'does_not_exist'))
    @pytest.mark.parametrize("treebank", (None, 'ewt', 'gum'))
    def test_stanford_downloader(self, lang: str, treebank: str):
        # Test that it should raise an error when the treebank does not 
        # exist for a language.
        if lang == 'does_not_exist':
            with pytest.raises(ValueError):
                stanford_downloader(lang=lang, treebank=treebank)
        elif treebank is not None and lang == 'de':
            with pytest.raises(ValueError):
                stanford_downloader(lang=lang, treebank=treebank)
        else:
            resolved_treebank = stanford_downloader(lang=lang, 
                                                    treebank=treebank)
            if treebank is None:
                if lang == 'de':
                    treebank = 'gsd'
                elif lang == 'en':
                    treebank = 'ewt'
                assert f'{lang}_{treebank}' == resolved_treebank
            else:
                assert f'{lang}_{treebank}' == resolved_treebank
    
    @pytest.mark.parametrize("ner", (True, False))
    @pytest.mark.parametrize("parse", (True, False))
    @pytest.mark.parametrize("pos_tags", (True, False))
    @pytest.mark.parametrize("model_name", ('en_core_web_sm', 'it_core_news_sm', 
                                            'ar_core_web_sm'))
    def test_spacy_downloader(self, model_name: str, pos_tags: bool, 
                              parse: bool, ner: bool):
        if model_name == 'ar_core_web_sm':
            with pytest.raises(ValueError):
                spacy_downloader(model_name, pos_tags, parse, ner)
        else:
            spacy_downloader(model_name, pos_tags, parse, ner)

        
        