import pytest

from target_extraction.taggers_helper import stanford_downloader

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