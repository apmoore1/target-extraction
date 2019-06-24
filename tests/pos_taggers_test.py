from typing import List, Tuple

import pytest

from target_extraction.pos_taggers import stanford, spacy_tagger

class TestPOSTaggers:

    def _en_emoji_sentence(self) -> Tuple[List[str], str]:
        return (['Hello', 'how', 'are', 'you', ',', 'with', 'other', "'s", ':)'],
                "  Hello how are  you, with other's :)")
    def _de_emoji_sentence(self) -> Tuple[List[str], str]:
        return (['Hallo', 'wie', 'geht', 'es', 'dir', 'bei', 'anderen', ':', ')'],
                "  Hallo wie geht  es dir bei anderen :)")
    def _de_spacy_emoji_sentence(self) -> Tuple[List[str], str]:
        return(['Hallo', 'wie', 'geht', 'es', 'dir', 'bei', 'anderen', ':)'],
               "  Hallo wie geht  es dir bei anderen :)")
    def _no_sentence(self) -> str:
        return ""

    def _spacy_ans(self, fine: bool, lang: str) -> List[str]:
        if lang == 'en':
            if fine:
                return ['UH', 'WRB', 'VBP', 'PRP', ',', 'IN', 'JJ', 'POS', '.']
            else:
                return ['INTJ', 'ADV', 'VERB', 'PRON', 'PUNCT', 'ADP', 'ADJ', 
                        'PART', 'PUNCT']
        elif lang == 'de':
            if fine:
                return ['NE', 'PWAV', 'VVFIN', 'PPER', 'PPER', 'APPR', 'ADJA', 
                        'NN']
            else:
                return ['PROPN', 'ADV', 'VERB', 'PRON', 'PRON', 'ADP', 'ADJ', 
                        'NOUN']
        else:
            raise ValueError(f'Do not recognise this language {lang}')

    def _stanford_en_ans(self, fine: bool, treebank: str) -> List[str]:
        if treebank is None:
            treebank = 'ewt'
        if fine and treebank == 'gum':
            return ['UH', 'WRB', 'VBP', 'PRP', ',', 'IN', 'JJ', 'POS', 'SYM']
        elif fine and treebank == 'ewt':
            return ['UH', 'WRB', 'VBP', 'PRP', ',', 'IN', 'JJ', 'POS', 'NFP']
        elif not fine and treebank == 'gum':
            return ['INTJ', 'SCONJ', 'AUX', 'PRON', 'PUNCT', 'ADP', 'ADJ', 
                    'PART', 'PUNCT']
        elif not fine and treebank == 'ewt':
            return ['INTJ', 'ADV', 'AUX', 'PRON', 'PUNCT', 'ADP', 'ADJ', 
                    'PART', 'SYM']
        else:
            raise ValueError(f'Do not recognise these arguments {fine} '
                             f'{treebank}')
    
    def _stanford_de_ans(self, fine: bool, treebank: str) -> List[str]:
        if treebank is not None:
            raise ValueError('Treebank should always be None for the '
                             f'German language. But it is {treebank}')
        if fine:
            return ['ADV', 'PWAV', 'VVFIN', 'PPER', 'PPER', 'APPR', 'PIS', '$.', 
                    '$(']
        else:
            return ['ADV', 'ADV', 'VERB', 'PRON', 'PRON', 'ADP', 'PRON', 
                    'PUNCT', 'PUNCT']
            

    @pytest.mark.parametrize("fine", (True, False))
    @pytest.mark.parametrize("lang", ('en', 'de'))
    @pytest.mark.parametrize("treebank", (None, 'ewt', 'gum'))
    def test_stanford(self, fine: bool, lang: str, treebank: str):
        if treebank is not None and lang == 'de':
            pass
        elif lang == 'en':
            # Tests a sentence on two treebanks and across UPOS and XPOS tags
            pos_tagger = stanford(fine=fine, lang=lang, treebank=treebank)
            emoji_tok_ans, text = self._en_emoji_sentence()
            emoji_toks, emoji_pos = pos_tagger(text)
            emoji_ans = self._stanford_en_ans(fine=fine, treebank=treebank)
            assert emoji_ans == emoji_pos
            assert emoji_tok_ans == emoji_toks
            # Ensures that it can handle no text input
            no_toks, no_tags = pos_tagger(self._no_sentence())
            assert [] == no_tags
            assert [] == no_toks
        elif lang == 'de':
            # Tests a sentence on default treebank and across UPOS and XPOS tags
            pos_tagger = stanford(fine=fine, lang=lang, treebank=treebank)
            emoji_tok_ans, text = self._de_emoji_sentence()
            emoji_toks, emoji_pos = pos_tagger(text)
            emoji_ans = self._stanford_de_ans(fine=fine, treebank=treebank)
            assert emoji_ans == emoji_pos
            assert emoji_tok_ans == emoji_toks
            # Ensures that it can handle no text input
            no_toks, no_tags = pos_tagger(self._no_sentence())
            assert [] == no_tags
            assert [] == no_toks
    
    @pytest.mark.parametrize("fine", (True, False))
    @pytest.mark.parametrize("lang", ('en', 'de'))
    def test_spacy(self, fine: bool, lang: str):
        if lang == 'en':
            model_name = 'en_core_web_sm'
            emoji_tok_ans, emoji_sentence = self._en_emoji_sentence()
        elif lang == 'de':
            model_name = 'de_core_news_sm'
            emoji_tok_ans, emoji_sentence = self._de_spacy_emoji_sentence()
        # Tests a sentence across UPOS and XPOS tags
        pos_tagger = spacy_tagger(fine=fine, spacy_model_name=model_name)
        emoji_toks, emoji_pos = pos_tagger(emoji_sentence)
        emoji_ans = self._spacy_ans(fine=fine, lang=lang)
        assert emoji_ans == emoji_pos
        assert emoji_tok_ans == emoji_toks
        # Ensures that it can handle no text input
        no_toks, no_tags = pos_tagger(self._no_sentence())
        assert [] == no_tags
        assert [] == no_toks