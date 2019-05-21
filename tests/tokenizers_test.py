from typing import List

import pytest

from target_extraction.tokenizers import whitespace, spacy_tokenizer, stanford

class TestTokenizers:

    def _emoji_sentence(self) -> str:
        return "Hello how are you, with other's :)"
    def _no_sentence(self) -> str:
        return ''
    def _whitespace_sentence(self) -> str:
        return 'another day is today'
    def _comma_sentence(self) -> str:
        return 'today Is a great, day I think'

    def test_whitespace(self):
        whitespace_tokenizer = whitespace()

        emoji_tokens = whitespace_tokenizer(self._emoji_sentence())
        assert emoji_tokens == ['Hello', 'how', 'are', 'you,', 'with',
                                "other's", ':)']
        
        no_sentence_tokens = whitespace_tokenizer(self._no_sentence())
        assert no_sentence_tokens == []

        whitespace_tokens = whitespace_tokenizer(self._whitespace_sentence())
        assert whitespace_tokens == ['another', 'day', 'is', 'today']

        comma_tokens = whitespace_tokenizer(self._comma_sentence())
        assert comma_tokens == ['today', 'Is', 'a', 'great,', 'day', 'I', 
                                'think']
    
    @pytest.mark.parametrize("lang", ('en', 'de', 'nn'))
    def test_spacy_tokenizer(self, lang: str):
        if lang == 'nn':
            with pytest.raises(ValueError):
                spacy_tok = spacy_tokenizer(lang=lang)
        else:
            spacy_tok = spacy_tokenizer(lang=lang)
            
            emoji_tokens = spacy_tok(self._emoji_sentence())
            assert emoji_tokens == ['Hello', 'how', 'are', 'you', ',', 'with',
                                    "other", "'s", ':)']

            no_sentence_tokens = spacy_tok(self._no_sentence())
            assert no_sentence_tokens == []

            whitespace_tokens = spacy_tok(self._whitespace_sentence())
            assert whitespace_tokens == ['another', 'day', 'is', 'today']

            comma_tokens = spacy_tok(self._comma_sentence())
            assert comma_tokens == ['today', 'Is', 'a', 'great', ',', 'day', 'I',
                                    'think']

    @pytest.mark.parametrize("lang", ('en', 'de'))
    @pytest.mark.parametrize("treebank", (None, 'ewt', 'gum'))
    def test_stanford_tokenizer(self, lang: str, treebank: str):
        '''
        This does not really currently test if the treebanks perform as they 
        should i.e. we do not currently test that the English EWT treebank
        tokeniser is any different to the Enlgish GUM tokeniser.
        '''
        if treebank is not None and lang == 'de':
            pass
        else:
            tokenizer = stanford(lang=lang, treebank=treebank)

            emoji_tokens = tokenizer(self._emoji_sentence())
            emoji_ans = ['Hello', 'how', 'are', 'you', ',', 'with',
                         "other", "'s", ':)']
            if lang == 'de':
                emoji_ans = ['Hello', 'how', 'are', 'you', ',', 'with',
                             "other", "'s", ':', ')']
            assert emoji_tokens == emoji_ans

            no_sentence_tokens = tokenizer(self._no_sentence())
            assert no_sentence_tokens == []

            whitespace_tokens = tokenizer(self._whitespace_sentence())
            assert whitespace_tokens == ['another', 'day', 'is', 'today']

            comma_tokens = tokenizer(self._comma_sentence())
            assert comma_tokens == ['today', 'Is', 'a', 'great', ',', 'day', 'I',
                                    'think']
