from typing import List, Callable, Tuple

import pytest

from target_extraction.tokenizers import whitespace, spacy_tokenizer, stanford, is_character_preserving, token_index_alignment

class TestTokenizers:

    def _emoji_sentence(self) -> str:
        return "Hello how are you, with other's :)"
    def _no_sentence(self) -> str:
        return ''
    def _whitespace_sentence(self) -> str:
        return 'another day is today'
    def _serveral_whitespace(self) -> str:
        return '   another    day is today   '
    def _comma_sentence(self) -> str:
        return 'today Is a great, day I think'
    def _difficult_tokenizer_sentence(self) -> str:
        return "But guess what?  (you have to buy an external dvd drive."

    def not_char_preserving_tokenizer(self, text: str) -> List[str]:
        tokens = text.split()
        alt_tokens = []
        for token in tokens:
            if token == "other's":
                alt_tokens.append('other')
            else:
                alt_tokens.append(token)
        return alt_tokens

    # This is bad coding pracice but the str.split with False value in the 
    # actual method we replace str.split with not_char_preserving_tokenizer
    @pytest.mark.parametrize("tokenizer_pass", ((whitespace(), True), (spacy_tokenizer(), True), 
                                                (stanford(), True), (str.split, False)))
    def test_is_character_preserving(self, 
                                     tokenizer_pass: Tuple[Callable[[str], List[str]], bool]):
        tokenizer, pass_or_not = tokenizer_pass
        sentence = self._emoji_sentence()
        tokens = tokenizer(sentence)
        if not pass_or_not:
            tokens = self.not_char_preserving_tokenizer(sentence)
        assert is_character_preserving(sentence, tokens) == pass_or_not

        if pass_or_not:
            sentence = self._difficult_tokenizer_sentence()
            tokens = tokenizer(sentence)
            assert is_character_preserving(sentence, tokens) == True

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

        more_whitespace_tokens = whitespace_tokenizer(self._serveral_whitespace())
        assert more_whitespace_tokens == ['another', 'day', 'is', 'today']
    
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

            more_whitespace_tokens = spacy_tok(self._serveral_whitespace())
            assert more_whitespace_tokens == ['another', 'day', 'is', 'today']

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

            more_whitespace_tokens = tokenizer(self._serveral_whitespace())
            assert more_whitespace_tokens == ['another', 'day', 'is', 'today']

            comma_tokens = tokenizer(self._comma_sentence())
            assert comma_tokens == ['today', 'Is', 'a', 'great', ',', 'day', 'I',
                                    'think']
    
    @pytest.mark.parametrize("tokenizer", (whitespace(), spacy_tokenizer(),
                                           stanford()))
    def test_token_index_alignment(self, tokenizer: Callable[[str], List[str]]):
        # Test a sentence where whitespace will be the only factor
        text = self._whitespace_sentence()
        token_indexs = [(0, 7), (8, 11), (12, 14), (15, 20)]
        assert token_indexs == token_index_alignment(text, tokenizer(text))
        
        # Test a sentence where we have a comma which will cause extra 
        # whitespace on the tokenization side
        text = self._comma_sentence()
        token_indexs = [(0, 5), (6, 8), (9, 10), (11, 16), (16,17), (18,21), 
                        (22,23), (24,29)]
        if tokenizer != whitespace():
            assert token_indexs == token_index_alignment(text, tokenizer(text))
        else:
            token_indexs = [(0, 5), (6, 8), (9, 10), (11, 17), (18, 21),
                            (22, 23), (24, 29)]
            assert token_indexs == token_index_alignment(text, tokenizer(text))

        # Test a sentence where we have multiple spaces in the text at the 
        # start, end and in between tokens
        text = '  I had,   great day  '
        token_indexs = [(2, 3), (4, 7), (7, 8), (11, 16), (17, 20)]
        if tokenizer != whitespace():
            assert token_indexs == token_index_alignment(text, tokenizer(text))
        else:
            token_indexs = [(2, 3), (4, 8), (11, 16), (17, 20)]
            assert token_indexs == token_index_alignment(text, tokenizer(text))

        # Test a sentence that has multiple space commas hyphens etc.
        text = "  I had,  isn't  great day  doesn't'"
        token_indexs = [(2, 3), (4, 7), (7, 8), (10, 12), (12, 15), (17, 22),
                        (23, 26), (28, 32), (32, 35), (35, 36)]
        if tokenizer != whitespace():
            assert token_indexs == token_index_alignment(text, tokenizer(text))
        else:
            token_indexs = [(2, 3), (4, 8),(10, 15), (17, 22),(23, 26), 
                            (28, 36)]
            assert token_indexs == token_index_alignment(text, tokenizer(text))
