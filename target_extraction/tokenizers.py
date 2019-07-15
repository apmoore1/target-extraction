'''
This modules contains a set of functions that return tokenization functions 
which can be defined by the following typing: Callable[[str], List[str]]. 
All of the functions take exactly no positional arguments but can take 
keyword arguments.
'''
import copy
from typing import List, Callable, Optional, Tuple
from pathlib import Path
import pkgutil

import spacy
import stanfordnlp
from stanfordnlp.utils import resources
import twokenize

from target_extraction.taggers_helper import stanford_downloader
from target_extraction.data_types_util import Span


def is_character_preserving(original_text: str, text_tokens: List[str]
                            ) -> bool:
    '''
    :param original_text: Text that has been tokenized
    :param text_tokens: List of tokens after the text has been tokenized
    :returns: True if the tokenized text when all characters are joined 
                together is equal to the original text with all it's 
                characters joined together.
    '''
    text_tokens_copy = copy.deepcopy(text_tokens)
    # Required as some of the tokenization tokens contain whitespace at the 
    # end of them I think this due to Stanford method being a Neural Network
    text_tokens_copy = [token.strip(' ') for token in text_tokens_copy]
    tokens_text = ''.join(text_tokens_copy)
    original_text = ''.join(original_text.split())
    if tokens_text == original_text:
        return True
    else:
        return False

def spacy_tokenizer(lang: str = 'en') -> Callable[[str], List[str]]:
    '''
    Given optionally the language (default English) it will return the 
    Spacy rule based tokeniser for that language but the function will now 
    return a List of String rather than Spacy tokens.

    If the whitespace between two words is more than one token then the Spacy 
    tokenizer treat it as in affect a special space token, we remove these 
    special space tokens.

    :param lang: Language of the rule based Spacy tokeniser to use.
    :returns: A callable that takes a String and returns the tokens for that 
              String.
    '''
    spacy_lang_modules = pkgutil.iter_modules(spacy.lang.__path__)
    spacy_lang_codes = [lang_code for _, lang_code, _ in spacy_lang_modules 
                        if len(lang_code) == 2]
    if lang not in spacy_lang_codes:
        raise ValueError('Spacy does not support the following language '
                         f'{lang}. These languages are supported '
                         f'{spacy_lang_codes}')
    sapcy_tokenizer_func = spacy.blank(lang)
    def _spacy_token_to_text(text: str) -> Callable[[str], List[str]]:
        return [spacy_token.text for spacy_token in sapcy_tokenizer_func(text) 
                if not spacy_token.is_space]
    return _spacy_token_to_text

def whitespace() -> Callable[[str], List[str]]:
    '''
    Standard whitespace tokeniser

    :returns: A callable that takes a String and returns the tokens for that 
              String.
    '''
    return str.split

def ark_twokenize() -> Callable[[str], List[str]]:
    '''
    A Twitter tokeniser from
    `CMU Ark <https://github.com/brendano/ark-tweet-nlp>`_

    :returns: A callable that takes a String and returns the tokens for 
              that String.
    '''
    return twokenize.tokenizeRawTweetText

def stanford(lang: str = 'en', treebank: Optional[str] = None, 
             download: bool = False) -> Callable[[str], List[str]]:
    '''
    Stanford neural network tokeniser that uses a BiLSTM and CNN at the 
    character and token level.

    ASSUMPTIONS: The returned callable tokeniser will assume that all text 
    that is given to it, is one sentence, as this method performs sentence 
    splitting but we assume each text is one sentence and we ignore the 
    sentence splitting.

    For Vietnamese instead of characters they used syllables.

    `Languages supported <https://stanfordnlp.github.io/stanfordnlp/installation_download.html#human-languages-supported-by-stanfordnlp>`_

    `Reference paper <https://www.aclweb.org/anthology/K18-2016>`_

    :param lang: Language of the Neural Network tokeniser
    :param treebank: The neural network model to use based on the treebank 
                     it was trained from. If not given the default treebank 
                     will be used. To see which is the default treebank 
                     and the treebanks available for each language go to this 
                     `link <https://stanfordnlp.github.io/stanfordnlp/models.html#human-languages-supported-by-stanfordnlp>`_
    :param download: If to re-download the model. 
    :returns: A callable that takes a String and returns the tokens for that 
              String.
    '''
    full_treebank_name = stanford_downloader(lang, treebank, download)
    nlp = stanfordnlp.Pipeline(lang=lang, processors='tokenize', 
                               treebank=full_treebank_name)

    def _stanford_doc_to_text(text: str) -> Callable[[str], List[str]]:
        '''
        This returns all of the words in each sentence however in the 
        documentation you do have the option to use the tokens instead but 
        the words are used for downstream application hence why the words 
        were chosen over the tokens. See here for more 
        `details <https://stanfordnlp.github.io/stanfordnlp/pipeline.html#accessing-word-information>`_
        '''
        if text.strip() == '':
            return []
        doc = nlp(text)
        sentences = doc.sentences
        tokens = []
        for sentence in sentences:
             for word in sentence.words:
                 tokens.append(word.text)
        return tokens
    return _stanford_doc_to_text


def token_index_alignment(text: str, tokens: List[str]
                          ) -> List[Span]:
    '''
    :param text: text that has been tokenized
    :param tokens: The tokens that were the output of the text and a tokenizer
                   (tokenizer has to be character preserving)
    :returns: A list of tuples where each tuple contains two ints each 
              representing the start and end index for each of the associated 
              tokens given as an argument.
    '''
    if not is_character_preserving(text, tokens):
        raise ValueError('The tokenization method used is not character'
                         f' preserving. Original text `{text}`\n'
                         f'Tokenized text `{tokens}`')
    token_index_list: List[Span] = []
    char_index = 0
    # Handle whitespace at the start of the text
    if len(text) > char_index:
        while text[char_index] == ' ':
            char_index += 1
            if len(text) <= char_index:
                break

    for token_index, token in enumerate(tokens):
        token_start = char_index
        token_end = token_start
        for token_char_index, token_char in enumerate(token):
            char = text[char_index]
            if token_char == char:
                char_index += 1
            else:
                raise ValueError('The tokenised output within the token should '
                                 f'be the same as the text. Token {token}\n'
                                 f'Text: {text}\nCharacter index {char_index}\n'
                                 f'Token index: {token_index}\nToken char '
                                 f'index {token_char_index}\nTokens {tokens}')
                
        token_end = char_index
        token_index_list.append(Span(token_start, token_end))
        # Covers the whitespaces of n length between tokens and after the text
        if len(text) > char_index:
            while text[char_index] == ' ':
                char_index += 1
                if len(text) <= char_index:
                    break
    
    if char_index != len(text):
        raise ValueError(f'Did not get to the end of the text: {text}\n'
                         f'Character index {char_index}\n'
                         f'Token index list {token_index_list}')
    return token_index_list

