'''
This modules contains a set of functions that return tokenization functions 
which can be defined by the following typing: Callable[[str], List[str]]. 
All of the functions take exactly no positional arguments but can take 
keyword arguments.
'''
from typing import List, Callable, Optional
from pathlib import Path
import pkgutil

import spacy
import stanfordnlp
from stanfordnlp.utils import resources

from target_extraction.taggers_helper import stanford_downloader

def spacy_tokenizer(lang: str = 'en') -> Callable[[str], List[str]]:
    '''
    Given optionally the language (default English) it will return the 
    Spacy rule based tokeniser for that language but the function will now 
    return a List of String rather than Spacy tokens.

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
        return [spacy_token.text for spacy_token in sapcy_tokenizer_func(text)]
    return _spacy_token_to_text

def whitespace() -> Callable[[str], List[str]]:
    '''
    Standard whitespace tokeniser

    :returns: A callable that takes a String and returns the tokens for that 
              String.
    '''
    return str.split


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

    Languages supported: 
    https://stanfordnlp.github.io/stanfordnlp/installation_download.html#human-
    languages-supported-by-stanfordnlp

    Reference paper:
    https://www.aclweb.org/anthology/K18-2016

    :param lang: Language of the Neural Network tokeniser
    :param treebank: The neural network model to use based on the treebank 
                     it was trained from. If not given the default treebank 
                     will be used. To see which is the default treebank 
                     and the treebanks available for each language go to:
                     https://stanfordnlp.github.io/stanfordnlp/installation_
                     download.html#human-languages-supported-by-stanfordnlp
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
        were chosen over the tokens. See here for more details:
        https://stanfordnlp.github.io/stanfordnlp
        /pipeline.html#accessing-word-information
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
