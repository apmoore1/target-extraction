'''
This modules contains a set of functions that return tokenization functions 
which can be defined by the following typing: Callable[[str], List[str]]. 
All of the functions take exactly no positional arguments but can take 
keyword arguments.
'''
from typing import List, Callable, Optional, Tuple
from pathlib import Path
import pkgutil

import spacy
import stanfordnlp
from stanfordnlp.utils import resources

from target_extraction.taggers_helper import stanford_downloader


def is_character_preserving(original_text: str, text_tokens: List[str]
                            ) -> bool:
    '''
    :param original_text: Text that has been tokenized
    :param text_tokens: List of tokens after the text has been tokenized
    :returns: True if the tokenized text when all characters are joined 
                together is equal to the original text with all it's 
                characters joined together.
    '''
    tokens_text = ''.join(text_tokens)
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


def token_index_alignment(text: str, tokenizer: Callable[[str], List[str]]
                          ) -> List[Tuple[str, Tuple[int, int]]]:
    tokens = tokenizer(text)
    if not is_character_preserving(text, tokens):
        raise ValueError('The tokenization method used is not character'
                         f' preserving. Original text `{text}`\n'
                         f'Tokenized text `{tokens}`')
    token_index_list: List[Tuple[str, Tuple[int, int]]] = []
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
        token_index_list.append((token, (token_start, token_end)))
        # Covers the whitespaces of n length between tokens and after the text
        if len(text) > char_index:
            print(f'index {char_index} token {token}')
            while text[char_index] == ' ':
                char_index += 1
                print(char_index)
                print(len(text))
                if len(text) <= char_index:
                    print('done')
                    break
            print(f'After index {char_index} token {token}')
    
    if char_index != len(text):
        print(char_index)
        print(len(text))
        raise ValueError(f'Did not get to the end of the text: {text}\n'
                         f'Character index {char_index}\n'
                         f'Token index list {token_index_list}')
    return token_index_list


def token_index_alignment_1(text: str, tokenizer: Callable[[str], List[str]]
                          ) -> List[Tuple[str, Tuple[int, int]]]:
    '''
    :param text: text to tokenise
    :param tokenizer: The tokenizer to use to tokenize the text 
                      (has to be character preserving)
    :returns: A list of tuples where each tuple contains the token from the 
              tokenised text with another tuple stating the index from the 
              original text.
    '''
    tokens = tokenizer(text)
    if not is_character_preserving(text, tokens):
        raise ValueError('The tokenization method used is not character'
                         f' preserving. Original text `{text}`\n'
                         f'Tokenized text `{tokens}`')
    token_index = 0
    token_char_index = 0
    token_start_index = 0
    token_end_index = 0
    token_index_list = []
    token = tokens[token_index]
    finished = False
    for char_index, char in enumerate(text):
        # The case where the tokenization produces more whitespace than their 
        # is naturally due to the tokenization
        last_char_index = char_index - 1
        if len(token) == token_char_index:
            # end of one token start of another
            finished = True
        elif char == ' ' and token[token_char_index] == text[last_char_index]:
            token_char_index += 1
            #if char != 

        if finished:
            finished = False
            token_end_index = last_char_index
            end_char = text[token_end_index]
            if end_char == ' ':
                raise ValueError('Token cannot end on whitespace '
                                 f'{token}, text {text} start and end index '
                                 f'{token_start_index} {last_char_index}')
            token_index_list.append((token, (token_start_index, char_index)))
            token_start_index = 0
            token_end_index = 0
            token_index += 1
            token = tokens[token_index]
            token_char_index = 0
        else:
            if char == token[token_char_index]:
                if token_char_index == 0:
                    if char == ' ':
                        raise ValueError('Token cannot start on whitespace '
                                        f'{token}, text {text} char index {char_index}')
                    token_start_index = char_index
            else:
                raise ValueError(f'{token} {text} {char} {char_index} {token_index}')
            token_char_index += 1
    if token_end_index == 0 and token_start_index != 0:
        token_index_list.append((token, (token_start_index, char_index + 1)))
    if len(token_index_list) != len(tokens):
        raise ValueError('The number of tokens through tokenization is not '
                         'the same as the number of tokens produced through '
                         f'appending the token index: Original tokens {tokens}'
                         f'\nTokens index: {token_index_list}')
    return token_index_list

