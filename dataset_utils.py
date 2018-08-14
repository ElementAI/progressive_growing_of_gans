import h5py
import string
import logging

from typing import Iterator, List, Set, Optional, Union, Callable
from collections import Counter

from tqdm import tqdm

BLACK_LIST = string.punctuation.replace('%', '') + '\n'


def normalize(text: str,
              black_list: str = BLACK_LIST,
              vocab: Optional[Set] = None,
              lowercase: bool = True,
              tokenize: bool = True) -> Union[str, List[str]]:
    ''' text preprocessing function

    Arguments:
        text       - text to be processed
        black_list - string with blacklisted charcters
        vocab      - if set, filters words not in the vocab
        lowercase  - if True, lowercases the words
        tokenize   - if True, returns a list of tokens
    '''
    if black_list:
        text = text.translate(str.maketrans(BLACK_LIST, ' ' * len(BLACK_LIST)))
    if lowercase:
        text = text.lower()
    if vocab:
        text = ' '.join([word for word in text.split() if word in vocab])
    if tokenize:
        return text.split()
    else:
        return ' '.join(text.split())


def stream_words_from_h5py(
        file_path: str,
        preprocessing_fn: Callable[[str], List[str]]) -> Iterator[str]:
    ''' Returns an interator streaming words from the source h5py file

    Arguments:
        file_path         - absolute path to the file
        preprocessing_fn  - takes text and returns tokenized version
    '''
    with h5py.File(file_path, 'r') as data_file:
        for text in tqdm(data_file['input_description']):
            # it's numpy byes
            text = text[0].decode('latin1')
            yield from preprocessing_fn(text)


class Vocabulary:
    ''' Holds info about vocabulary '''

    def __init__(self,
                 words: Iterator[str],
                 preprocessing_fn: Callable[[str], List[str]],
                 min_count: Optional[int] = None,
                 word_limit: Optional[int] = None):
        '''
        Arguments:
            words            - yield words to construct word counter
            preprocessing_fn - takes text and returns tokenized version
            min_count        - if set, filters word less frequent than the limit
            word_limit       - if set, cuts of the number of words used to build the
                                dictionary
        '''
        self.word2idx = {'UNK': 0, '<EOS>': 1, '<PAD>': 2}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.preprocessing_fn = preprocessing_fn

        self.counts = Counter(words)
        if min_count:
            self.counts = Counter({
                k: c for k, c in self.counts.items() if c >= min_count
            })

        self.counts = self.counts.most_common(word_limit)

        for i, (word, count) in enumerate(self.counts, len(self)):
            self.word2idx[word] = i
            self.idx2word[i] = word

        logger = logging.getLogger(__name__)
        logger.info("Loaded dictionary with %d words" % len(self.idx2word))

    def __word2idx(self, word: str) -> int:
        return self.word2idx.get(word, self.word2idx['UNK'])

    def __len__(self):
        return len(self.word2idx)

    def encode(self, text: str) -> List[int]:
        return [self.__word2idx(word) for word in self.preprocessing_fn(text)]

    def decode(self, idxs: List[int]) -> str:
        return ' '.join([self.idx2word[idx] for idx in idxs])


if __name__ == '__main__':
    import argparse
    import json
    import os
    import pickle

    parser = argparse.ArgumentParser(description='Script to build and save vocabulary')
    parser.add_argument('input_path', help='Directory containing H5 dataset')
    parser.add_argument('output_path', help='Directory where to store the vocabulary')

    parser.add_argument('--min_count', type=int, help='if set, filters word less frequent than the limit')
    parser.add_argument('--word_limit', type=int, help='if set, cuts of the number of words used to build'
                                                       'the dictionary')
    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info('Loading data from %s' % args.input_path)

    # build an interator
    words = stream_words_from_h5py(args.input_path, normalize)

    # build the vocabulary
    vocab = Vocabulary(words, normalize, min_count=args.min_count, word_limit=args.word_limit)

    with open(os.path.join(args.output_path, 'vocab.pickle'), 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(args.output_path, 'word2idx.json'), 'wt') as f:
        json.dump(vocab.word2idx, f, indent=4)

    with open(os.path.join(args.output_path, 'idx2word.json'), 'wt') as f:
        json.dump(vocab.idx2word, f, indent=4)

    with open(os.path.join(args.output_path, 'counts.json'), 'wt') as f:
        json.dump(vocab.counts, f, indent=4)
