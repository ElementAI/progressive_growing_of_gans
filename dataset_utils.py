import h5py
import string
import logging

from collections import Counter, defaultdict
from itertools import islice, tee
from typing import Iterator, List, Set, Optional, Union, Callable, Tuple

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
        black_list - string with blacklisted characters
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
    ''' Returns an iterator streaming words from the source HDF5 file

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
        self.min_count = min_count
        self.word_limit = word_limit

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
        logger.info("Build a dictionary with %d words" % len(self.idx2word))

    def __word2idx(self, word: str) -> int:
        return self.word2idx.get(word, self.word2idx['UNK'])

    def __len__(self):
        return len(self.word2idx)

    def encode(self, text: str) -> List[int]:
        return [self.__word2idx(word) for word in self.preprocessing_fn(text)]

    def decode(self, idxs: List[int]) -> str:
        return ' '.join([self.idx2word[idx] for idx in idxs])

    def __str__(self) -> str:
        return "Vocab with %d words, word limit of %d, minimum allowed count: %d" % (
            len(self), self.word_limit, self.min_count)


def build_vocab(input_path: str,
                output_path: str,
                min_count: Optional[int] = None,
                word_limit: Optional[int] = None) -> None:
    ''' Creates and saves a Vocabulary object '''
    import json
    import os
    import pickle

    logger = logging.getLogger(__name__)
    logger.info('Loading data from %s' % input_path)

    # build an iterator
    words = stream_words_from_h5py(input_path, normalize)

    # build the vocabulary
    vocab = Vocabulary(words, normalize, min_count=min_count, word_limit=word_limit)

    logger.info("Saving %s to %s" % (str(vocab), output_path))
    with open(os.path.join(output_path, 'vocab.pickle'), 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(output_path, 'word2idx.json'), 'wt') as f:
        json.dump(vocab.word2idx, f, indent=4)

    with open(os.path.join(output_path, 'idx2word.json'), 'wt') as f:
        json.dump(vocab.idx2word, f, indent=4)

    with open(os.path.join(output_path, 'counts.json'), 'wt') as f:
        json.dump(vocab.counts, f, indent=4)


def save_encoded(data_path: str,
                 vocab_path: str,
                 add_text: bool = False) -> None:
    ''' Save mapping from product id to product description '''
    import os
    import pickle
    import h5py

    keys = ['input_productID', 'input_description']

    id2desc = defaultdict(dict)
    vocabulary = pickle.load(open(os.path.join(vocab_path, 'vocab.pickle'), 'rb'))

    logger = logging.getLogger(__name__)
    logger.info('Building id2desc')

    with h5py.File(data_path, 'r') as f:
        for product_id, description in tqdm(zip(*[f[k] for k in keys])):
            product_id = product_id[0]
            description = description[0].decode('latin1')
            if product_id not in id2desc:
                id2desc[product_id]['encoded'] = vocabulary.encode(description)
                if add_text:
                    id2desc[product_id]['text'] = description

    logger.info('Saving id2desc')
    with open(os.path.join(vocab_path, 'id2desc.pickle'), 'wb') as f:
        pickle.dump(id2desc, f, pickle.HIGHEST_PROTOCOL)


def consume(iterator, n=None):
    ''' Advance the iterator n-steps ahead. If n is None, consume entirely '''
    # Use functions that consume iterators at C speed.
    if n is None:
        # feed the entire iterator into a zero-length deque
        collections.deque(iterator, maxlen=0)
    else:
        # advance to the empty slice starting at position n
        next(islice(iterator, n, n), None)


def bigrams(iterable: Iterator[str]) -> Iterator[Tuple[str, str]]:
    ''' Create bi-gram form an iterable
    i -> (i_0, i_1), (i_1, i_2), (i_2, i_3), ...

       > for bigram in bigrams(['all', 'this', 'happened', 'more', 'or', 'less']):
    ...:     print(bigram)
    ...:

    ('all', 'this')
    ('this', 'happened')
    ('happened', 'more')
    ('more', 'or')
    ('or', 'less')
    '''
    a, b = tee(iterable)
    consume(b, 1)
    return zip(a, b)


def ngrams(iterable: Iterator[str], n: int) -> Iterator[Tuple[str, ...]]:
    ''' Create n-grams form an iterable
    i -> (i_0, i_1, ..., i_n), (i_1, i_2, ..., i_n+1),  ...

    Example:
       > for ngram in ngrams(['all', 'this', 'happened', 'more', 'or', 'less'], 3):
    ...:     print(ngram)

    ('all', 'this', 'happened')
    ('this', 'happened', 'more')
    ('happened', 'more', 'or')
    ('more', 'or', 'less')
    '''
    iters = tee(iterable, n)
    for skip, i in enumerate(iters):
        consume(i, skip)

    return zip(*iters)


def save_ngrams(input_path: str,
                n: Optional[int] = 2) -> None:
    ''' saves n-grams '''
    words = stream_words_from_h5py(input_path, normalize)
    counter = Counter(ngrams(words, n))
    with open(f"/data/vocab/ngrams-{n}.txt", 'wt') as f:
        for ngram, count in counter.most_common():
            f.write(f"{count:<6} | {' '.join(ngram)}\n")


def execute_cmdline(argv):
    '''
    This function is what you get if you want the codebase to be consistent, but
    the original code didn't use Click
    '''
    import argparse

    prog = argv
    parser = argparse.ArgumentParser(description='Script to build and save vocabulary')

    subparsers = parser.add_subparsers(dest='command')
    subparsers.required = True

    def add_command(cmd, desc, example=None):
        epilog = 'Example: %s %s' % (prog, example) if example is not None else None
        return subparsers.add_parser(cmd, description=desc, help=desc, epilog=epilog)

    # add command to build the vocabulary
    p = add_command('build_vocab', 'build the vocabulary')
    p.add_argument('input_path', help='Directory containing HDF5 dataset')
    p.add_argument('output_path', help='Directory where to store the vocabulary')
    p.add_argument('--min_count', type=int, help='if set, filters word less frequent than the limit')
    p.add_argument('--word_limit', type=int, help='if set, cuts of the number of words used to build'
                                                   'the dictionary')

    # add command to build n-grams
    p = add_command('save_encoded', 'Calculates the token n-grams')
    p.add_argument('data_path', help='Directory containing HDF5 dataset')
    p.add_argument('vocab_path', help='Directory containing pickled vocabulary')
    p.add_argument('--add_text', help='Whether to store source text or not', action='store_true')

    # add command to build n-grams
    p = add_command('save_ngrams', 'Calculates the token n-grams')
    p.add_argument('input_path', help='Directory containing HDF5 dataset')
    p.add_argument('--n', type=int, help='The ngram level')

    # parse the arguments
    args = parser.parse_args(argv[1:] if len(argv) > 1 else ['-h'])
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f"Executing: {args.command}")

    # fetch the function from globals
    func = globals()[args.command]
    del args.command
    func(**vars(args))


if __name__ == '__main__':
    import sys
    execute_cmdline(sys.argv)
