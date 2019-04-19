import json
import numpy as np
import collections
import random


class ASTLoader(object):
    """Handles access of data and provides functionality to build the vocabulary space
    """
    def __init__(self, train_source):
        """
        Args:
            self.word2_index_dict: dictionary that maps words to indices
            self._train_source: variable that stores the path to the file source
            self._train_file: inital file stream
            self._sample_index: integer that handles the indexing of AST Nodes
            self.index_2_word_map: reverse dictionary that is used to map the indices back to the original vocabulary
            self._line_generator: iterable generator object that iterates over individual ASTs
            self._current_line: list containing the Nodes of an AST that is processed next

        Params:
            train_source: path to the source file for the ASTs
        """
        self.word2_index_dict = {}
        self._train_source = train_source
        self._train_file = open(train_source)
        self._sample_index = 0
        self.index_2_word_map = self.index_2_word_map()
        self._line_generator = self.new_line_generator()
        self._current_line = next(self._line_generator)

    def new_line_generator(self):
        """Accesses the file stream and iterate over individual ASTs. Starts from the beginning of the file
        the end of file is reached

        Yields: the next AST
        """
        while True:
            for line in self._train_file:
                yield(self.get_type(line))
            # reload file if we reached the end
            self._train_file.close()
            self._train_file = open(self._train_source)

    def get_type(self, line):
        """gets the type information of a given AST

        Args:
            line: string representing one program AST in the dataset

        Returns:
            list of Node types from the AST
        """

        # read line as json format
        data = json.loads(line)
        # return type information
        words = [Node['type'] for Node in data]
        return [self.word2_index_dict[word] for word in words]

    def index_2_word_map(self):
        """Builds the vocabulary space

        Returns:
            index2_word_dict: Dictionary that translates
                              indices back to original words
        """
        with open(self._train_source) as file:
            for line in file:
                sentence = json.loads(line)
                for word in sentence:
                    if word['type'] not in self.word2_index_dict:
                        self.word2_index_dict[word['type']] \
                            = len(self.word2_index_dict)
        index2_word_dict = dict(zip(self.word2_index_dict.values(),
                                    self.word2_index_dict.keys()))
        return index2_word_dict


    def generate_batch(self, batch_size, skip_window, num_skips):
        """Function for generating a training batch for the skip-gram model

        Args:
            batch_size: Integer specifying number of samples in one batch
            skip_window: Integer specifying how many words to use left and right
            num_skips: Integer specifying how often to reuse a word

        Returns:
            batch: Array of used words
            labels: Array of targets for the words in batch
        """
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window
        batch = np.ndarray(shape=batch_size, dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

        # [skip_window target skip_window]
        span = 2 * skip_window + 1

        # initialize empty buffer
        buffer = collections.deque(maxlen=span)

        # If we are at the end of an data point switch to the next one
        if self._sample_index + span > len(self._current_line):
            self._current_line = next(self.new_line_generator())
            # discard empty ASTs
            while len(self._current_line) < span:
                self._current_line = next(self.new_line_generator())
            # Start from first item if we reach the end of the data
            self._sample_index = 0

        buffer.extend(self._current_line
                      [self._sample_index: self._sample_index + span])
        self._sample_index += span

        for i in range(batch_size // num_skips):
            context_words = [w for w in range(span) if w != skip_window]
            words_to_use = random.sample(context_words, num_skips)
            for j, context_word in enumerate(words_to_use):
                batch[i * num_skips + j] = buffer[skip_window]
                labels[i * num_skips + j, 0] = buffer[context_word]
            if self._sample_index >= len(self._current_line):
                self._current_line = next(self.new_line_generator())
                # discard empty ASTs
                while len(self._current_line) < span:
                    self._current_line = next(self.new_line_generator())
                self._sample_index = 0
                buffer.extend(self._current_line[0:span])
                self._sample_index = span
            else:
                buffer.append(self._current_line[self._sample_index])
                self._sample_index += 1
        return batch, labels
