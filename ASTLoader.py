import json
import numpy as np
import collections
import random


class ASTLoader(object):
    def __init__(self, train_source, num_samples):
        self._train_source = train_source
        self._num_samples = num_samples
        self._sample_index = 0
        self._data_index = 0
        self.types_train = None
        self.data = None
        self.index2_word_dict = None

    def load_types(self):
        """loads types into training self.types_train"""
        # open source file
        _file_train = open(self._train_source)

        # read lines and format data
        data_train = [json.loads(_file_train.readline())
                      for i in range(self._num_samples[0])]

        # save the data
        self.types_train = [[Node['type'] for Node in data_train[i]]
                            for i in range(self._num_samples[0])]

    def build_dataset(self):
        """Builds the vocabulary and translates self.types_train
           into the vocabulary space

            Returns:
                data: An index representation of train_data

                index2_word_dict: Dictionary that translates
                                  indices back to original words
            """
        word2_index_dict = {}
        for sentence in self.types_train:
            for word in sentence:
                if word not in word2_index_dict:
                    word2_index_dict[word] = len(word2_index_dict)
        index2_word_dict = dict(zip(word2_index_dict.values(),
                                    word2_index_dict.keys()))

        # build the data
        data = [[word2_index_dict.get(word, 0) for word in sentence]
                for sentence in self.types_train]

        self.data = data
        self.index2_word_dict = index2_word_dict

        return data, index2_word_dict

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
        if self._sample_index + span > len(self.data[self._data_index]):
            self._data_index += 1

            # Start from first item if we reach the end of the data
            self._data_index = self._data_index % len(self.data)
            self._sample_index = 0

        buffer.extend(self.data[self._data_index]
                      [self._sample_index: self._sample_index + span])
        self._sample_index += span

        for i in range(batch_size // num_skips):
            context_words = [w for w in range(span) if w != skip_window]
            words_to_use = random.sample(context_words, num_skips)
            for j, context_word in enumerate(words_to_use):
                batch[i * num_skips + j] = buffer[skip_window]
                labels[i * num_skips + j, 0] = buffer[context_word]
            if self._sample_index >= len(self.data[self._data_index]):
                self._data_index += 1
                self._data_index = self._data_index % len(self.data)
                self._sample_index = 0
                buffer.extend(self.data[self._data_index][0:span])
                self._sample_index = span
            else:
                buffer.append(self.data[self._data_index][self._sample_index])
                self._sample_index += 1
        return batch, labels