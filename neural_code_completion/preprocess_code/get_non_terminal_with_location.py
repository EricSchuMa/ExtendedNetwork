## Changed by Artur Andrzejak on 10-01-2020
# 1. Refactored
# 2. Added information about location to each AST node

from six.moves import cPickle as pickle
import json
import time
from collections import Counter, defaultdict
import copy
import itertools


class ProcessorForNonTerminals(object):

    def __init__(self):
        # instance variables, previously were global vars
        # map N's name into its original ID(before expanding into 4*base_ID)
        # It is an autoincrementing ID table, see http://bit.ly/2QHVQyo
        self.typeDict = defaultdict(lambda: len(self.typeDict))
        self.all_sparse_IDs = set()  # the set to include all sparse ID
        self.dicID = dict()  # map sparse id to dense id (remove empty id inside 4*base_ID)
        self.ids_of_nodes_with_terminals = set()

    def process_file(self, filename):
        """
        Fills the type_dict and calculates the number of types.
        Also converts AST terminal names to IDs (stored in typeDict) - contains information about children and siblings
        :param filename: File to be processed
        :return: corpus_N - IDs for N with information about children and siblings, corpus_parent - Offsets to parents
        """
        with open(filename, encoding='latin-1') as lines:
            corpus_N = list()
            corpus_parent = list()

            for line_index, line in enumerate(lines):
                if line_index % 1000 == 0:
                    print('Processing line: ', line_index)

                line_N, parent_list = self._process_line(line)
                if line_N:
                    corpus_N.append(line_N)
                    corpus_parent.append(parent_list)
            return corpus_N, corpus_parent

    def _process_line(self, line):
        data = json.loads(line)
        if len(data) >= 3e4:
            return None, None

        line_N = list()
        has_sibling = Counter()
        parent_counter = defaultdict(lambda: 1)  # default parent is previous 1
        parent_list = list()

        for i, dic in enumerate(data):  # JS data[:-1] or PY data
            typeName = dic['type']
            # This is an auto-incrementing default dict
            base_ID = self.typeDict[typeName]

            # expand the ID into the range of 4*base_ID, according to whether it has sibling or children.
            # Sibling information is got by the ancestor's children information

            if 'children' in dic:
                if has_sibling[i]:
                    ID = base_ID * 4 + 3
                else:
                    ID = base_ID * 4 + 2

                childs = dic['children']
                for j in childs:
                    parent_counter[j] = j - i

                if len(childs) > 1:
                    for j in childs:
                        has_sibling[j] = 1
            else:
                if has_sibling[i]:
                    ID = base_ID * 4 + 1
                else:
                    ID = base_ID * 4
            # record the Non-terminals which have a non-empty Terminal
            if 'value' in dic:
                self.ids_of_nodes_with_terminals.add(ID)

            line_N.append(ID)
            parent_list.append(parent_counter[i])
            self.all_sparse_IDs.add(ID)
        return line_N, parent_list

    def map_dense_id(self, data):
        result = list()
        for line_id in data:
            line_new_id = list()
            for i in line_id:
                if i in self.dicID:
                    line_new_id.append(self.dicID[i])
                else:
                    self.dicID[i] = len(self.dicID)
                    line_new_id.append(self.dicID[i])
            result.append(line_new_id)
        return result

    def get_empty_set_dense(self):
        vocab_size = len(self.all_sparse_IDs)
        assert len(self.dicID) == vocab_size
        assert self.ids_of_nodes_with_terminals.issubset(self.all_sparse_IDs)
        ids_of_nodes_without_terminals = self.all_sparse_IDs.difference(self.ids_of_nodes_with_terminals)
        empty_set_dense = set()
        # print('The dicID: %s' % dicID)
        # print('The vocab_size: %s' % vocab_size)
        for i in ids_of_nodes_without_terminals:
            empty_set_dense.add(self.dicID[i])
        return empty_set_dense, vocab_size

    def save(self, filename, vocab_size, trainData, testData, trainParent, testParent, empty_set_dense):
        """
        :param filename: Name of destination (pickle file)
        :param vocab_size: the vocabulary size to restrict the number of words
        :param trainData: Processed training data (mapped to respective IDs)
        :param testData: Processed testing data (mapped to respective IDs)
        :param trainParent: Offsets to parent in training set
        :param testParent: Offsets to parent in testing set
        :param empty_set_dense: Dense set of non-terminals who can't have value

        Used from fields:
        typeDict: Dictionary for inferring types to IDs
        numType: Number of total types
        dicID: Maps sparse IDs to dense IDs
        """
        with open(filename, 'wb') as f:
            save = {
                'typeDict': dict(self.typeDict),
                'numType': len(self.typeDict),
                'dicID': self.dicID,
                'vocab_size': vocab_size,
                'trainData': trainData,
                'testData': testData,
                'trainParent': trainParent,
                'testParent': testParent,
                'typeOnlyHasEmptyValue': empty_set_dense,
            }
            pickle.dump(save, f, protocol=2)

    def process_all_and_save(self, train_filename, test_filename, target_filename, useFakeTestData=False):
        print('Start procesing %s' % (train_filename))
        trainData, trainParent = self.process_file(train_filename)
        print('Start procesing %s' % (test_filename))
        testData, testParent = self.process_file(test_filename)

        # todo: clean up the following; some args should become instance fields
        trainData = self.map_dense_id(trainData)
        testData = self.map_dense_id(testData)
        empty_set_dense, vocab_size = self.get_empty_set_dense()

        print('The N set that only has empty terminals: ', len(empty_set_dense), empty_set_dense)
        print('The vocabulary:', vocab_size, self.all_sparse_IDs)

        print("Saving results ...")
        if useFakeTestData:
            encoderTestData = DataEncoder(base_integer=10000)
            testDataFake = encoderTestData.encode(testData)
            encoderTestParent = DataEncoder(base_integer=100000)
            testParentFake = encoderTestParent.encode(testParent)

            self.save(target_filename, vocab_size, trainData, testDataFake, trainParent, testParentFake,
                      empty_set_dense)
            return {'encoderTestData': encoderTestData, 'encoderTestParent': encoderTestParent }
        else:
            self.save(target_filename, vocab_size, trainData, testData, trainParent, testParent, empty_set_dense)
            return None



class DataEncoder:
    """
    Encodes input data by unique integers (keys), and stores a mapping keys => original data.
    Maintains the structure of the original data.
    """

    def __init__(self, base_integer=10000):
        self.keys_to_original_values = dict()
        self.base_integer = base_integer

    def _transform(self, input_list_of_list, output_list_of_list, map_element_func, side_effect_func=None):
        step_idx = 0
        for outer_idx, outer_list in enumerate(input_list_of_list):
            for inner_idx, inner_val in enumerate(outer_list):
                key = map_element_func(inner_val, step_idx)
                output_list_of_list[outer_idx][inner_idx] = key
                if side_effect_func:
                    side_effect_func(key=key, value=inner_val)
                step_idx += 1
        return output_list_of_list

    def encode(self, original_values):
        """
        Recursively walks through input data, and for each entry creates a unique key, and stores (key, original_value)
        in self.keys_to_original_values. Note: unique only "locally" in the input.
        :param original_values: list of lists with original values
        :return: data structure with same dims as input but entries replaced by unique keys
        """

        assert (isinstance(original_values, list))
        assert (len(original_values) == 0 or isinstance(original_values[0], list))

        def generate_next_key(value, step_idx):
            return self.base_integer + step_idx

        def dict_update(key, value):
            self.keys_to_original_values[key] = value

        output_list_of_lists = copy.deepcopy(original_values)
        return self._transform(original_values, output_list_of_lists, map_element_func=generate_next_key,
                               side_effect_func=dict_update)

    def decode(self, encoded_values):
        """
        Returns a data structure of same shape as the encoded values, but each value replaced by
        a corresponding entry from self.keys_to_original_values (i.e. kind of
        :param encoded_values: list of lists with keys to original values
        :return: data structure with same dims as input but keys (entries) replaced by values
        """

        output_list_of_lists = copy.deepcopy(encoded_values)
        return self._transform(encoded_values, output_list_of_lists,
                               map_element_func=lambda key, step_idx: self.keys_to_original_values[key])


def main(train_filename, test_filename, target_filename) -> None:
    """
    Main routine for processing called by a centralized script
    :param train_filename:
    :param test_filename:
    :param target_filename:
    """
    processor = ProcessorForNonTerminals()
    # processor.process_all_and_save(train_filename, test_filename, target_filename)
    processor.process_all_and_save(train_filename, test_filename, target_filename, useFakeTestData=True)


if __name__ == '__main__':
    train_filename = '../../data/python100k_train.json'
    test_filename = '../../data/python50k_eval.json'
    target_filename = '../pickle_data/PY_non_terminal_with_location.pickle'
    target_filename_fake = '../pickle_data/PY_non_terminal_with_location_fake.pickle'

    main(train_filename=train_filename, test_filename=test_filename, target_filename=target_filename_fake)
