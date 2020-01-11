## Changed by Artur Andrzejak on 10-01-2020
# 1. Refactored
# 2. Added information about location to each AST node

from six.moves import cPickle as pickle
import json
import time
from collections import Counter, defaultdict


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
                if i in self.dicID.keys():
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

    def process_all_and_save(self, train_filename, test_filename, target_filename):
        print('Start procesing %s' % (train_filename))
        trainData, trainParent = self.process_file(train_filename)
        print('Start procesing %s' % (test_filename))
        testData, testParent = self.process_file(test_filename)

        # todo: clean up the following; some args should become instance fields
        trainData = self.map_dense_id(trainData)
        testData = self.map_dense_id(testData)
        empty_set_dense, vocab_size = self.get_empty_set_dense()

        print("Saving results ...")
        self.save(target_filename, vocab_size, trainData, testData, trainParent, testParent, empty_set_dense)
        print('The N set that only has empty terminals: ', len(empty_set_dense), empty_set_dense)
        print('The vocabulary:', vocab_size, self.all_sparse_IDs)


# todo: Create a top-level file for all preprocessing.
#  Use explicit file names and params only in this file.
if __name__ == '__main__':
    train_filename = '../../data/python100k_train.json'
    test_filename = '../../data/python50k_eval.json'
    target_filename = '../pickle_data/PY_non_terminal_with_location.pickle'

    start_time = time.time()
    processor = ProcessorForNonTerminals()
    processor.process_all_and_save(train_filename, test_filename, target_filename)
    print('Finished generating terminals. It took %.2fs' % (time.time() - start_time))
