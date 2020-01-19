## Changes by Artur Andrzejak (Jan 2020)
# 1. Refactored

import json
from collections import Counter, defaultdict

from six.moves import cPickle as pickle


class ProcessorForNonTerminals():
    MAX_NODES_PER_AST = 3e4

    def __init__(self):
        # instance variables, previously were global vars

        # Encoding of node types via consecutive integers
        # (as a val-auto-incrementing default dict, see http://bit.ly/2QHVQyo)
        self.nodeType_to_ID = defaultdict(lambda: len(self.nodeType_to_ID))
        self.nodes_with_terminal_values = set()
        self.set_all_IDs = set()  # the set to include all (sparse) ID
        self.sparseIDs_to_denseIDs = dict()  # map sparse id to dense id (remove empty id inside 4*base_ID)

    def process_file(self, filename: str):
        """
        Fills the type_dict and calculates the number of types.
        Also converts AST terminal names to IDs (stored in nodeType_to_ID) - contains information about children and siblings
        :param filename: File to be processed
        :return: corpus_node_encoding - IDs for N with information about children and siblings,
                corpus_parent_offsets- Offset of each node to its parent (according to json file)
        """
        with open(filename, encoding='latin-1') as lines:
            corpus_node_encoding = list()
            corpus_parent_offsets = list()

            for line_index, line in enumerate(lines):
                if line_index % 1000 == 0:
                    print('Processing line: ', line_index)

                ast_decoded = json.loads(line)
                if len(ast_decoded) < ProcessorForNonTerminals.MAX_NODES_PER_AST:
                    full_ast_encoding, parent_list = self.process_AST(ast_decoded)
                    corpus_node_encoding.append(full_ast_encoding)
                    corpus_parent_offsets.append(parent_list)

            return corpus_node_encoding, corpus_parent_offsets

    def process_AST(self, ast_decoded: dict):
        # Encodes type (+ info on children and siblings) of each ast node (result)
        full_ast_encoding = list()
        # Encodes offsets of this node to its parent (according to json line order) (result)
        parent_offset_encoding = list()
        # If node with index idx has siblings, we have nodeIdx_to_siblingFlag[idx] == 1
        nodeIdx_to_siblingFlag = Counter()
        # for node with index idx,  nodeIdx_to_parentOffset[idx] = offset of node to parent par_idx (i.e. idx - par_idx)
        nodeIdx_to_parentOffset = defaultdict(lambda: 1)  # default parent is previous 1

        for node_idx, node_elements in enumerate(ast_decoded):
            typeName = node_elements['type']
            # For each unknown key, insert new value:= len(dict) (i.e. new vals are n, n+1, ...)
            base_ID = self.nodeType_to_ID[typeName]

            has_children = 'children' in node_elements
            has_sibling = nodeIdx_to_siblingFlag[node_idx] > 0
            ID = self.encode_sibling_and_children_info(base_ID, has_children, has_sibling)

            # Update infos for children of this node
            if has_children:
                children = node_elements['children']
                self.update_siblings_data(children, nodeIdx_to_siblingFlag)
                self.update_parent_offsets(children, nodeIdx_to_parentOffset, node_idx)

            # record node IDs which have a non-empty Terminal
            if 'value' in node_elements:
                self.nodes_with_terminal_values.add(ID)

            full_ast_encoding.append(ID)
            parent_offset_encoding.append(nodeIdx_to_parentOffset[node_idx])
            self.set_all_IDs.add(ID)

        return full_ast_encoding, parent_offset_encoding

    def update_parent_offsets(self, children, nodeIdx_to_parentOffset, node_idx):
        for j in children:
            nodeIdx_to_parentOffset[j] = j - node_idx

    def update_siblings_data(self, children, sibling_flags):
        # Sibling information is got by the ancestor's children information
        if len(children) > 1:
            for j in children:
                sibling_flags[j] = 1

    def encode_sibling_and_children_info(self, base_ID, has_children_flag, has_sibling_flag):
        """
            Encodes info on siblings and children of this node into bit 0 and bit 1
                * has sibling: bit 0 set to 1
                * has children: bit 1 set to 1
            In any case, the base_ID is becomes 4*base_ID
        :return: 4*base_ID + info on siblings and children of this node in bit 0 and bit 1
        """
        result_id = base_ID * 4
        result_id += 1 if has_sibling_flag else 0
        result_id += 2 if has_children_flag else 0

        # Following kept just as a reminiscence ...
        # if has_children_flag:
        #     if has_sibling_flag:
        #         result_id = base_ID * 4 + 3
        #     else:
        #         result_id = base_ID * 4 + 2
        # else:
        #     if has_sibling_flag:
        #         result_id = base_ID * 4 + 1
        #     else:
        #         result_id = base_ID * 4
        return result_id

    def map_dense_id(self, corpus_node_encoding_sparse_IDs):
        """
        Create a new "non-terminal data" encoding, removing all gaps bw existing IDs
        :param corpus_node_encoding_sparse_IDs:
        :return: A new encoding which uses all values 0, 1, ..., |# different IDs|
        """
        result = list()
        for line_id in corpus_node_encoding_sparse_IDs:
            line_new_id = list()
            for i in line_id:
                if i in self.sparseIDs_to_denseIDs:
                    line_new_id.append(self.sparseIDs_to_denseIDs[i])
                else:
                    self.sparseIDs_to_denseIDs[i] = len(self.sparseIDs_to_denseIDs)
                    line_new_id.append(self.sparseIDs_to_denseIDs[i])
            result.append(line_new_id)
        return result

    def get_empty_set_dense(self):
        vocab_size = len(self.set_all_IDs)
        assert len(self.sparseIDs_to_denseIDs) == vocab_size
        assert self.nodes_with_terminal_values.issubset(self.set_all_IDs)
        ids_of_nodes_without_terminals = self.set_all_IDs.difference(self.nodes_with_terminal_values)
        empty_set_dense = set()

        for i in ids_of_nodes_without_terminals:
            empty_set_dense.add(self.sparseIDs_to_denseIDs[i])
        return empty_set_dense, vocab_size

    def save(self, filename, vocab_size, trainData, testData, trainParent, testParent, empty_set_dense,
             encoderTestData):
        """
        :param filename: Name of destination (pickle file)
        :param vocab_size: the vocabulary size to restrict the number of words
        :param trainData: Processed training data (mapped to respective IDs)
        :param testData: Processed testing data (mapped to respective IDs)
        :param trainParent: Offsets to parent in training set
        :param testParent: Offsets to parent in testing set
        :param empty_set_dense: Dense set of non-terminals who can't have value

        """
        with open(filename, 'wb') as f:
            save = {
                'nodeType_to_ID': dict(self.nodeType_to_ID),
                'numType': len(self.nodeType_to_ID),
                'dicID': self.sparseIDs_to_denseIDs,
                'vocab_size': vocab_size,
                'trainData': trainData,
                'testData': testData,
                'trainParent': trainParent,
                'testParent': testParent,
                'typeOnlyHasEmptyValue': empty_set_dense,
                'encoderTestData': encoderTestData
            }
            pickle.dump(save, f, protocol=2)

    def process_all_and_save(self, train_filename, test_filename, target_filename):
        print('Start procesing %s' % (train_filename))
        train_data, train_parent_offsets = self.process_file(train_filename)
        print('Start procesing %s' % (test_filename))
        test_data, test_parent_offset = self.process_file(test_filename)

        # todo: clean up the following; some args should become instance fields
        train_data = self.map_dense_id(train_data)
        test_data = self.map_dense_id(test_data)
        empty_set_dense, vocab_size = self.get_empty_set_dense()

        print('The N set that only has empty terminals: ', len(empty_set_dense), empty_set_dense)
        print('The vocabulary:', vocab_size, self.set_all_IDs)

        print("Saving results ...")

        self.save(target_filename, vocab_size, train_data, test_data,
                  train_parent_offsets, test_parent_offset, empty_set_dense, None)


def main(train_filename, test_filename, target_filename) -> None:
    """
    Main routine for processing called by a centralized script
    :param train_filename:
    :param test_filename:
    :param target_filename:
    """
    processor = ProcessorForNonTerminals()
    # processor.process_all_and_save(train_filename, test_filename, target_filename)
    processor.process_all_and_save(train_filename, test_filename, target_filename)


if __name__ == '__main__':
    train_filename = '../../data/python100k_train.json'
    test_filename = '../../data/python50k_eval.json'
    debug_filename = '../../data/python10_debug.json'
    target_filename = '../pickle_data/PY_non_terminal_with_location.pickle'
    target_filename_fake = '../pickle_data/PY_non_terminal_with_location_fake.pickle'
    target_debug_filename_fake = '../pickle_data/PY_non_terminal_debug.pickle'

#    main(train_filename=train_filename, test_filename=test_filename, target_filename=target_filename_fake)
    main(train_filename=debug_filename, test_filename=debug_filename, target_filename=target_debug_filename_fake)
