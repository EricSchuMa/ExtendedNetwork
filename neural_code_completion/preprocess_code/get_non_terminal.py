# rewrite on 2018/1/8 by xxx, add parent

import numpy as np
from six.moves import cPickle as pickle
import json
import time
from collections import Counter, defaultdict

# attention line 42: for python dataset, not exclude the last one
train_filename = '../../data/python100k_train.json'
test_filename = '../../data/python50k_eval.json'
target_filename = '../pickle_data/PY_non_terminal_small.pickle'

# global variables
typeDict = dict()  # map N's name into its original ID(before expanding into 4*base_ID)
numID = set()  # the set to include all sparse ID
no_empty_set = set()
typeList = list()  # the set to include all Types
numType = 0
dicID = dict()  # map sparse id to dense id (remove empty id inside 4*base_ID)


def process(filename):
    """
    Fills the type_dict and calculates the number of types.
    Also converts AST terminal names to IDs (stored in typeDict) - contains information about children and siblings
    :param filename: File to be processed
    :return: corpus_N - IDs for N with information about children and siblings, corpus_parent - Offsets to parents
    """
    with open(filename, encoding='latin-1') as lines:
        print('Start procesing %s !!!' % (filename))
        line_index = 0
        corpus_N = list()
        corpus_parent = list()

        for line in lines:
            line_index += 1
            if line_index % 1000 == 0:
                print('Processing line: ', line_index)
            data = json.loads(line)
            line_N = list()
            has_sibling = Counter()
            parent_counter = defaultdict(lambda: 1)  # default parent is previous 1
            parent_list = list()

            if len(data) >= 3e4:
                continue

            for i, dic in enumerate(data):  # JS data[:-1] or PY data
                typeName = dic['type']
                if typeName in typeList:
                    base_ID = typeDict[typeName]
                else:
                    typeList.append(typeName)
                    global numType
                    typeDict[typeName] = numType
                    base_ID = numType
                    numType = numType + 1

                # expand the ID into the range of 4*base_ID, according to whether it has sibling or children.
                # Sibling information is got by the ancestor's children information
                if 'children' in dic.keys():
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
                # recording the N which has non-empty T
                if 'value' in dic.keys():
                    no_empty_set.add(ID)

                line_N.append(ID)
                parent_list.append(parent_counter[i])
                numID.add(ID)

            corpus_N.append(line_N)
            corpus_parent.append(parent_list)
        return corpus_N, corpus_parent


def map_dense_id(data):
    result = list()
    for line_id in data:
        line_new_id = list()
        for i in line_id:
            if i in dicID.keys():
                line_new_id.append(dicID[i])
            else:
                dicID[i] = len(dicID)
                line_new_id.append(dicID[i])
        result.append(line_new_id)
    return result


def save(filename, typeDict, numType, dicID, vocab_size, trainData, testData, trainParent, testParent, empty_set_dense):
    """
    :param filename: Name of destination (pickle file)
    :param typeDict: Dictionary for inferring types to IDs
    :param numType: Number of total types
    :param dicID: Maps sparse IDs to dense IDs
    :param vocab_size: the vocabulary size to restrict the number of words
    :param trainData: Processed training data (mapped to respective IDs)
    :param testData: Processed testing data (mapped to respective IDs)
    :param trainParent: Offsets to parent in training set
    :param testParent: Offsets to parent in testing set
    :param empty_set_dense: Dense set of non-terminals who can't have value
    """
    with open(filename, 'wb') as f:
        save = {
            'typeDict': typeDict,
            'numType': numType,
            'dicID': dicID,
            'vocab_size': vocab_size,
            'trainData': trainData,
            'testData': testData,
            'trainParent': trainParent,
            'testParent': testParent,
            'typeOnlyHasEmptyValue': empty_set_dense,
        }
        pickle.dump(save, f, protocol=2)


if __name__ == '__main__':
    start_time = time.time()
    trainData, trainParent = process(train_filename)
    testData, testParent = process(test_filename)
    trainData = map_dense_id(trainData)
    testData = map_dense_id(testData)
    vocab_size = len(numID)
    assert len(dicID) == vocab_size

    # for print the N which can only has empty T
    assert no_empty_set.issubset(numID)
    empty_set = numID.difference(no_empty_set)
    empty_set_dense = set()
    print(dicID)
    print(vocab_size)
    for i in empty_set:
        empty_set_dense.add(dicID[i])
    print('The N set that can only has empty terminals: ', len(empty_set_dense), empty_set_dense)
    print('The vocaburary:', vocab_size, numID)

    save(target_filename, typeDict, numType, dicID, vocab_size, trainData, testData, trainParent, testParent,
         empty_set_dense)
    print('Finishing generating terminals and takes %.2fs' % (time.time() - start_time))
