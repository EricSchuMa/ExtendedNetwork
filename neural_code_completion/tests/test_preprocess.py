import unittest
import numpy as np

from six.moves import cPickle as pickle
from preprocess_code.get_terminal_extended import save, process
from preprocess_code.get_terminal_whole import restore_terminal_dict
import models.reader_pointer_extended as reader


class TestGetTerminalExtended(unittest.TestCase):
    def setUp(self):
        self.terminal_whole = '../pickle_data/PY_terminal_1k_whole.pickle'
        self.terminal_dict_filename = '../pickle_data/terminal_dict_1k_PY.pickle'
        self.train_filename = '../../data/python100k_train.json'
        self.trainHOG_filename = '../../data/phog_pred_100k_train.json'
        self.test_filename = '../../data/python50k_eval.json'
        self.testHOG_filename = '../../data/phog_pred_50k_eval.json'
        self.target_filename = './test_get_terminal_extended.pickle'
        self.terminal_dict, self.terminal_num, self.vocab_size = restore_terminal_dict(self.terminal_dict_filename)
        self.attn_size = 50
        self.trainData = process(self.train_filename, self.trainHOG_filename,
                                 self.terminal_dict, self.vocab_size, attn_size=self.attn_size,
                                 verbose=False, is_train=True)

        self.testData = process(self.test_filename, self.testHOG_filename, self.terminal_dict,
                                self.vocab_size, attn_size=self.attn_size,
                                verbose=False, is_train=False)

        save(self.target_filename, self.terminal_dict, self.terminal_num, self.vocab_size,
             self.attn_size, self.trainData, self.testData)

    def test_data(self):
        with open(self.target_filename, 'rb') as target:
            target_data = pickle.load(target)

        with open(self.terminal_whole, 'rb') as whole:
            original_data = pickle.load(whole)

        self.assertEqual(target_data["terminal_dict"], original_data["terminal_dict"])
        self.assertEqual(target_data["terminal_num"], original_data["terminal_num"])
        self.assertEqual(target_data["vocab_size"], original_data["vocab_size"])
        self.assertEqual(target_data["attn_size"], original_data["attn_size"])

        trainDataTarget = target_data["trainData"]
        trainDataOriginal = original_data["trainData"]
        testDataTarget = target_data["testData"]
        testDataOriginal = original_data["testData"]

        attn_size = 50
        unk = original_data["vocab_size"]
        hog = target_data["vocab_size"] + 1

        train_test_data_target = {"train": trainDataTarget, "test": testDataTarget}
        train_test_data_original = {"train": trainDataOriginal, "test": testDataOriginal}

        for key in ["train", "test"]:
            cnt_hog_nodes = 0
            cnt_total_nodes = 0
            for lineTarget, lineOriginal in zip(train_test_data_target[key], train_test_data_original[key]):
                self.assertEqual(len(lineTarget), len(lineOriginal))
                idx = np.not_equal(lineOriginal, lineTarget)  # indices where lines differ
                changes = zip(np.array(lineOriginal)[idx], np.array(lineTarget)[idx])  # original and changed values

                # only unk ID should be replaced by construction of terminal_extended corpus
                # all location indices should be shifted by 1
                if len(idx) > 0:
                    for original_val, changed_val in changes:
                        if original_val == unk:
                            self.assertTrue(changed_val == unk or
                                            changed_val == hog)
                        else:
                            self.assertTrue(changed_val == original_val + 1)
                cnt_hog_nodes += len(np.where(np.array(lineTarget) == hog)[0])
                cnt_total_nodes += len(lineTarget)
            # Percentage of hog nodes should be ~3.44% for train- and ~1.56% for test-set
            print("Percentage of hog nodes for %s-set %.2f" % (key, cnt_hog_nodes/cnt_total_nodes * 100))


if __name__ == '__main__':
    unittest.main()
