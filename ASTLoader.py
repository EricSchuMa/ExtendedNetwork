import json


class ASTLoader(object):
    def __init__(self, train_source, test_source, num_samples):
        self._train_source = train_source
        self._num_samples = num_samples
        self.types_train = None

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
