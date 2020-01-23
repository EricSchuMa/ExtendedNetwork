#Utilities for preprocess the data

from six.moves import cPickle as pickle
import copy


def read_N_pickle(filename):
  with open(filename, 'rb') as f:
    print ("Reading data from ", filename)
    save = pickle.load(f)
    train_data = save['trainData']
    test_data = save['testData']
    vocab_size = save['vocab_size']
    print ('the vocab_size is %d' %vocab_size)
    print ('the number of training data is %d' %(len(train_data)))
    print ('the number of test data is %d' %(len(test_data)))
    print ('Finish reading data!!')
    return train_data, test_data, vocab_size

def read_T_pickle(filename):
  with open(filename, 'rb') as f:
    print ("Reading data from ", filename)
    save = pickle.load(f)
    train_data = save['trainData']
    test_data = save['testData']
    vocab_size = save['vocab_size']
    attn_size = save['attn_size']
    print ('the vocab_size is %d' %vocab_size)
    print ('the attn_size is %d' %attn_size)
    print ('the number of training data is %d' %(len(train_data)))
    print ('the number of test data is %d' %(len(test_data)))
    print ('Finish reading data!!')
    return train_data, test_data, vocab_size, attn_size


def save(filename, terminal_dict, terminal_num, vocab_size, sorted_freq_dict):
  with open(filename, 'wb') as f:
    save = {'terminal_dict': terminal_dict,'terminal_num': terminal_num, 'vocab_size': vocab_size, 'sorted_freq_dict': sorted_freq_dict,}
    pickle.dump(save, f)

def change_protocol_for_N(filename):

    f = open(filename, 'rb')
    save = pickle.load(f)
    typeDict = save['typeDict']
    numType = save['numType']
    dicID = save['dicID']
    vocab_size = save['vocab_size']
    trainData = save['trainData']
    testData = save['testData']
    typeOnlyHasEmptyValue = save['typeOnlyHasEmptyValue']
    f.close()

    f = open(filename, 'wb')
    save = {
        'typeDict': typeDict,
        'numType': numType,
        'dicID': dicID,
        'vocab_size': vocab_size,
        'trainData': trainData,
        'testData': testData,
        'typeOnlyHasEmptyValue': typeOnlyHasEmptyValue,
        }
    pickle.dump(save, f, protocol=2)
    f.close()


def change_protocol_for_T(filename):
    f = open(filename, 'rb')
    save = pickle.load(f)
    terminal_dict = save['terminal_dict']
    terminal_num = save['terminal_num']
    vocab_size = save['vocab_size']
    attn_size = save['attn_size']
    trainData = save['trainData']
    testData = save['testData']
    f.close()

    f = open(target_filename, 'wb')
    save = {'terminal_dict': terminal_dict,
            'terminal_num': terminal_num,
            'vocab_size': vocab_size, 
            'attn_size': attn_size,
            'trainData': trainData, 
            'testData': testData,
            }
    pickle.dump(save, f, protocol=2)
    f.close()

# AA - not used
class DataEncoder:
    """
    Encodes input data by unique integers (keys), and stores a mapping keys => original data.
    Maintains the structure of the original data.
    Artur Andrzejak, Jan 2020
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


from recordclass import RecordClass

default_filename_node_facts = 'PY_node_facts_python10k_dev_1k_dict.pickle'

class PredictionData(RecordClass):
    has_terminal: bool = False
    in_dict: bool = False
    in_attn_window: bool = False
    phog_ok: bool = False
    ast_idx: int = None
    node_idx: int = None


class PredictionsContainer():
    """
    A container for facts about ground-truth/predictions for an AST-node.
    Organized as a dict (node location as tuple (file_id, line_id, ast_id)) -> PredictionData)
    Supports picle save/load
    """
    def __init__(self, filename=None):
        # self.key_class = namedtuple('Location', ['file_id', 'line_id', 'node_id'])
        self.data = dict()
        if filename is not None:
            self.from_pickle(filename)

    # def add(self, location, has_terminal, in_dict, in_attn_window, phog_ok, ast_idx):
    def add(self, location: tuple, predictionData: PredictionData):
        location_tuple = tuple(location) if isinstance(location, list) else location
        self.data[location_tuple] = tuple(predictionData)

        # sample = PredictionData(has_terminal, in_dict, in_attn_window, phog_ok, ast_idx)
        # key = PredictionLocation(file_id, line_id, node_id)
        # key = self.key_class(file_id, line_id, node_id)

    def get(self, location_triple: tuple):
        return self.data[location_triple]

    def to_pickle(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.data, f, fix_imports=True)

    def from_pickle(self, filename):
        with open(filename, 'rb') as f:
            loaded = pickle.load(f)
            self.data = loaded.data


def from_pickle(filename):
    with open(filename, 'rb') as f:
        loaded = pickle.load(f)
        return loaded


if __name__ == '__main__':
    
    # train_filename = '../json_data/small_programs_training.json'
    # test_filename = '../json_data/small_programs_eval.json'
    # N_pickle_filename = '../pickle_data/JS_non_terminal.pickle'
    # T_pickle_filename = '../pickle_data/JS_terminal_1k.pickle'
    filename = '../pickle_data/PY_non_terminal.pickle'
    read_N_pickle(filename)
    # filename = '../pickle_data/JS_terminal_1k_whole.pickle'
    # change_protocol_for_T(filename, target_filename)


    # N_train_data, N_test_data, N_vocab_size = read_N_pickle(N_pickle_filename)
    # T_train_data, T_test_data, T_vocab_size, attn_size = read_T_pickle(T_pickle_filename)
    # print(len(N_train_data), len(T_train_data))

