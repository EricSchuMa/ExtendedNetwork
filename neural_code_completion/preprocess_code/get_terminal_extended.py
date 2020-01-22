# According to the terminal_dict you choose (i.e. 5k, 10k, 50k), parse the json file and turn them into ids
# that are stored in pickle file
# Output just one vector for terminal, the upper part is the word id while the lower part is the location
# 0108 revise the Empty into EmptY, normal to NormaL
# Here attn_size matters

from six.moves import cPickle as pickle
from collections import deque

import json
import time

from neural_code_completion.preprocess_code.get_terminal_original import restore_terminal_dict
#from preprocess_code.get_terminal_whole import restore_terminal_dict

from utils import PredictionData, PredictionsContainer

terminal_dict_filename = '../pickle_data/terminal_dict_1k_PY_train_dev.pickle'
train_filename = '../../data/python90k_train.json'
trainHOG_filename = '../../data/phog-json/phog_train.json'
test_filename = '../../data/python10k_dev.json'
testHOG_filename = '../../data/phog-json/phog_dev.json'
target_filename = '../pickle_data/PY_terminal_1k_extended_dev.pickle'



def process(filename, hog_filename, terminal_dict, unk_id, attn_size, verbose=False, is_train=False):
    """ creates the terminal corpus which for every node contains the value: either emptY if terminal is empty,
    terminal_dict[value], location in atn window (if word is not in terminal_dict and is not empty) or unk-keyword
    :param filename: file for saving terminal corpus
    :param hog_filename: file which includes predictions from hog
    :param terminal_dict: terminal_dict from get_terminal_dict.py
    :param unk_id: first ID that is not in terminal_dict
    :param attn_size: size of attention window
    :param verbose: boolean; if true show statistics
    :param is_train: whether data is training or not
    :return: terminal_corpus
    """
    with open(filename, encoding='latin-1') as lines:
        with open(hog_filename, encoding='latin-1') as lines_hog:
            with open('fail.txt', 'a') as failout:
                print('Start procesing %s !!!' % filename)
                terminal_corpus = list()
                attn_que = deque(maxlen=attn_size)
                attn_success_total = 0
                attn_fail_total = 0
                hog_success_total = 0
                hog_fail_total = 0
                length_total = 0
                hog = unk_id + 1

                node_facts_container = PredictionsContainer()
                ast_index = -1

                for line, line_hog in zip(lines, lines_hog):
                    ast_index += 1
                    if ast_index % 1000 == 0:
                        print('Processing ast/line:', ast_index)
                    data = json.loads(line)
                    data_hog = json.loads(line_hog)
                    if len(data) < 3e4:
                        terminal_line = list()
                        attn_que.clear()  # have a new queue for each file
                        attn_success_cnt = 0
                        attn_fail_cnt = 0
                        hog_success_cnt = 0
                        hog_fail_cnt = 0

                        for node_idx, (dic, dic_hog) in enumerate(zip(data, data_hog)):  # JS data[:-1] or PY data

                            node_truths = PredictionData()
                            node_truths.has_terminal = 'value' in dic
                            node_truths.ast_idx = ast_index
                            node_truths.node_idx = node_idx

                            if node_truths.has_terminal:
                                dic_value = dic['value']
                                node_truths.in_dict = dic_value in terminal_dict
                                node_truths.in_attn_window = dic_value in attn_que
                                node_truths.phog_ok = (dic_value == dic_hog["value"])

                                if node_truths.in_dict:
                                    handle_node_in_terminal_dict(attn_que, dic_value, terminal_dict, terminal_line)
                                else:
                                    if node_truths.in_attn_window:
                                        handle_node_in_attn_que(attn_que, dic_value, terminal_line, unk_id)
                                        attn_success_cnt += 1
                                    else:
                                        attn_fail_cnt += 1
                                        # pointer network cannot predict, try phog now
                                        if node_truths.phog_ok:
                                            # obviously phog has predicted correctly (ask Max)
                                            hog_success_cnt += 1
                                            terminal_line.append(hog)
                                        else:
                                            # pointer cannot predict, phog failed
                                            terminal_line.append(unk_id)
                                            failout.write('Prediction %s, GroundTruth %s \n'
                                                          % (dic_value, dic_hog["value"]))
                                            hog_fail_cnt += 1
                                    attn_que.append(dic_value)
                            else:
                                terminal_line.append(terminal_dict['EmptY'])
                                attn_que.append('EmptY')
                            # end of inner loop over ast nodes
                            if 'location' in dic:
                                node_facts_container.add(dic['location'], node_truths)

                        attn_fail_total += attn_fail_cnt
                        attn_total = attn_success_total + attn_fail_total
                        hog_success_total += hog_success_cnt
                        hog_fail_total += hog_fail_cnt
                        length_total += len(data)

                        if verbose and ast_index % 1000 == 0:
                            print('\nUntil line %d: attn_success_total: %d, attn_fail_total: %d, success/attn_total: %.4f,'
                                  ' length_total: %d, attn_success percentage: %.4f, total unk percentage: %.4f\n' %
                                  (ast_index, attn_success_total, attn_fail_total,
                                   float(attn_success_total)/attn_total, length_total,
                                   float(attn_success_total)/length_total,
                                   float(attn_total)/length_total))
                write_output_file(attn_fail_total, attn_success_total, attn_total, hog_fail_total, hog_success_total,
                                  length_total)
            return terminal_corpus, node_facts_container


def write_output_file(attn_fail_total, attn_success_total, attn_total, hog_fail_total, hog_success_total, length_total):
    with open('output.txt', 'a') as fout:
        fout.write('New Experiment: terminal dict = %s, hog files are %s , %s \n' %
                   (terminal_dict_filename, trainHOG_filename, testHOG_filename))
        fout.write('Statistics: attn_success_total: %d, attn_fail_total: %d, success/fail: %.4f,'
                   ' length_total: %d, attn_success percentage: %.4f, total unk percentage: %.4f\n' %
                   (attn_success_total, attn_fail_total,
                    float(attn_success_total) / attn_fail_total, length_total,
                    float(attn_success_total) / length_total,
                    float(attn_total) / length_total))
        fout.write('\n Statistics: hog_success_total: %d, hog_fail_total: %d, success/fail: %.4f,'
                   ' length_total: %d, hog_success percentage: %.4f, total unk percentage: %.4f\n \n' %
                   (hog_success_total, hog_fail_total,
                    float(hog_success_total) / hog_fail_total, length_total,
                    float(hog_success_total) / length_total,
                    float(attn_total) / length_total))


def handle_node_in_attn_que(attn_que, dic_value, terminal_line, unk_id):
    # token is in attention window, but _not_ in seq model dict,
    location_index = [len(attn_que) - ind for ind, x
                      in enumerate(attn_que) if x == dic_value][-1]
    location_id = unk_id + 2 + location_index  # [unk, hog_id, eof, loc_idx]
    terminal_line.append(location_id)


def handle_node_in_terminal_dict(attn_que, dic_value, terminal_dict, terminal_line):
    # Token is in the sequence model dictionary
    terminal_line.append(terminal_dict[dic_value])
    attn_que.append('NormaL')


def save(filename, terminal_dict, terminal_num, vocab_size, attn_size, trainData, testData):
    with open(filename, 'wb') as f:
        save = {'terminal_dict': terminal_dict,
                'terminal_num': terminal_num,
                'vocab_size': vocab_size,
                'attn_size': attn_size,
                'trainData': trainData,
                'testData': testData,
                }
        pickle.dump(save, f, protocol=2)


if __name__ == '__main__':
    import gc
    gc.disable()
    start_time = time.time()
    attn_size = 50
    SKIP_TRAIN_DATA = True
    terminal_dict, terminal_num, vocab_size = restore_terminal_dict(terminal_dict_filename)
    if SKIP_TRAIN_DATA:
        target_filename_debug = '../pickle_data/PY_terminal_1k_extended_all_predictions.pickle'
        testData, node_facts_container = process(test_filename, testHOG_filename, terminal_dict, vocab_size, attn_size=attn_size,
                                                 verbose=False, is_train=False)
        save(target_filename, terminal_dict, terminal_num, vocab_size, attn_size, testData, testData)
    else:
        trainData, _ = process(train_filename, trainHOG_filename, terminal_dict, vocab_size, attn_size=attn_size,
                                                  verbose=False, is_train=True)
        testData, node_facts_container = process(test_filename, testHOG_filename, terminal_dict, vocab_size, attn_size=attn_size,
                                                 verbose=False, is_train=False)
        save(target_filename, terminal_dict, terminal_num, vocab_size, attn_size, trainData, testData)

    from utils import default_filename_node_facts
    node_facts_container.to_pickle(default_filename_node_facts)

    print('Finishing generating terminals and takes %.2f' % (time.time() - start_time))