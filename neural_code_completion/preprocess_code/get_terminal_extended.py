# According to the terminal_dict you choose (i.e. 5k, 10k, 50k), parse the json file and turn them into ids
# that are stored in pickle file
# Output just one vector for terminal, the upper part is the word id while the lower part is the location
# 0108 revise the Empty into EmptY, normal to NormaL
# Here attn_size matters

from six.moves import cPickle as pickle
from collections import deque

import json
import time

from .get_terminal_whole import restore_terminal_dict

terminal_dict_filename = '../pickle_data/terminal_dict_1k_PY.pickle'
train_filename = '../../data/python100k_train.json'
trainHOG_filename = '../../data/phog_pred_100k_train.json'
test_filename = '../../data/python50k_eval.json'
testHOG_filename = '../../data/phog_pred_50k_eval.json'
target_filename = '../pickle_data/PY_terminal_1k_extended.pickle'


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
                line_index = 0
                hog_id = unk_id + attn_size + 2
                for line, line_hog in zip(lines, lines_hog):
                    line_index += 1
                    # if is_train and line_index == 11:
                    #   continue
                    if line_index % 1000 == 0:
                        print('Processing line:', line_index)
                    data = json.loads(line)
                    data_hog = json.loads(line_hog)
                    if len(data) < 3e4:
                        terminal_line = list()
                        attn_que.clear()  # have a new queue for each file
                        attn_success_cnt = 0
                        attn_fail_cnt = 0
                        hog_success_cnt = 0
                        hog_fail_cnt = 0
                        for i, (dic, dic_hog) in enumerate(zip(data, data_hog)):  # JS data[:-1] or PY data
                            if 'value' in dic.keys():
                                dic_value = dic['value']
                                if dic_value in terminal_dict.keys():  # take long time!!!
                                    terminal_line.append(terminal_dict[dic_value])
                                    attn_que.append('NormaL')
                                else:
                                    if dic_value in attn_que:
                                        location_index = [len(attn_que)-ind for ind, x
                                                          in enumerate(attn_que) if x == dic_value][-1]
                                        location_id = unk_id + 1 + location_index
                                        terminal_line.append(location_id)
                                        attn_success_cnt += 1
                                        if dic_value == dic_hog["value"]:
                                            hog_success_cnt += 1
                                        else:
                                            hog_fail_cnt += 1

                                    else:
                                        attn_fail_cnt += 1
                                        if dic_value == dic_hog["value"]:
                                            hog_success_cnt += 1
                                            terminal_line.append(hog_id)
                                        else:
                                            terminal_line.append(unk_id)
                                            failout.write('Prediction %s, GroundTruth %s \n'
                                                          % (dic_value, dic_hog["value"]))
                                            hog_fail_cnt += 1
                                    attn_que.append(dic_value)
                            else:
                                terminal_line.append(terminal_dict['EmptY'])
                                attn_que.append('EmptY')
                        terminal_corpus.append(terminal_line)
                        attn_success_total += attn_success_cnt
                        attn_fail_total += attn_fail_cnt
                        attn_total = attn_success_total + attn_fail_total
                        hog_success_total += hog_success_cnt
                        hog_fail_total += hog_fail_cnt
                        length_total += len(data)
                        if verbose and line_index % 1000 == 0:
                            print('\nUntil line %d: attn_success_total: %d, attn_fail_total: %d, success/attn_total: %.4f,'
                                  ' length_total: %d, attn_success percentage: %.4f, total unk percentage: %.4f\n' %
                                  (line_index, attn_success_total, attn_fail_total,
                                   float(attn_success_total)/attn_total, length_total,
                                   float(attn_success_total)/length_total,
                                   float(attn_total)/length_total))
                with open('output.txt', 'a') as fout:
                    fout.write('Statistics: attn_success_total: %d, attn_fail_total: %d, success/fail: %.4f,'
                               ' length_total: %d, attn_success percentage: %.4f, total unk percentage: %.4f\n' %
                               (attn_success_total, attn_fail_total,
                                float(attn_success_total)/attn_fail_total, length_total,
                                float(attn_success_total)/length_total,
                                float(attn_total)/length_total))
                    fout.write('Statistics: hog_success_total: %d, hog_fail_total: %d, success/fail: %.4f,'
                               ' length_total: %d, hog_success percentage: %.4f, total unk percentage: %.4f\n' %
                               (hog_success_total, hog_fail_total,
                                float(hog_success_total) / hog_fail_total, length_total,
                                float(hog_success_total) / length_total,
                                float(attn_total) / length_total))
            return terminal_corpus


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
    start_time = time.time()
    attn_size = 50
    terminal_dict, terminal_num, vocab_size = restore_terminal_dict(terminal_dict_filename)
    trainData = process(train_filename, trainHOG_filename, terminal_dict, vocab_size, attn_size=attn_size,
                        verbose=False, is_train=True)
    testData = process(test_filename, testHOG_filename, terminal_dict, vocab_size, attn_size=attn_size,
                       verbose=False, is_train=False)
    save(target_filename, terminal_dict, terminal_num, vocab_size, attn_size, trainData, testData)
    print('Finishing generating terminals and takes %.2f' % (time.time() - start_time))