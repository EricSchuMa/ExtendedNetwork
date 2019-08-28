import random
import numpy as np


def create_train_dev_split(filename, destination, ratio=0.9, seed=42):
    """ Creates the train dev split from the training data
    :param filename: the source file which gets split
    :param destination: the directory in which to save the splits
    :param ratio: the ratio in which the file gets split
    :param seed: random seed to make split re- producable
    :return:
    """
    with open(filename, encoding='latin-1') as lines:
        data = lines.readlines()
        nr_samples = len(data)
        indices = np.arange(nr_samples)
        random.seed(seed)  # use seed 42 to reproduce train/dev split from thesis
        random.shuffle(indices)

        # split into ratios and write data
        indices_train = indices[:int(ratio*nr_samples)]
        indices_dev = indices[int(ratio*nr_samples):]
        data_train = [data[idx] for idx in indices_train]
        data_dev = [data[idx] for idx in indices_dev]

        # save into files
        filename_train = destination + "python_train.json"
        with open(filename_train, "w") as fout_train:
            fout_train.writelines(data_train)

        filename_dev = destination + "python_dev.json"
        with open(filename_dev, "w") as fout_dev:
            fout_dev.writelines(data_dev)


if __name__ == '__main__':
    sourceFile = "../../data/python100k_train.json"
    destination_dir = "../../data/"
    create_train_dev_split(sourceFile, destination_dir)
