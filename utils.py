import numpy as np
from typing import Union

def train_test_split(file_path: str, test_size: Union[float, int] = 0.2, save_files: bool = True):
    """Split the corpus into train and test files"""

    corpus = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            if line != '\n':
                corpus.append(line)

    assert test_size < len(corpus), f"The number of lines in test set should be less than the total number of lines. Choose a number less than {len(corpus)}"

    if test_size < 1:
        num_test = int(len(corpus)*test_size)
    else:
        num_test = test_size

    indices = np.random.permutation(len(corpus))
    test_inds = indices[:num_test]
    train_inds = indices[num_test:]

    train_corpus = [corpus[i] for i in train_inds]
    test_corpus = [corpus[i] for i in test_inds]

    if save_files:
        with open("./data/train.txt", 'w') as f:
             for line in train_corpus:
                 f.write(line+'\n')


        with open("./data/test.txt", 'w') as f:
            for line in test_corpus:
                f.write(line+'\n')

    return train_corpus, test_corpus


if __name__ == '__main__':

    train_test_split("./data/brown.txt")