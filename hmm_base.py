

import numpy as np
from tqdm import tqdm
from collections import defaultdict



def train_test_split(pth, test_size=0.2, random_state=None):

    with open(pth, 'r') as f:
        all_lines = f.readlines()

    # Remove \n as separate lines
    # Remove \t,\n at both ends of string
    data = []
    for line in all_lines:
        if line != '\n':
            data.append(line.strip())

    R = np.random.RandomState(seed=random_state)

    inds = R.permutation(len(data))
    test_inds = inds[:int(test_size*len(data))]
    train_inds = inds[int(test_size*len(data)):]

    train, test = [], []
    for ind in train_inds:
        train.append(data[ind])
    for ind in test_inds:
        test.append(data[ind])

    return train, test

    





class BaseHMM:
    """
    A base class for implementing Hidden Markov Models
    with different stratergies for handling unknown words
    for POS tagging.
    """

    def __init__(self, train_data):
        self.train_data = train_data

    def extract_vocab_and_tags(self):
        """ 
        Extracts the vocalubary and all the possible 
        tags from the training data.
        """

        self.vocab = defaultdict(int)
        self.tags = defaultdict(int)

        for line in self.train_data:
            for word_tag in line.lower().split():
                try:
                    word, tag = word_tag.split('/')
                    self.vocab[word] += 1
                    self.tags[tag] += 1
                except:
                    # Leave the ambiguous ones
                    pass
        
        # Calculate the probablity of each tag
        self.tag_prob = {}
        tot = 0
        for tag in self.tags:
            tot += self.tags[tag]

        self.tag_prob = {tag: self.tags[tag]/tot for tag in self.tags}

        # Add start of sentence tag to tags list
        self.tags['<s>'] = len(self.train_data)
        self.tag_prob['<s>'] = 0

    def handle_unknown(self):
        """
        Stratergy for handling unknown words.
        This class is the main method differentiating
        the sub-classes.
        """
        pass

    def create_embeddings(self):
        """
        Create embeddings by mapping vocabulary
        and tags to numbers.
        """
        self.vocab_map = {}
        self.tag_map = {}
        self.inv_tag_map = {}

        for i, word in enumerate(self.vocab):
            self.vocab_map[word] = i
        for i, tag in enumerate(self.tags):
            self.tag_map[tag] = i
            self.inv_tag_map[i] = tag

        self.vocab_size = len(self.vocab)
        self.num_tags = len(self.tags)

    
    def calculate_probabilities(self):
        """
        Calculates the emmision probablility and 
        transition probability from the train data.
        """

        self.emmision_prob = np.zeros((self.vocab_size, self.num_tags))
        self.transition_prob = np.zeros((self.num_tags, self.num_tags))

        for line in self.train_data:
            prev_tag = '<s>'
            for word_tag in line.lower().split():
                try:
                    word, tag = word_tag.split('/')
                    self.transition_prob[self.tag_map[tag]][self.tag_map[prev_tag]] += 1
                    if word in self.vocab:
                        self.emmision_prob[self.vocab_map[word]][self.tag_map[tag]] += 1
                    else:
                        self.emmision_prob[self.vocab_map['<unk>']][self.tag_map[tag]] += 1
                    prev_tag = tag

                except:
                    pass

        tag_count = np.zeros((self.num_tags,))
        for tag in self.tags:
            tag_count[self.tag_map[tag]] = self.tags[tag]

        # Now divide both the self.emmision_prob and the 
        # self.transition_prob with the tag_counts to 
        # convert the numbers into probabilities
        self.emmision_prob = self.emmision_prob/tag_count
        self.transition_prob = self.transition_prob/tag_count


    def train(self):
        self.extract_vocab_and_tags()
        self.handle_unknown()
        self.create_embeddings()
        self.calculate_probabilities()



    def access_model(self, test, verbose=True):
        """
        Accesses the performance of the model 
        on test data.
        """

        test_sents, test_tags = [], []

        for line in test:
            sents, tags = [], []
            for word_tag in line.lower().split():
                tag = word_tag.split('/')[-1]
                word = "/".join(word_tag.split('/')[:-1])
                sents.append(word)
                tags.append(tag)
            test_sents.append(" ".join(sents))
            test_tags.append(tags)

        pred_sent_tags = []
        for sent in tqdm(test_sents, disable=not verbose):
            word_tag = self.predict(sent)
            pred_sent_tags.append(word_tag)

        correct, total = 0, 0
        for i in range(len(pred_sent_tags)):
            for j in range(len(test_tags[i])):
                if pred_sent_tags[i][j][1] == test_tags[i][j]:
                    correct += 1
                total += 1

        accu = (correct/total)*100
        return accu

                
            


    def predict(self,sent):
        """
        Given a sentence returns the tags predicted
        by the model. The sentence is expected to
        be a a string without any tag information.
        """
        pass

