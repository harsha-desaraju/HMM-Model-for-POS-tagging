

import numpy as np
from tqdm import tqdm
from collections import defaultdict

def train_test_split(pth, test_size=0.2):

    with open(pth, 'r') as f:
        all_lines = f.readlines()

    # Remove \n as separate lines
    # Remove \t,\n at both ends of string
    data = []
    for line in all_lines:
        if line != '\n':
            data.append(line.strip())

    inds = np.random.permutation(len(data))
    test_inds = inds[:int(test_size*len(data))]
    train_inds = inds[int(test_size*len(data)):]

    train, test = [], []
    for ind in train_inds:
        train.append(data[ind])
    for ind in test_inds:
        test.append(data[ind])

    return train, test




class HMM:
    """
        A class implementing the Hidden Markov Model
        for POS tagging.
    """
    def __init__(self, train_data, model='bigram'):
        self.train_data = train_data
        self.model = model

    def calculate_vocab_and_tags(self):
        """ 
            Calculates the vocalubary and all the possible 
            tags from the training data. Replace all the 
            words with a frequency of one as unknown words
            with token <unk>.
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

        # Replace single words in vocab
        vocab_words = list(self.vocab.values())
        num_unk = 0
        for word in vocab_words:
            if self.vocab[word] == 1:
                num_unk +=1 
                self.vocab.pop(word)
        self.vocab['<unk>'] = num_unk
        
        # Add start of sentence tag to tags list
        self.tags['<s>'] = len(self.train_data)
        
        # The tag map and the vocab map are
        # for mapping tags and vocabulary to numbers
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
        print(f"No. of tags: {self.num_tags}")


    def calculate_stats(self):
        """
            Calculates the emmision probablility and 
            transition probability from tha data
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


    def access_model(self, test):
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
        for sent in tqdm(test_sents):
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
            Given a sentence return the tags predicted
            by the model. The sentence is expected to
            be a a string without any tag information.
        """
        tokens = sent.lower().split()
        tokens = [word if word in self.vocab else '<unk>' for word in tokens]
        
        probs = np.zeros((len(tokens), self.num_tags))
        paths = np.full((len(tokens), self.num_tags), -1)

        for col in range(len(probs[0])):
            tag = self.inv_tag_map[col]
            probs[0][col] = self.transition_prob[col][self.tag_map['<s>']] * self.emmision_prob[self.vocab_map[tokens[0]]][col]

        for row in range(1, len(probs)):
            for tag_id in range(self.num_tags):
                em_p = self.emmision_prob[self.vocab_map[tokens[row]]][tag_id]
                if em_p != 0:
                    temp_probs = []
                    for prev_tag_id in range(self.num_tags):
                        prob = probs[row-1][prev_tag_id] * self.transition_prob[tag_id][prev_tag_id] * self.emmision_prob[self.vocab_map[tokens[row]]][tag_id]
                        temp_probs.append(prob)
                    probs[row][tag_id] = max(temp_probs)
                    paths[row][tag_id] = temp_probs.index(max(temp_probs))
                else:
                    probs[row][tag_id] = 0
                    paths[row][tag_id] = 0
                

        pred_tags = []
        ind = np.argmax(probs[-1])
        for i in range(len(tokens)-1, -1, -1):
            pred_tags.append([tokens[i],self.inv_tag_map[ind]])
            ind = paths[i][ind]

        pred_tags = pred_tags[::-1]
        for i in range(len(tokens)):
            pred_tags[i][0] = sent.lower().split()[i]

        return pred_tags
        












if __name__ == '__main__':

    train, test = train_test_split('brown.txt', test_size=0.01)
    print(f"The size of test set is {len(test)}")

    hmm_model = HMM(train_data=train)
    hmm_model.calculate_vocab_and_tags()
    # tags = hmm_model.tags
    # tags = tags.items()
    # tags = sorted(tags,key = lambda x:x[1], reverse=True)
    # for tup in tags:
    #     print(tup[0], tup[1])
    hmm_model.calculate_stats()
    emmi_prob = hmm_model.emmision_prob
    trans_prob = hmm_model.transition_prob
    tags = hmm_model.predict("They also output confusion matrix and accuracy on the terminal .")
    print(tags)

    accu = hmm_model.access_model(test)
    print(f"The accuracy of the model is {accu}")

