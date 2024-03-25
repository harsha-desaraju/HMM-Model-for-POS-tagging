

import numpy as np
from tqdm import tqdm
from collections import defaultdict
from hmm_base import train_test_split, BaseHMM



class TagProbUnkHMM(BaseHMM):
    """
        This class uses the probability of occurance 
        of a tag as emmision probability when an 
        unknown word is encountered.
    """

    def predict(self,sent):
        
        tokens = sent.lower().split()
        
        probs = np.zeros((len(tokens), self.num_tags))
        paths = np.full((len(tokens), self.num_tags), -1)

        for col in range(len(probs[0])):
            if tokens[0] in self.vocab:
                probs[0][col] = self.transition_prob[col][self.tag_map['<s>']] \
                * self.emmision_prob[self.vocab_map[tokens[0]]][col]
            else:
                probs[0][col] = self.transition_prob[col][self.tag_map['<s>']] \
                * self.tag_prob[self.inv_tag_map[col]]

        for row in range(1, len(probs)):
            for tag_id in range(self.num_tags):
                if tokens[row] in self.vocab:
                    em_p = self.emmision_prob[self.vocab_map[tokens[row]]][tag_id]
                else:
                    em_p = self.tag_prob[self.inv_tag_map[tag_id]]
                if em_p != 0:
                    temp_probs = []
                    for prev_tag_id in range(self.num_tags):
                        prob = probs[row-1][prev_tag_id] * self.transition_prob[tag_id][prev_tag_id]*em_p
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

        return pred_tags
        












if __name__ == '__main__':

    train, test = train_test_split('brown.txt', test_size=0.01, random_state=0)
    print(f"The size of test set is {len(test)}")

    hmm_model = TagProbUnkHMM(train_data=train)
    hmm_model.train()
    tags = hmm_model.predict("The horse raced past the barn fell")
    print(tags)

    accu = hmm_model.access_model(test, verbose=True)
    print(f"The accuracy of the model is {accu}")

