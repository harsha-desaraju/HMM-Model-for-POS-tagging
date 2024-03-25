
import numpy as np
from hmm_base import BaseHMM, train_test_split




class SingleFreqUnkHMM(BaseHMM):
    """
    This class treats all the words which
    occur once in the training data as 
    unknown words and calculates the probabilities
    for <unk> tag with that.
    """

    def handle_unknown(self):
        # Replace single words in vocab
        # with <unk> tag
        vocab_words = list(self.vocab.values())
        num_unk = 0
        for word in vocab_words:
            if self.vocab[word] == 1:
                num_unk +=1 
                self.vocab.pop(word)
        self.vocab['<unk>'] = num_unk



    def predict(self, sent):
        tokens = sent.lower().split()
        tokens = [word if word in self.vocab else '<unk>' for word in tokens]
        
        probs = np.zeros((len(tokens), self.num_tags))
        paths = np.full((len(tokens), self.num_tags), -1)

        for col in range(len(probs[0])):
            tag = self.inv_tag_map[col]
            probs[0][col] = self.transition_prob[col][self.tag_map['<s>']] \
                * self.emmision_prob[self.vocab_map[tokens[0]]][col]

        for row in range(1, len(probs)):
            for tag_id in range(self.num_tags):
                em_p = self.emmision_prob[self.vocab_map[tokens[row]]][tag_id]
                if em_p != 0:
                    temp_probs = []
                    for prev_tag_id in range(self.num_tags):
                        prob = probs[row-1][prev_tag_id] * self.transition_prob[tag_id][prev_tag_id] \
                        * self.emmision_prob[self.vocab_map[tokens[row]]][tag_id]
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

    train, test = train_test_split('brown.txt', test_size=0.01, random_state=0)
    print(f"The size of test set is {len(test)}")

    hmm_model = SingleFreqUnkHMM(train_data=train)
    hmm_model.train()
    tags = hmm_model.predict("They also output confusion matrix and accuracy on the terminal .")
    print(tags)

    accu = hmm_model.access_model(test)
    print(f"The accuracy of the model is {accu}")

