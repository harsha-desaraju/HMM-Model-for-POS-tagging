import numpy as np
from tqdm import tqdm
from typing import List
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field


class Word(BaseModel):
    text: str
    count: int
    probability: float = Field(..., ge=0, lt=1)
    index: int = None


class Tag(BaseModel):
    text: str
    count: int
    probability: float = Field(..., ge=0, lt=1)
    index: int = None


class BaseHMM(ABC):
    """
    A base class for implementing Hidden Markov Models
    with different strategies for handling unknown words
    for POS tagging.
    """

    def __init__(self):
        self.vocab = {}
        self.tags = {}
        self.emission_prob = None
        self.transition_prob = None
        self.num_tags = None

    @staticmethod
    def get_word_tag(word_tag: str):
        lst = word_tag.split("/")
        word, tag = "/".join(lst[:-1]), lst[-1]
        return word, tag

    def extract_vocab_and_tags(self, train_data: List[str]):
        """
        Extracts the vocabulary and all the possible
        tags from the training data.
        """
        num_words, num_sentences = 0, 0
        for line in train_data:
            line = line.strip().lower()
            if line:
                word_tags = line.split()
                for word_tag in word_tags:
                    word, tag = self.get_word_tag(word_tag)

                    if word in self.vocab:
                        self.vocab[word].count += 1
                    else:
                        self.vocab[word] = Word(text=word, count=1, probability=0)

                    if tag in self.tags:
                        self.tags[tag].count += 1
                    else:
                        self.tags[tag] = Tag(text=tag, count=1, probability=0)
                    num_words += 1
                num_sentences += 1

        # Calculate the probability of each tag
        for tag in self.tags:
            self.tags[tag].probability = self.tags[tag].count / num_words

        # Add start of sentence tag to tags list
        # ----------- Think if the probability of start should be 0 ????????????
        self.tags["<s>"] = Tag(text="<s>", count=num_sentences, probability=0)

        self.num_tags = len(self.tags)

    @abstractmethod
    def handle_unknown(self):
        """
        Strategy for handling unknown words.
        This class is the main method differentiating
        the subclasses.
        """
        pass

    def _generate_index_map(self):
        """
        Create indices by mapping vocabulary
        and tags to numbers.
        """
        for i, word in enumerate(self.vocab):
            self.vocab[word].index = i
        for i, tag in enumerate(self.tags):
            self.tags[tag].index = i

    def calculate_probabilities(self, train_data: List[str]):
        """
        Calculates the emission probability and
        transition probability from the train data.
        """

        self.emission_prob = np.zeros((len(self.vocab), self.num_tags))
        self.transition_prob = np.zeros((self.num_tags, self.num_tags))

        for line in train_data:
            prev_tag = "<s>"
            line = line.strip().lower()
            if line:
                word_tags = line.split()
                for word_tag in word_tags:
                    word, tag = self.get_word_tag(word_tag)
                    self.transition_prob[self.tags[tag].index][
                        self.tags[prev_tag].index
                    ] += 1
                    if word in self.vocab:
                        self.emission_prob[self.vocab[word].index][
                            self.tags[tag].index
                        ] += 1
                    else:
                        self.emission_prob[self.vocab["<unk>"].index][
                            self.tags[tag].index
                        ] += 1
                    prev_tag = tag

        tag_count = np.array([tag.count for tag in self.tags.values()])

        # Now divide both the self.emission_prob and the
        # self.transition_prob with the tag_counts to
        # convert the numbers into probabilities
        self.emission_prob = self.emission_prob / tag_count
        self.transition_prob = self.transition_prob / tag_count

    def train(self, train_data: List[str]):
        self.extract_vocab_and_tags(train_data)
        self.handle_unknown()
        self._generate_index_map()
        self.calculate_probabilities(train_data)

    def access_model(self, test_data: List[str], verbose=True):
        """
        Accesses the performance of the model
        on test data.
        """

        test_sentences, test_tags = [], []

        for line in test_data:
            line = line.strip().lower()
            if line:
                words, tags = [], []
                for word_tag in line.split():
                    word, tag = self.get_word_tag(word_tag)
                    words.append(word)
                    tags.append(tag)
                test_sentences.append(" ".join(words))
                test_tags.append(tags)

        pred_sent_tags = []
        for sent in tqdm(test_sentences, disable=not verbose):
            word_tag = self.predict(sent)
            pred_sent_tags.append(word_tag)

        correct, total = 0, 0
        for i in range(len(pred_sent_tags)):
            for j in range(len(test_tags[i])):
                if pred_sent_tags[i][j][1] == test_tags[i][j]:
                    correct += 1
                total += 1

        accuracy = (correct / total) * 100
        return accuracy

    @abstractmethod
    def predict(self, sent):
        """
        Given a sentence returns the tags predicted
        by the model. The sentence is expected to
        be a string without any tag information.
        """
        pass
