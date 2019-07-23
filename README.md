# NLP 2018/2019 ([more][1])

## Homework 1

Chinese word segmentation is an instance of word segmentation, which is the task of splitting a string of written natural language into its component words. The majority of languages relies on a space character to delimit each word in written texts, but there are languages like Chinese, Japanese, Thai, Lao, and Vietnamese which either delimit sentences, phrases, or syllables instead of words, making the task non-trivial.

In order to tackle such non-triviality, a variety of neural network approaches has been used, like the one presented in **State-of-the-art Chinese Word Segmentation with Bi-LSTMs**, which is the basis for this homework assignment. Instead of a sequence to sequence task, for this homework word segmentation has been reduced to sequence tagging, making use of the BIES format to tag each character of a given sequence.

The authors describe a quite simple model: based on 1-grams and 2-grams, using pretrained word embeddings, their model is composed of two layers: a Bi-LSTM and a dense one.

The concatenation of the embeddings of 1-grams and 2-grams is fed into the Bi-LSTM layer, and its output into the dense layer to obtain a probability distribution over the BIES tags for each character in the sequence.

[Continue reading][2]

[1]: http://naviglinlp.blogspot.com/
[2]: ./hw1_report_anonymous.pdf
