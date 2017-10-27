# LSTM for sequence classification in the IMDB dataset
from multiprocessing import *
import pandas as pd
import os
import numpy as np

MAX_NB_WORDS = 20000
MAX_SEQUENCE_LENGTH = 500
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2


def fork(model, n=2):
    forks = []
    for i in range(n):
        f = Sequential()
        f.add(model)
        forks.append(f)
    return forks


if __name__ == '__main__':
    freeze_support()  # Optional under circumstances described in docs
    import numpy
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    from keras.optimizers import adam

    print('Indexing word vectors.')

    embeddings_index = {}
    f = open(os.path.join("", 'glove.6B.100d.txt'), encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    # Read data from files
    train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0,
                        delimiter="\t", quoting=3)
    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t",
                       quoting=3)
    test_label = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testLabel.csv'), header=0,
                             delimiter=",", quoting=3)

    # Create an empty list and append the clean reviews one by one
    num_reviews = len(train["review"])

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(pd.concat([train["review"], test["review"]], ignore_index=True))
    sequences = tokenizer.texts_to_sequences(train["review"])
    test_sequences = tokenizer.texts_to_sequences(test["review"])
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    print(test_sequences)
    x_train = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    y_train = train['sentiment']

    x_val = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    y_val = test_label['sentiment']
    print('Preparing embedding matrix.')

    # prepare embedding matrix
    num_words = min(MAX_NB_WORDS, len(word_index))
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            print(word)
            print("None")

    print('Training model.')

    # fix random seed for reproducibility
    numpy.random.seed(7)
    # load the dataset but only keep the top n words, zero the rest
    top_words = 5000

    # create the model
    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)

    embedding_vecor_length = 32

    model = Sequential()
    model.add(embedding_layer)
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    adam = adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, clipnorm=1)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    print(model.summary())
    model.fit(x_train, y_train, nb_epoch=20, batch_size=164)

    pred = model.predict(x_val)

    # Final evaluation of the model
    scores = model.evaluate(x_val, y_val, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

    # Write the test results
    output = pd.DataFrame(data={"id": test["id"], "sentiment": pred})
    output.to_csv("NBSVM.csv", index=False, quoting=3)
