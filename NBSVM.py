import numpy as np
from collections import Counter
import pandas as pd
import os
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
from sklearn import linear_model


def tokenize(sentence, grams):
    tokens = []
    words = sentence.split()
    for gram in grams:
        for i in range(len(words) - gram + 1):
            tokens += ["_*_".join(words[i:i + gram])]
    return tokens


def build_dict(train, grams):
    dic = Counter()
    for review in train:
        dic.update(tokenize(review, grams))
    return dic


def compute_ratio(poscounts, negcounts, alpha=1):
    keys = []
    for i in negcounts.keys():
        keys.append(i)
    for i in poscounts.keys():
        keys.append(i)
    alltokens = list(set(keys))
    dic = dict((t, i) for i, t in enumerate(alltokens))
    d = len(dic)
    p, q = np.ones(d) * alpha, np.ones(d) * alpha
    for t in alltokens:
        p[dic[t]] += poscounts[t]
        q[dic[t]] += negcounts[t]
    p /= abs(p).sum()
    q /= abs(q).sum()
    r = np.log(p / q)
    return dic, r


def process_text(text, dic, r, grams):
    """
    Return sparse feature matrix
    """
    X = lil_matrix((len(text), len(dic)))
    for i, l in enumerate(text):
        tokens = tokenize(l, grams)
        indexes = []
        for t in tokens:
            try:
                indexes += [dic[t]]
            except KeyError:
                pass
        indexes = list(set(indexes))
        indexes.sort()
        for j in indexes:
            X[i, j] = r[j]
    return csr_matrix(X)


if __name__ == '__main__':

    # Read data from files
    train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0,
                        delimiter="\t", quoting=3)
    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t",
                       quoting=3)
    pos = []
    neg = []
    for i in range(len(train["sentiment"])):
        if train["sentiment"][i] == 1:
            pos.append(train["review"][i])
        else:
            neg.append(train["review"][i])
    poscounts = build_dict(pos, [1, 2, 3])
    negcounts = build_dict(neg, [1, 2, 3])
    dic, r = compute_ratio(poscounts, negcounts)
    matrix = process_text(train["review"], dic, r, [1, 2, 3])

    logreg = linear_model.LogisticRegression(penalty='l2', dual=True, tol=0.0001,
                                             C=1, fit_intercept=True, intercept_scaling=1.0,
                                             class_weight=None, random_state=None)

    logreg.fit(matrix, train["sentiment"])
    label = logreg.predict(process_text(test["review"], dic, r, [1, 2, 3]))

    # Write the test results
    output = pd.DataFrame(data={"id": test["id"], "sentiment": label})
    output.to_csv("NBSVM.csv", index=False, quoting=3)
    print("Wrote NBSVM.csv")
