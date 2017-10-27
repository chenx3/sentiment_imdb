import numpy as np
from sklearn.metrics import accuracy_score
from numpy import round
import pandas as pd
import os


def ensemble(models, label):
    values = np.arange(0.0, 1.0, 0.1)
    print(values)
    max = 0
    max_score = []
    for i in values:
        for j in values:
            score = round(i * models[0] + j * models[1] + (1 - i - j) * models[2])
            accuracy = accuracy_score(score, label)
            if accuracy > max:
                max = accuracy
                max_score = score
    return max_score, max


label = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testLabel.csv'), header=0,
                    delimiter=",", quoting=3)
BOW = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Bag_of_Words_model.csv'), header=0,
                  delimiter=",", quoting=3)
NBSVM = pd.read_csv(os.path.join(os.path.dirname(__file__), 'NBSVM.csv'), header=0,
                    delimiter=",", quoting=3)
Word2Vec = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Word2Vec_AverageVectors.csv'), header=0,
                       delimiter=",", quoting=3)
models = [BOW['sentiment'], NBSVM['sentiment'], Word2Vec['sentiment']]
[max_score , max] = ensemble(models, label['sentiment'])
accuracy = accuracy_score(round(BOW['sentiment']), label['sentiment'])
print(accuracy)
# Write the test results
output = pd.DataFrame(data={"id": label["id"], "sentiment": max_score})
output.to_csv("ensemble.csv", index=False, quoting=3)
