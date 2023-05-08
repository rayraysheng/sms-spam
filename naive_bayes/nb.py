#!/usr/bin/env python

import numpy as np
import scipy as sp
import os
import argparse

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB


# utilities
def build_split(data, labels, indices):
    d = []
    l = []
    for i in indices:
        d.append(data[i])
        l.append(labels[i])
    return (d, l)

def predict_input(file, vect, model):
    mat_in = vect.transform(file)
    
    p = model.predict(mat_in)
    c = np.max(model.predict_proba(mat_in), axis=1)
    return p, c # predicted outcome and certainty


def main():
    ###### Pre-Processing #####
    with open("../data/SMSSpamCollection", "r", encoding="utf8") as tf:
        lines = tf.readlines()  

    # dedupe the original data
    lines = list(set(lines))

    # split data
    labels = []
    text = []

    for line in lines:
        labels.append(line.split('\t')[0])
        text.append(line.split('\t')[1])

    for i in range(len(labels)):
        if labels[i] == "ham":
            labels[i] = "not_spam"
        else:
            labels[i] = "SPAM"


    ###### Training #####
    vectorizer = CountVectorizer()
    mat_train = vectorizer.fit_transform(text)

    bnb = BernoulliNB()
    bnb.fit(mat_train, labels)


    ###### Input File Processing #####
    in_path = "../input"
    in_files = []

    for f in os.listdir(in_path):
        if os.path.isfile(os.path.join(in_path, f)):
            in_files.append(os.path.join(in_path, f))

    ##### Predict #####
    results = [] # for the whole session

    for in_fn in in_files:
        with open(in_fn, "r", encoding="utf8") as inf:
            txt = inf.readlines()
            
        results.append(predict_input(txt, vectorizer, bnb))
        

    ##### Write Output #####
    out_path = "../output"
    out_files = []

    for fn in in_files:
        name = fn.split('\\')[-1]
        out_files.append(name + "_results.csv")

    for file in out_files:
        out_fn = os.path.join(out_path, file)
        file_ind = out_files.index(file)
        predictions = results[file_ind][0]
        certainty = results[file_ind][1]

        with open(out_fn, 'w') as outf:
        #f_out = open(out_fn, "w")
            for i in range(len(txt)):
                p = predictions[i].astype(str)
                c = round(certainty[i], 2).astype(str)
                outf.write(p + ',' + c + "\n")
            outf.close()

        check = open(out_fn, "r")
        print(check.read())
        check.close()


if __name__ == '__main__':
    """
    parser = argparse.ArgumentParser(description='Description of your script')
    parser.add_argument('arg1', type=str, help='Description of argument 1')
    parser.add_argument('--arg2', type=int, default=0, help='Description of argument 2')
    parser.add_argument('--arg3', action='store_true', help='Description of argument 3')
    args = parser.parse_args()
    """

    main()
