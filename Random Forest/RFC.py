#!/usr/bin/env python

import numpy as np
import scipy as sp
import os
import argparse

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier


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
    return p # predicted outcome


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

    rfc = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rfc.fit(mat_train, labels)


    ###### Input File Processing #####
    in_path = "../input"
    in_files = []

    for f in os.listdir(in_path):
        if(not f.endswith(".DS_Store")): # for cross-platform work
            if os.path.isfile(os.path.join(in_path, f)):
                in_files.append(os.path.join(in_path, f))


    ##### Predict #####
    results = [] # for the whole session

    for in_fn in in_files:
        if(not in_fn.endswith(".DS_Store")):
            with open(in_fn, "r", encoding="utf8") as fh:
                txt = fh.readlines()

            results.append(predict_input(txt, vectorizer, rfc))


    ##### Write Output #####
    out_path = "../output"
    out_files = []

    for f in os.listdir(in_path):
        if(not f.endswith(".DS_Store")): # for cross-platform work
            out_files.append(f + "_results.csv")

    for file in out_files:
        out_fn = os.path.join(out_path, file)
        file_ind = out_files.index(file)
        predictions = results[file_ind]
        

        f_out = open(out_fn, "w")
        f_out.write("Prediction" + "\n")
        f_out.close()
        
        f_out = open(out_fn, "a")
        for i in range(len(results[file_ind])):
            p = predictions[i]
            f_out.write(p + "\n")
        f_out.close()
        f_out = open(out_fn, "a")
        
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
