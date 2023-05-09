import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np
import csv

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer_rfc.pkl','rb'))
model = pickle.load(open('model_rfc.pkl','rb'))



#input_sms = input("Enter the message: ")

with open("batch_2.txt", "r") as f:
    messages = f.readlines()
l=[]

for input_sms in messages:
    print(input_sms)
    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    c = np.max(model.predict_proba(vector_input), axis=1)
    # 4. Display
    rr=[]
    if result == 1:
        rr.append("Spam")
    else:
        rr.append("Not Spam")
    rr.append(c)
    rr.append(input_sms)
    l.append(rr)
#print(l)
# open the CSV file for writing
with open('output_rfc2.csv', 'w', newline='') as csvfile:
    # create a CSV writer object
    writer = csv.writer(csvfile)

    # write the data to the CSV file
    for row in l:
        writer.writerow(row)
    
