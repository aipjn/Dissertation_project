"""\
------------------------------------------------------------
USE: Some useful functions
------------------------------------------------------------\
"""
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import string

stopWords = set(stopwords.words('english'))
ps = PorterStemmer()
wl = WordNetLemmatizer()


# lowercase the texts and remove punctuation
def formateLine(line):
    line = line.lower()
    translator = str.maketrans('', '', string.punctuation)
    return line.translate(translator)

def removeStopwords(text):
    newtext = ''
    for word in text.split():
        if word not in stopWords:
            newtext += wl.lemmatize(ps.stem(word)) + ' '
    return newtext

def stemming(text):
    newtext = ''
    for word in text.split():
        newtext += wl.lemmatize(ps.stem(word)) + ' '
    return newtext

# find the vocabulary in traindata
def vocabulary(traindata):
    vocab = ''
    for instance in traindata:
        vocab += stemming(instance['text'])
        for question in instance['questions']:
            vocab += stemming(question['question'])
            for answer in question['answers']:
                vocab += stemming(answer['answer'])
    vocabset = set(vocab.split())

    return vocabset