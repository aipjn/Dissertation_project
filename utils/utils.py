from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

stopWords = set(stopwords.words('english'))
ps = PorterStemmer()
wl = WordNetLemmatizer()


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

def vocabulary(traindata, embedding):
    vocab = ''
    for instance in traindata:
        vocab += stemming(instance['text'])
        for question in instance['questions']:
            vocab += stemming(question['question'])
            for answer in question['answers']:
                vocab += stemming(answer['answer'])
    # keys = []
    # for key in embedding.keys():
    #     keys.append(key)
    # print(len(keys), len(vocab.split()))
    vocabset = set(vocab.split())

    return vocabset