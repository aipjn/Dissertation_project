from sklearn import svm

from utils.dataset import Dataset
from utils.evaluation import Evaluation

from utils.embedding import Embedding
import numpy as np
import time
# nltk.download('stopwords')
from utils.utils import removeStopwords


embedding = Embedding()

def extra_feature(text, answer):
    features = []
    count = 0
    for word in answer.split():
        if word in text:
            count += 1
    # how many words in answer appeared in text divide answer length
    if len(answer) == 0:
        features.append(0)
    else:
        features.append(count/len(answer))
    return features


def similarities(text, answer, question):
    # bagsim = 0
    # embsim = 0
    sim = []
    windowsize = 30
    for i in range(1):
        index = alignment(text, question, windowsize)
        if answer == '':
            bagsim = 0
            embsim = 0
        else:
            bagsim = bagofwordSimilarity(text, index, windowsize, answer)
            textembeddings = textembs(text, index, windowsize)
            embsim = cos(textembeddings, wordembedding(answer))
            # embsim += cos(textembeddings, wordembedding(answer) + wordembedding(question))
        windowsize += 1
        sim.append(bagsim)
        sim.append(embsim)
    return sim

def q_a_similarities(answer, question):
    sim = []
    if answer == '' or question == '':
        bagsim = 0
        embsim = 0
    else:
        bagsim = bagofwordSimilarity(question, 0, len(question.split()), answer)
        textembeddings = textembs(question, 0, len(question.split()))
        embsim = cos(textembeddings, wordembedding(answer))
        # embsim += cos(textembeddings, wordembedding(answer) + wordembedding(question))
    sim.append(bagsim)
    sim.append(embsim)
    return sim

def bagofwordSimilarity(text, index, windowsize, answer):
    queans = {}
    for word in answer.split():
        queans[word] = 1 if word not in queans.keys() else queans[word] + 1
    # for word in question.split():
    #     queans[word] = 1 if word not in queans.keys() else queans[word] + 1
    textDict = {}
    for i in range(windowsize):
        word = text.split()[i + index]
        textDict[word] = 1 if word not in textDict.keys() else textDict[word] + 1
    dot = 0
    for word, value in queans.items():
        if word in textDict.keys():
            dot += value * textDict[word]
    size1 = 0
    for value in queans.values():
        size1 += value*value
    size2 = 0
    for value in textDict.values():
        size2 += value*value
    return dot / (np.sqrt(size1) * np.sqrt(size2))

def wordembedding(text):
    embs = np.zeros(100)
    for word in text.split():
        if word in embedding.embeddings.keys():
            embs += embedding.embeddings[word]
    return embs


def textembs(text, index, windowsize):
    embs = np.zeros(100)
    for i in range(windowsize):
        word = text.split()[i + index]
        if word in embedding.embeddings.keys():
            embs += embedding.embeddings[word]
    return embs

# from http://grokbase.com/t/python/python-list/12c61pc8gh/cosine-similarity
def cos(v1, v2):
    if np.sqrt(np.dot(v1, v1)) == 0 or np.sqrt(np.dot(v2, v2)) == 0:
        return 0
    return np.dot(v1, v2) / (np.sqrt(np.dot(v1, v1)) * np.sqrt(np.dot(v2, v2)))

 # use sliding window find alignment
def alignment(text, question, windowsize):
    count = 0
    # count how many overlap words in the first windowsize text with question
    for word in question.split():
        for j in range(windowsize):
            if word == text.split()[j]:
                count += 1
    maxcount = count
    index = 0
    # when sliding window just consider the discarded word and the added word
    for i in range(len(text.split()) - windowsize):
        if text.split()[i] in question.split():
            count = count - 1
        if text.split()[i + windowsize] in question.split():
            count = count + 1
        if count > maxcount:
            maxcount = count
            index = i + 1
    return index


def train(trainset, model):
    X = []
    y = []
    for instance in trainset:
        for question in instance['questions']:
            for answer in question['answers']:
                text = removeStopwords(instance['text'])
                ques = removeStopwords(question['question'])
                ans = removeStopwords(answer['answer'])
                # text = instance['text']
                # ques = question['question']
                # ans = answer['answer']
                X.append(extra_feature(text, ans) + similarities(text, ans, ques) + q_a_similarities(ans, ques))
                if answer['correct'] == 'True':
                    y.append(0)
                else:
                    y.append(1)
    model.fit(X, y)

def test(testset, model):
    X = []
    y = []
    for instance in testset:
        for question in instance['questions']:
            text = removeStopwords(instance['text'])
            ques = removeStopwords(question['question'])
            ans1 = removeStopwords(question['answers'][0]['answer'])
            ans2 = removeStopwords(question['answers'][1]['answer'])
            # text = instance['text']
            # ques = question['question']
            # ans1 = question['answers'][0]['answer']
            # ans2 = question['answers'][1]['answer']
            X.append(extra_feature(text, ans1) + similarities(text, ans1, ques) + q_a_similarities(ans1, ques))
            X.append(extra_feature(text, ans2) + similarities(text, ans2, ques) + q_a_similarities(ans2, ques))
            if question['answers'][0]['correct'] == 'True':
                y.append(0)
            else:
                y.append(1)
    return y, model.predict_proba(X)

if __name__ == '__main__':
    begin = time.time()
    data = Dataset()
    model = svm.SVC(gamma=10, probability=True)
    train(data.trainset, model)
    y, predicty = test(data.testset, model)
    eval = Evaluation()
    eval.accuracy(y, predicty, data)
    final = time.time()
    print("time", final - begin)
