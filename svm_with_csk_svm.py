import time
from utils.dataset import Dataset
from utils.utils import stemming, vocabulary
from utils.evaluation import Evaluation
from utils.bing import search


import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

from baseline_svm import train as svm_train
from baseline_svm import test as svm_test
from baseline_svm import similarities
from baseline_svm import q_a_similarities
from baseline_svm import extract_feature
from sklearn import svm

# prepare data
begin = time.time()
data = Dataset()
vocab = vocabulary(data.trainset)

low_pro = 0.5
low_diff = 0.05


def use_csk(data, predicty, model):
    index = 0
    count = 0
    data_c = []
    for instance in data:
        for question in instance['questions']:
            if (predicty[index][0] < low_pro and predicty[index + 1][0] < low_pro):
                count += 1
                texts = search(question['question'])
                if texts == '':
                    continue
                # print("texts  finish")
                ques = stemming(question['question'])
                ans1 = stemming(question['answers'][0]['answer'])
                ans2 = stemming(question['answers'][1]['answer'])
                data_c.append([index, ques, ans1, ans2, texts])
            index += 2
    print("changed num", count)
    #  update result
    for value in data_c:
        key = value[0]
        ans1_pros = [predicty[key][0]]
        ans2_pros = [predicty[key + 1][0]]
        ques = value[1]
        ans1 = value[2]
        ans2 = value[3]
        texts = value[4]
        X = []
        for text in texts:
            # print(len(text))
            text = text[0:100000]
            text = stemming(text)
            if text == '' or len(text.split()) < 30:
                continue
            X.append(extract_feature(text, ans1) + similarities(text, ans1, ques) + q_a_similarities(ans1, ques))
            X.append(extract_feature(text, ans2) + similarities(text, ans2, ques) + q_a_similarities(ans2, ques))
        if len(X) == 0:
            continue
        result = model.predict_proba(X)
        i = 0
        while i < len(result):
            ans1_pros.append(result[i][0])
            ans2_pros.append(result[i + 1][0])
            i += 2
        if sorted(ans1_pros, reverse=True)[0] > sorted(ans2_pros, reverse=True)[0]:
            predicty[key][0] = 1
            predicty[key + 1][0] = 0
        else:
            predicty[key][0] = 0
            predicty[key + 1][0] = 1
        print("update")

    return predicty

# train
model = svm.SVC(gamma=10, probability=True)
svm_train(data.trainset, model, 1470)
y, predicty = svm_test(data.testset, model)
eval1 = Evaluation()
eval1.accuracy(y, predicty, data)
with open('result_svm.txt', 'w') as f:
    for index, maxd in enumerate(eval1.wrong):
        f.write("Case #{}: {} ".format(index + 1, maxd) + '\n')
# predicty=[[0.1], [0.2], [0.1], [0.2], [0.1], [0.2]]
predicty = use_csk(data.testset, predicty, model)
# Evaluation
eval = Evaluation()
eval.accuracy(y, predicty, data)

final = time.time()
print("time:", final - begin)