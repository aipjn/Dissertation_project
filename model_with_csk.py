import time
import torch.optim as optim
import torch.nn as nn
from utils.dataset import Dataset
from utils.utils import stemming, vocabulary
from utils.evaluation import Evaluation
from utils.bing import search
from RNN import train as rnn_train
from RNN import test as rnn_test
from RNN import RNN

# prepare data
begin = time.time()
data = Dataset()
vocab = vocabulary(data.trainset)

low_pro = 0.5
low_diff = 0.05


def use_csk(data, predicty, model):
    index = 0
    count = 0
    for instance in data:
        for question in instance['questions']:
            if (predicty[index][0] < low_pro and predicty[index+1][0] < low_pro) \
                    or (abs(predicty[index+1][0] - predicty[index][0]) < low_diff):
                count += 1
                texts = search(question['question'])
                if texts == '':
                    continue
                print("texts  finish")
                ques = stemming(question['question'])
                ans1 = stemming(question['answers'][0]['answer'])
                ans2 = stemming(question['answers'][1]['answer'])
                ans1_pros = []
                ans2_pros = []
                for text in texts:
                    # print(len(text))
                    text = text[0:100000]
                    text = stemming(text)
                    if text == '':
                        continue
                    output1 = model(text, ques, ans1)
                    if output1 is not None:
                        ans1_pros.append(output1.data[0][0])
                    output2 = model(text, ques, ans2)
                    if output2 is not None:
                        ans2_pros.append(output2.data[0][0])
                if len(ans1_pros) == 0 or len(ans2_pros) == 0:
                    continue
                if sorted(ans1_pros, reverse=True)[0] > sorted(ans2_pros, reverse=True)[0]:
                    predicty[index][0] = 1
                    predicty[index + 1][0] = 0
                else:
                    predicty[index][0] = 0
                    predicty[index + 1][0] = 1
            index += 2
    print("changed num", count)
    return predicty

# train
rnn = RNN(100, 128, len(vocab))
optimizer = optim.SGD(rnn.parameters(), lr=0.1)
loss_function = nn.BCELoss()
rnn_train(data.trainset, rnn, optimizer, loss_function)
# test
y, predicty = rnn_test(data.testset, rnn)
# predicty=[0.1, 0.2, 0.55, 0.51, 0.53, 0.7]
# use common sense
predicty = use_csk(data.testset, predicty, rnn)
# Evaluation
eval = Evaluation()
eval.accuracy(y, predicty, data)
final = time.time()
print("time:", final - begin)