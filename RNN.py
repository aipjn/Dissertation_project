import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim
import time

from utils.dataset import Dataset
from utils.evaluation import Evaluation
from utils.embedding import Embedding
from utils.utils import stemming, vocabulary

torch.manual_seed(1)
embedding = Embedding()
data = Dataset()
vocab = vocabulary(data.trainset, embedding.embeddings)
word_to_ix = {word: i for i, word in enumerate(vocab)}

class RNN(nn.Module):
    def __init__(self, embedding_dim, hidden_size, vocab_size):
        super(RNN, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size + 1, embedding_dim)
        self.embedding.weight.data.copy_(self.loadEmbedding())
        self.t_lstm = nn.LSTM(embedding_dim, hidden_size,
                              num_layers=1, bidirectional=True)
        self.q_lstm = nn.LSTM(embedding_dim, hidden_size,
                              num_layers=1, bidirectional=True)
        self.a_lstm = nn.LSTM(embedding_dim, hidden_size,
                              num_layers=1, bidirectional=True)
        self.bilinear1 = nn.Bilinear(hidden_size * 2, hidden_size * 2, 1)
        self.bilinear2 = nn.Bilinear(hidden_size * 2, hidden_size * 2, 1)
        # self.atten_q_t = nn.Linear(hidden_size*2, hidden_size)
        # self.atten_a_t = nn.Linear(hidden_size*2, hidden_size)
        # self.atten_a_q = nn.Linear(hidden_size*2, hidden_size)
        # self.linear_a = nn.Linear(hidden_size*2, hidden_size*2)
        # self.linear_t = nn.Linear(hidden_size*2, hidden_size*2)
        # self.linear_q = nn.Linear(hidden_size*2, hidden_size*2)
        # self.linear_a2 = nn.Linear(hidden_size * 2, hidden_size * 2)
        # self.linear_t2 = nn.Linear(hidden_size * 2, hidden_size * 2)
        # self.linear_q2 = nn.Linear(hidden_size * 2, hidden_size * 2)




    def forward(self, text, question, answer):
        emb_t = self.getEmbedding(text).view(-1, 1, self.embedding_dim)
        emb_q = self.getEmbedding(question).view(-1, 1, self.embedding_dim)
        emb_a = self.getEmbedding(answer).view(-1, 1, self.embedding_dim)
        self.hidden_t = self.init_hidden(1)
        self.hidden_q = self.init_hidden(1)
        self.hidden_a = self.init_hidden(1)
        # self.hidden_t = self.init_hidden(len(text.split()))
        # self.hidden_q = self.init_hidden(len(question.split()))
        # self.hidden_a = self.init_hidden(len(answer.split()))
        output_t, (h_t, c_t) = self.t_lstm(emb_t, self.hidden_t)
        output_q, (h_q, c_q) = self.q_lstm(emb_q, self.hidden_q)
        output_a, (h_a, c_a) = self.a_lstm(emb_a, self.hidden_a)

        # output_a = output_a.view(-1, self.hidden_size*2)
        # output_q = output_q.view(-1, self.hidden_size * 2)
        # output_t = output_t.view(-1, self.hidden_size*2)

        # attention
        # output_t = self.atten(output_a, output_t, self.atten_a_t)
        # output_q = self.atten(output_a, output_q, self.atten_a_q)

        # t_a = torch.mm(output_a, torch.t(output_t))
        # t_a = t_a.sum() / (len(output_t) * len(output_a))
        #
        # q_a = torch.mm(output_a, torch.t(output_q))
        # q_a = q_a.sum() / (len(output_q) * len(output_a))
        #
        #
        # result = F.sigmoid((t_a + q_a) / 2)
        # result = F.sigmoid(t_a)

        h_t = h_t.view(1, -1)
        h_q = h_q.view(1, -1)
        h_a = h_a.view(1, -1)
        result = F.sigmoid(self.bilinear1(h_t, h_a) + self.bilinear2(h_q, h_a))
        print(result)
        return result


    def loadEmbedding(self):
        weights_matrix = torch.zeros((self.vocab_size + 1, self.embedding_dim))
        # from https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76
        weights_matrix[0] = torch.randn(self.embedding_dim)
        for i, word in enumerate(vocab):
            try:
                weights_matrix[i+1] = torch.from_numpy(embedding.embeddings[word])
            except KeyError:
                weights_matrix[i+1] = torch.randn(self.embedding_dim)
        return weights_matrix


    def getEmbedding(self, text):
        # embs = []
        indexs = []
        for word in text.split():
            if word in vocab:
                indexs.append(word_to_ix[word] + 1)
            # elif word in embedding.embeddings.keys():
            #     emb = torch.from_numpy(embedding.embeddings[word])
            #     embs.append(emb)
            else:
                indexs.append(0)
        return self.embedding(autograd.Variable(torch.LongTensor(indexs)))

    def init_hidden(self, batch):
        return (autograd.Variable(torch.randn(2, batch, self.hidden_size)),
                autograd.Variable(torch.randn(2, batch, self.hidden_size)))


    def atten(self, embs1, embs2, linear):
        # embs1 = embs1.squeeze(0)
        # embs2 = embs2.squeeze(0)
        for i in range(embs1.size()[0]):
            dots = torch.dot(F.relu(linear(embs1[i])), F.relu(linear(embs2[0])))
            for j in range(embs2.size()[0] - 1):
                dot = torch.dot(F.relu(linear(embs1[i])), F.relu(linear(embs2[j+1])))
                dots = torch.cat((dots, dot), 0)
            attention = autograd.Variable(torch.zeros(self.hidden_size * 2), requires_grad=True)
            alphas = F.softmax(dots, 0)
            for j in range(embs2.size()[0]):
                attention = torch.add(alphas[j] * embs2[j], attention)
            if i == 0:
                embs = attention.view(1, -1)
            else:
                embs = torch.cat((embs, attention.view(1, -1)), 0)
        return embs

    # def getEmbedding(self, text):
    #     embs = []
    #     for word in text.split():
    #         emb = torch.zeros(self.embedding_dim)
    #         if word in embedding.embeddings.keys():
    #             emb = torch.from_numpy(embedding.embeddings[word])
    #         embs.append(emb)
    #     return autograd.Variable(torch.cat(embs).view(1, len(embs), -1))

def train(trainset, model, optimizer, loss_function):
    index = 0
    losses = []
    for epoch in range(10):
        total_loss = torch.Tensor([0])
        for instance in trainset:
            print(index)
            index += 1
            for question in instance['questions']:
                text = stemming(instance['text'])
                ques = stemming(question['question'])
                for answer in question['answers']:
                    model.zero_grad()
                    ans = stemming(answer['answer'])
                    output = model(text, ques, ans)
                    if answer['correct'] == 'True':
                        y = autograd.Variable(torch.FloatTensor([1]))
                    else:
                        y = autograd.Variable(torch.FloatTensor([0]))
                    loss = loss_function(output, y)
                    # print('output', output.data[0])
                    # print('loss', loss.data[0])
                    loss.backward()
                    optimizer.step()
                    # for param in model.parameters():
                    #     print('param', param.grad.data.sum())
                    total_loss += loss.data[0]
        losses.append(total_loss)
    print(losses)

def test(testset, model):
    y = []
    predict_proba = []
    for instance in testset:
        for question in instance['questions']:
            text = stemming(instance['text'])
            ques = stemming(question['question'])
            for answer in question['answers']:
                ans = stemming(answer['answer'])
                output = model(text, ques, ans)
                predict_proba.append([output.data[0][0]])
                # print(output)
            if question['answers'][0]['correct'] == 'True':
                y.append(0)
            else:
                y.append(1)
    return y, predict_proba

if __name__ == '__main__':
    begin = time.time()
    rnn = RNN(100, 128, len(vocab))
    optimizer = optim.SGD(rnn.parameters(), lr=0.1)
    loss_function = nn.BCELoss()
    train(data.trainset, rnn, optimizer, loss_function)
    y, predicty = test(data.testset, rnn)
    # print(len(y))
    eval = Evaluation()
    eval.accuracy(y, predicty, data)
    final = time.time()
    print("time:", final - begin)
