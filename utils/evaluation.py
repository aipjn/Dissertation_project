class Evaluation(object):
    def __init__(self):

        self.how = 0
        self.where = 0
        self.why = 0
        self.yesorno = 0
        self.what = 0
        self.who = 0
        self.when = 0
        self.which = 0
        self.others = 0
        self.commonsense = 0

    def accuracy(self, y, predicty, dataset):
        accurate = 0
        i = 0
        y2 = []
        while i < len(predicty):
            if predicty[i][0] > predicty[i+1][0]:
                y2.append(0)
            else:
                y2.append(1)
            i += 2
        for j, val in enumerate(y):
            if val == y2[j]:
                accurate += 1
                self.questionType(dataset.questionList[j][1])
                if dataset.questionList[j][2] == 'commonsense':
                    self.commonsense += 1

        print("how question:", self.how/dataset.how)
        print("where question:", self.where / dataset.where)
        print("why question:", self.why / dataset.why)
        print("yesorno question:", self.yesorno / dataset.yesorno)
        print("what question:", self.what / dataset.what)
        print("who question:", self.who / dataset.who)
        print("when question:", self.when / dataset.when)
        print("which question:", self.which / dataset.which)
        print("others question:", self.others / dataset.others)
        print("commonsense question:", self.commonsense / dataset.commonsense)
        print("all accuracy", accurate/len(y))

    def questionType(self, quetype):
        if quetype == 'how':
            self.how += 1
        elif quetype == 'where':
            self.where += 1
        elif quetype == 'why':
            self.why += 1
        elif quetype == 'yesorno':
            self.yesorno += 1
        elif quetype == 'what':
            self.what += 1
        elif quetype == 'who':
            self.who += 1
        elif quetype == 'when':
            self.when += 1
        elif quetype == 'which':
            self.which += 1
        else:
            self.others += 1

