import xml.etree.ElementTree as ET
import string

class Dataset(object):

    def __init__(self):

        trainset_path = "MCScript/train-data.xml"
        testset_path = "MCScript/dev-data.xml"

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

        self.questionList = []

        self.trainset = self.read_xml(trainset_path)
        self.testset = self.read_xml(testset_path, test=True)


    def read_xml(self, path, test=False):
        datalist = []
        tree = ET.ElementTree(file=path)
        root = tree.getroot()
        for child in root:
            ins = {}
            # print(child.tag, child.attrib['id'])
            # text
            # print(child[0].tag)
            # print(child[0].text)
            ins['text'] = self.formateLine(child[0].text)
            questions = []
            for question in child[1]:
                # question
                ques = {}
                # print(questions.tag)
                # print(questions.attrib['text'])
                ques['question'] = self.formateLine(question.attrib['text'])
                answers = []
                for answer in question:
                    # answers
                    ans = {}
                    # print(answer.tag)
                    # print(answer.attrib['text'])
                    ans['answer'] = self.formateLine(answer.attrib['text'])
                    ans['correct'] = answer.attrib['correct']
                    answers.append(ans)
                # store question information for evaluation
                if test:
                    self.questionList.append([ques['question'],
                                              self.questionType(ques['question'].split()[0]),
                                              question.attrib['type'],
                                              answers])
                    if question.attrib['type'] == 'commonsense':
                        self.commonsense += 1
                ques['answers'] = answers
                questions.append(ques)
            ins['questions'] = questions
            datalist.append(ins)
        return datalist

    # lowercase the texts and remove punctuation
    def formateLine(self, line):
        line = line.lower()
        translator = str.maketrans('', '', string.punctuation)
        return line.translate(translator)

    def questionType(self, queBeginner):
        if queBeginner == 'how':
            self.how += 1
            return 'how'
        elif queBeginner == 'where':
            self.where += 1
            return 'where'
        elif queBeginner == 'why':
            self.why += 1
            return 'why'
        elif queBeginner in 'did do does was were can could is are have had will would':
            self.yesorno += 1
            return 'yesorno'
        elif queBeginner in 'what whats':
            self.what += 1
            return 'what'
        elif queBeginner in 'who whom whose':
            self.who += 1
            return 'who'
        elif queBeginner == 'when':
            self.when += 1
            return 'when'
        elif queBeginner == 'which':
            self.which += 1
            return 'which'
        else:
            self.others += 1
            return 'others'





