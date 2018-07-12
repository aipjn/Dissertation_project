import xml.etree.ElementTree as ET
import string

class Dataset(object):

    def __init__(self):

        trainset_path = "MCScript/train-data.xml"
        testset_path = "MCScript/dev-data.xml"

        self.trainset = self.read_xml(trainset_path)
        self.testset = self.read_xml(testset_path)


    def read_xml(self, path):
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




