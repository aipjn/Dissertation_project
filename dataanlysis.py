from utils.dataset import Dataset

if __name__ == '__main__':
    data = Dataset()
    data.read_xml("MCScript/test-data.xml", test=True)
    print("how question:", data.how)
    print("where question:", data.where)
    print("why question:", data.why)
    print("yesorno question:", data.yesorno)
    print("what question:", data.what)
    print("who question:", data.who)
    print("when question:", data.when)
    print("which question:", data.which)
    print("others question:", data.others)
    print("commonsense question:", data.commonsense)
    print("long answer:", data.longAns)
    print("short answer:", data.shortAns)
    print("text_len:", data.text_len)
    print("ques_len:", data.ques_len)