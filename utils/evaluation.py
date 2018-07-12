

def accuracy(y, predicty):
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
    return accurate/len(y)
