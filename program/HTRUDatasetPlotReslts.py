import HTRUDatasetTesting
import matplotlib.pyplot as plt

def plotResults():
    # getting results
    r1, r2, r3, acc = HTRUDatasetTesting.getResults()
    r1v, r2v, r3v, accv = HTRUDatasetTesting.getResultsVoted()

    c = []
    c.append(r1)
    c.append(r2)
    c.append(r3)

    cv = []
    cv.append(r1v)
    cv.append(r2v)
    cv.append(r3v)

    acc = str(acc)
    accv = str(accv)

    a = ""
    for i in range(0, 2):
        a = a + acc[i + 2]
    a = a + ('%')

    av = ""
    for i in range(0, 2):
        av = av + accv[i + 2]
    av = av + ('%')

    acc = []
    acc.append(a)

    accv = []
    accv.append(av)

    # creating confusion matrix for perceptron
    columns = ('classe1', 'classe2', 'tot')
    rows = ('HTRUC1P', 'HTRUC2P', 'tot')

    tab, ax, = plt.subplots()
    tab.patch.set_visible(False)
    ax.axis('tight')

    plt.subplot(221)
    tab = plt.table(cellText=c,
                    rowLabels=rows,
                    colLabels=columns,
                    bbox=[0.0, -0.45, 1, .28],
                    loc='center')

    # table for accuracy
    plt.subplot(222)
    tab1 = plt.table(cellText=acc,
                     rowLabels='A',
                     colLabels='Per',
                     bbox=[0.0, -0.45, 1, .28],
                     loc='center')
    plt.show()

    # creating confusion matrix for votedperceptron
    columns = ('classe1', 'classe2', 'tot')
    rows = ('HTRUC1VP', 'HTRUC2VP', 'tot')

    tab, ax, = plt.subplots()
    tab.patch.set_visible(False)
    ax.axis('tight')

    plt.subplot(221)
    tab = plt.table(cellText=cv,
                    rowLabels=rows,
                    colLabels=columns,
                    bbox=[0.0, -0.45, 1, .28],
                    loc='center')

    # table for accuracy
    plt.subplot(222)
    tab1 = plt.table(cellText=accv,
                     rowLabels='A',
                     colLabels='Per',
                     bbox=[0.0, -0.45, 1, .28],
                     loc='center')
    plt.show()

plotResults()