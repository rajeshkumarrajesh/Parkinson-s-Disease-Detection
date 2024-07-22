import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from sklearn.metrics import roc_curve
from itertools import cycle


def Statistical(data):
    Min = np.min(data)
    Max = np.max(data)
    Mean = np.mean(data)
    Median = np.median(data)
    Std = np.std(data)
    return np.asarray([Min, Max, Mean, Median, Std])


def plotConvResults():
    # matplotlib.use('TkAgg')
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'DMO', 'BFGO', 'RHA', 'POA', 'PROPOSED']

    Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    for i in range(1):
        Conv_Graph = np.zeros((5, 5))
        for j in range(5):  # for 5 algms
            Conv_Graph[j, :] = Statistical(Fitness[i, j, :])

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('-------------------------------------------------- Statistical Analysis  ',
              '--------------------------------------------------')
        print(Table)

        length = np.arange(50)
        Conv_Graph = Fitness[i]
        plt.plot(length, Conv_Graph[0, :], color='#FF69B4', linewidth=3, marker='*', markerfacecolor='red',
                 markersize=12, label='DMO-MWF-RconvLSTM')
        plt.plot(length, Conv_Graph[1, :], color='#7D26CD', linewidth=3, marker='*', markerfacecolor='#00FFFF',
                 markersize=12, label='BFGO-MWF-RconvLSTM')
        plt.plot(length, Conv_Graph[2, :], color='#FF00FF', linewidth=3, marker='*', markerfacecolor='blue',
                 markersize=12, label='RHA-MWF-RconvLSTM')
        plt.plot(length, Conv_Graph[3, :], color='#43CD80', linewidth=3, marker='*', markerfacecolor='magenta',
                 markersize=12, label='POA-MWF-RconvLSTM')
        plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
                 markersize=12, label='EPOA-MWF-RconvLSTM')
        plt.xlabel('No. of Iteration')
        plt.ylabel('Cost Function')
        plt.legend(loc=1)
        # plt.savefig("./Results/Conv.png")
        plt.show()


def Plot_ROC_Curve():
    lw = 2
    cls = ['PDCNNet', 'NN', 'CNN', 'MWF-RconvLSTM', 'EPOA-MWF-RconvLSTM']
    Actual = np.load('Target.npy', allow_pickle=True).astype('int')

    colors = cycle(["blue", "darkorange", "y", "cyan", "black"])
    for i, color in zip(range(5), colors):  # For all classifiers
        Predicted = np.load('Y_Score.npy', allow_pickle=True)[0][i]
        false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual.ravel(), Predicted.ravel())
        plt.plot(
            false_positive_rate1,
            true_positive_rate1,
            color=color,
            lw=lw,
            label=cls[i])

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('Accuracy')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    # path = "./Results/ROC.png"
    # plt.savefig(path)
    plt.show()


def plot_results():
    eval = np.load('Eval_KFold.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1_score', 'MCC',
             'FOR', 'PT',
             'BA', 'FM', 'BM', 'MK', 'PLHR', 'lrminus', 'DOR', 'Prevalence', 'Threat Score']
    Graph_Terms = [0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 18, 20]
    Algorithm = ['TERMS', 'DMO', 'BFGO', 'RHA', 'POA', 'PROPOSED']
    Classifier = ['TERMS', 'PDCNNet', 'NN', 'CNN', 'MWF-RconvLSTM ', 'PROPOSED']
    for i in range(eval.shape[0]):
        value1 = eval[i, 4, :, 4:]

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], value1[j, :])
        print('--------------------------------------------------  K-fold - Dataset', i + 1, 'Algorithm Comparison ',
              '--------------------------------------------------')
        print(Table)

        Table = PrettyTable()
        Table.add_column(Classifier[0], Terms)
        for j in range(len(Classifier) - 2):
            Table.add_column(Classifier[j + 1], value1[len(Algorithm) + j - 1, :])
        Table.add_column(Classifier[5], value1[4, :])
        print('-------------------------------------------------- K-fold - Dataset', i + 1, 'Classifier Comparison',
              '--------------------------------------------------')
        print(Table)

    kfold = [1, 2, 3, 4, 5]
    for i in range(eval.shape[0]):
        for j in range(len(Graph_Terms)):
            Graph = np.zeros(eval.shape[1:3])
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    if j == 9:
                        Graph[k, l] = eval[i, k, l, Graph_Terms[j] + 4]
                    else:
                        Graph[k, l] = eval[i, k, l, Graph_Terms[j] + 4]

            length = np.arange(5)
            fig = plt.figure()
            ax = fig.add_axes([0.15, 0.1, 0.7, 0.7])

            ax.plot(length, Graph[:, 0], color='#010fcc', linewidth=3, marker='*', markerfacecolor='red',  # 98F5FF
                    markersize=16,
                    label='DMO-MWF-RconvLSTM')
            ax.plot(length, Graph[:, 1], color='#08ff08', linewidth=3, marker='*', markerfacecolor='green',  # 7FFF00
                    markersize=16,
                    label='BFGO-MWF-RconvLSTM')
            ax.plot(length, Graph[:, 2], color='#fe420f', linewidth=3, marker='*', markerfacecolor='cyan',  # C1FFC1
                    markersize=16,
                    label='RHA-MWF-RconvLSTM')
            ax.plot(length, Graph[:, 3], color='#f504c9', linewidth=3, marker='*', markerfacecolor='#fdff38',
                    markersize=16,
                    label='POA-MWF-RconvLSTM')
            ax.plot(length, Graph[:, 4], color='k', linewidth=3, marker='*', markerfacecolor='r', markersize=16,
                    label='EPOA-MWF-RconvLSTM')

            ax.fill_between(length, Graph[:, 0], Graph[:, 1], color='#ff8400', alpha=.5)
            ax.fill_between(length, Graph[:, 1], Graph[:, 2], color='#19abff', alpha=.5)
            ax.fill_between(length, Graph[:, 2], Graph[:, 3], color='#00f7ff', alpha=.5)
            ax.fill_between(length, Graph[:, 3], Graph[:, 4], color='#ecfc5b', alpha=.5)
            plt.xticks(length, ('1', '2', '3', '4', '5'))
            plt.xlabel('KFold')
            plt.ylabel(Terms[Graph_Terms[j]])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2, fancybox=True, shadow=True)
            # path = "./Results/%s_line.png" % (Terms[Graph_Terms[j]])
            # plt.savefig(path)
            plt.show()

            fig = plt.figure()
            ax = fig.add_axes([0.12, 0.12, 0.8, 0.8])
            X = np.arange(5)
            ax.barh(X + 0.00, Graph[:, 5], color='#4EEE94',  height=0.10, label="PDCNNet")
            ax.barh(X + 0.10, Graph[:, 6], color='#9A32CD', height=0.10,  label="NN")
            ax.barh(X + 0.20, Graph[:, 7], color='#FF1493', height=0.10, label="CNN")
            ax.barh(X + 0.30, Graph[:, 8], color='#FFC125', height=0.10, label="MWF-RconvLSTM")
            ax.barh(X + 0.40, Graph[:, 4], color='k', height=0.10, label="EPOA-MWF-RconvLSTM")
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13),
                       ncol=3, fancybox=True, shadow=True)
            plt.yticks(X + 0.20, ('1', '2', '3', '4', '5'))
            plt.ylabel('KFold')
            plt.xlabel(Terms[Graph_Terms[j]])
            # path = "./Results/%s_bar.png" % (Terms[Graph_Terms[j]])
            # plt.savefig(path)
            plt.show()


if __name__ == '__main__':
    plotConvResults()
    plot_results()
    Plot_ROC_Curve()
