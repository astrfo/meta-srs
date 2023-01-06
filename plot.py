import os
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt

def print_regret(args):
    csv_files = glob.glob('./log/' + args[2] + '/regret.csv')
    for file in csv_files:
        name = os.path.splitext(os.path.basename(file))[0]
        data = pd.read_csv(file)
        plt.plot(data, label=name)
    plt.legend()
    plt.title(args[1])
    plt.xlabel('step')
    plt.ylabel('regret')
    plt.savefig('./log/' + args[2] + '/csv_plot_regret.png')
    plt.show()

def print_rate(args):
    csv_files = glob.glob('./log/' + args[2] + '/rate.csv')
    for file in csv_files:
        # name = os.path.splitext(os.path.basename(file))[0]
        data = pd.read_csv(file, names=['SRS', 'SRS-CH'])
        plt.plot(data, label=data.columns)
    # print(csv_files_rate)
    # plt.plot(csv_files_rate, label=csv_files_rate.columns)
    plt.legend()
    plt.title(args[1])
    plt.xlabel('step')
    plt.ylabel('rate')
    plt.ylim(-0.03, 1.03)
    plt.savefig('./log/' + args[2] + '/csv_plot_rate.png')
    plt.show()


if __name__ == '__main__':
    args = sys.argv
    if len(args) <= 1:
        print('wrong number of arguments')
        sys.exit()
    print_regret(args)
    print_rate(args)
    