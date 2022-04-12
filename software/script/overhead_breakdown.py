import os
import sys
import numpy as np


def list2txt(fileName="", myfile=[]):
    fileout = open(fileName, 'w')
    for i in range(len(myfile)):
        for j in range(len(myfile[i])):
            fileout.write(str(myfile[i][j]) + ' , ')
        fileout.write('\r\n')
    fileout.close()


def main():
    convs = ['GCN', 'GraphSAGE', 'GAT', 'GIN', 'mix']
    est_time = np.zeros((len(convs), 3), dtype='float32')
    sch_time = np.zeros((len(convs), 3), dtype='float32')
    gro_time = np.zeros((len(convs), 3), dtype='float32')

    for convId in range(0, 5):
        for pId in range(0, 3):
            logfile = open('{}_p{}_w2.log'.format(convs[convId], pId))
            for line in logfile:
                if line.find('PMC') > -1:
                    seg = line.split()
                    est_time[convId][pId] = float(seg[-2])
                if line.find('grouping') > -1:
                    seg = line.split()
                    gro_time[convId][pId] = float(seg[-2])
                if line.find('scheduling') > -1:
                    seg = line.split()
                    sch_time[convId][pId] = float(seg[-2])
            logfile.close()

    output = []
    for i in range(0, est_time.shape[0]):
        output.append([])
        for j in range(0, est_time.shape[1]):
            output[i].append(est_time[i][j])
    list2txt('PMC_estimation_time.csv', output)

    output = []
    for i in range(0, sch_time.shape[0]):
        output.append([])
        for j in range(0, sch_time.shape[1]):
            output[i].append(sch_time[i][j])
    list2txt('task_scheduling_time.csv', output)

    output = []
    for i in range(0, gro_time.shape[0]):
        output.append([])
        for j in range(0, gro_time.shape[1]):
            output[i].append(gro_time[i][j])
    list2txt('task_grouping_time.csv', output)


if __name__ == '__main__':
    main()
