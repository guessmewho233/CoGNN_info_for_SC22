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
    makespan_total = np.zeros((len(convs), 6), dtype='float32')
    ave_qt_total = np.zeros((len(convs), 6), dtype='float32')
    ave_jct_total = np.zeros((len(convs), 6), dtype='float32')

    for convId in range(0, 5):
        queueCounter = 0
        for pId in range(0, 3):
            for wId in range(2, 4):
                ne_list, dur_list, group_dur_list = [], [], []
                logfile = open('{}_p{}_w{}.log'.format(convs[convId], pId, wId))
                for line in logfile:
                    if line.find('elements') > -1:
                        seg = line.split()
                        ne_list.append(int(seg[-1]))
                    if line.find('Training') > -1:
                        seg = line.split()
                        dur_list.append(float(seg[-2])
                logfile.close()

                eleCounter = 0
                for groupId in range(0, len(ne_list)):
                    group_dur_list.append([])
                    for _ in range(ne_list[groupId]):
                        group_dur_list[groupId].append(dur_list[eleCounter])
                        eleCounter += 1
                
                group_max_dur, group_qt = [], []
                tmp_qt = 0
                for groupId in range(0, len(group_dur_list)-1):
                    tmp_max_dur = 0
                    for eleId in range(0, len(group_dur_list[groupId])):
                        if group_dur_list[groupId][eleId] > tmp_max_dur:
                            tmp_max_dur = group_dur_list[groupId][eleId]
                    group_max_dur.append(tmp_max_dur)
                    tmp_qt += tmp_max_dur
                    group_qt.append(tmp_qt)

                qt, jct = [], []
                for groupId in range(0, len(group_dur_list)):
                    for eleId in range(0, len(group_dur_list[groupId])):
                        ele_qt = group_qt[groupId-1]
                        if groupId == 0:  ele_qt = 0
                        ele_jct = ele_qt + group_dur_list[groupId][eleId]
                        qt.append(ele_qt)
                        jct.append(ele_jct)
                ave_qt, ave_jct = np.mean(qt), np.mean(jct)

                makespan_total[convId][queueCounter] = jct[-1]
                ave_qt_total[convId][queueCounter] = ave_qt
                ave_jct_total[convId][queueCounter] = ave_jct
                queueCounter += 1

    output = []
    for i in range(0, makespan_total.shape[0]):
        output.append([])
        for j in range(0, makespan_total.shape[1]):
            output[i].append(makespan_total[i][j])
    list2txt('cognn_makespan.csv', output)

    output = []
    for i in range(0, ave_jct_total.shape[0]):
        output.append([])
        for j in range(0, ave_jct_total.shape[1]):
            output[i].append(ave_jct_total[i][j])
    list2txt('cognn_avejct.csv', output)

    output = []
    for i in range(0, ave_qt_total.shape[0]):
        output.append([])
        for j in range(0, ave_qt_total.shape[1]):
            output[i].append(ave_qt_total[i][j])
    list2txt('cognn_aveqt.csv', output)


if __name__ == '__main__':
    main()
