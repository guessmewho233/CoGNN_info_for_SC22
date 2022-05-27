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
    all_qt, all_jct = [], []
    
    ne_list = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    for convId in range(0, 5):
        dur_list = []
        group_dur_list = []

        logfile = open('{}.log'.format(convs[convId]))
        for line in logfile:
            if line.find('Training') > -1:
                seg = line.split()
                dur_list.append(float(seg[-2]))
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

        all_qt.append(qt)
        all_jct.append(jct)

    list2txt('default_all_qt.csv', all_qt)
    list2txt('default_all_jct.csv', all_jct)

if __name__ == '__main__':
    main()
