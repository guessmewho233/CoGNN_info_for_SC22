import os
import sys
import numpy as np


def list2txt(fileName='', myfile=[]):
    fileout = open(fileName, 'w')
    for i in range(len(myfile)):
        for j in range(len(myfile[i])):
            fileout.write(str(myfile[i][j]) + ' , ')
        fileout.write('\r\n')
    fileout.close()


def main():
    convs = ['GCN', 'GraphSAGE', 'GAT', 'GIN']
    layers = [4, 6, 8, 10]
    datasets = ['citeseer', 'cora', 'pubmed', 'PROTEINS_full', 'artist', 'soc-BlogCatalog', 'DD', 'amazon0601', 'TWITTER-Real-Graph-Partial', 'Yeast', 'OVCAR-8H']

    relative_error = np.zeros((len(convs), len(datasets), len(layers)), dtype='float32')

    for convId in range(0, len(convs)):
        for datasetId in range(0, len(datasets)):
            for layerId in range(0, len(layers)):
                logfile = open('{}_{}_layer{}.log'.format(convs[convId], datasets[datasetId], layers[layerId]), 'r')
                para_size, est_comp_size, peak_mem = 0, 0, 0
                for line in logfile:
                    if line.find('para_size') > -1:
                        seg = line.split()
                        para_size, est_comp_size = float(seg[1]), float(seg[-2])
                    if line.find('Allocated memory') > -1:
                        seg = line.split()
                        peak_mem, mem_unit = int(seg[7]), seg[8]
                        if mem_unit == 'KB':
                            peak_mem /= 1024
                logfile.close()
                relative_error[convId][datasetId][layerId] = abs(peak_mem - para_size - est_comp_size) / peak_mem

    for convId in range(len(convs)):
        output = []
        for i in range(len(datasets)):
            output.append([])
            for j in range(len(layers)):
                output[i].append(relative_error[convId][i][j])
        list2txt('{}_pmc_est_re.csv'.format(convs[convId]), output)

if __name__ == '__main__':
    main()
