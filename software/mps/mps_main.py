import sys
import time
import importlib

import torch

def read_list(model_list_file_name):
    model_list = []
    with open(model_list_file_name) as f:
        for line in f.readlines():
            if len(line.split()) != 3:
                continue
            model_list.append([line.split()[0], line.split()[1], line.split()[2]])
    return model_list


def main():
    # Load model list (task & data)
    model_list = read_list(sys.argv[1])
    model_id = int(sys.argv[2])

    task_name = model_list[model_id][0]
    data_name = model_list[model_id][1]
    num_layers = int(model_list[model_id][2])

    model_module = importlib.import_module('task.' + task_name)
    model, func, _ = model_module.import_task(data_name, num_layers)
    _, data = model_module.import_model(data_name, num_layers)
    output = func(model, data)
#    print('Training time: {} ms'.format(output))


if __name__ == '__main__':
    main()
