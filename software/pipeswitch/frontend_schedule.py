import threading
import torch
import importlib
import time

from util.util import timestamp
from cognn.policy import *

class FrontendScheduleThd(threading.Thread):
    def __init__(self, model_list, qin, worker_list):
        super(FrontendScheduleThd, self).__init__()
        self.model_list = model_list
        self.qin = qin
        self.worker_list = worker_list
        self.cur_w_idx = 0

    def run(self):
        timestamp('schedule', 'start')

        # Load models
        models = {}
        para_bytes = {}
        comp_bytes = {}
        
        for model_name in self.model_list:
            hash_name = hash('{}_{}_{}'.format(model_name[0], model_name[1], int(model_name[2])))
            models[hash_name], _, _ = self._load_model(model_name)

        timestamp('schedule', 'load_model')

        # Create CUDA stream
        cuda_stream_for_parameter = torch.cuda.Stream()
        timestamp('schedule', 'create_stream')
        
        job_list = []
        while True:
            # Get request           
            agent, task_name, data_name, num_layers = self.qin.get()
            job_list.append([agent, task_name, data_name, num_layers])
            timestamp('schedule', 'get_request')
            if len(job_list) == len(models):
                break
        
#        for group_iter in range(len(reorder_job_list)):
#            print('num of elements: {}'.format(len(reorder_job_list[group_iter])))
#        print('after here')
        para_cache_size, comp_cache_size = 1024 * 1024 * 1024, 25 * 1024 * 1024 * 1024
        for group_iter in range(len(reorder_job_list)):
            for job in reorder_job_list[group_iter]:
                agent, task_name, data_name, num_layers = job[0], job[1], job[2], job[3]
                # get next worker to work on request
                self.cur_w_idx %= len(self.worker_list)
                new_pipe, _, param_trans_pipe_parent, _ = self.worker_list[self.cur_w_idx]
                self.cur_w_idx += 1

                # send request to new worker
                model_name = []
                for model in self.model_list:
                    if model[0] == task_name and model[1] == data_name and model[2] == num_layers:
                        model_name = model
                new_pipe.send((agent, model_name, para_cache_size, comp_cache_size))
                timestamp('schedule', 'notify_new_worker')

                # allocate cache to streams
                num_layers = int(model_name[2])                
                timestamp('schedule', 'insert_cache')
                with torch.cuda.stream(cuda_stream_for_parameter):
                    torch.cuda.insert_shared_cache_for_computation(para_cache_size, 0)

                # transfer parameters to gpu
                batched_parameter_list = models[hash('{}_{}_{}'.format(task_name, data_name, num_layers))]
                self._transfer_parameter(new_pipe,
                                         batched_parameter_list, 
                                         cuda_stream_for_parameter,
                                         param_trans_pipe_parent)
                timestamp('schedule', 'transfer_parameters')

                # clear status
                with torch.cuda.stream(cuda_stream_for_parameter):
                    torch.cuda.clear_shared_cache() # pylint: disable=no-member
                timestamp('schedule', 'clear_status')

            res_counter = 0
            while True:
                self.cur_w_idx %= len(self.worker_list)
                new_pipe, _, _, _ = self.worker_list[self.cur_w_idx]
                self.cur_w_idx += 1
                # Recv response
                if new_pipe.poll():
                    res = new_pipe.recv()
                    res_counter += 1
                if res_counter == len(reorder_job_list[group_iter]):
                    break

    def _load_model(self, model_name):
        # Import parameters
        model_module = importlib.import_module('task.' + model_name[0])
        batched_parameter_list, comp_total_bytes = model_module.import_parameters(model_name[1], int(model_name[2]))
        # Preprocess batches
        processed_batched_parameter_list = []
        batched_parameter_total_bytes = 0
        for param, mod_list in batched_parameter_list:
            if param is None:
                processed_batched_parameter_list.append((None, mod_list))
            else:
                processed_batched_parameter_list.append((param.pin_memory(), mod_list))
                batched_parameter_total_bytes += param.element_size() * param.nelement()

        return processed_batched_parameter_list, batched_parameter_total_bytes, comp_total_bytes

    def _transfer_parameter(self, pipe, 
                            batched_parameter_list, 
                            cuda_stream_for_parameter,
                            param_trans_pipe):
        param_cuda_list = []
        for param, mod_list in batched_parameter_list:
            with torch.cuda.stream(cuda_stream_for_parameter):
                if param is not None:
                    param_cuda = param.cuda(non_blocking=True)
                    param_cuda_list.append(param_cuda)
                    e = torch.cuda.Event()
                    e.record()
                    e.synchronize()
                param_trans_pipe.send(mod_list[0])
