from queue import Queue
from multiprocessing import Process
import torch
import time

from cognn.worker_common import ModelSummary
from util.util import timestamp

class WorkerProc(Process):
    def __init__(self, model_list, pipe, param_trans_pipe):
        super(WorkerProc, self).__init__()
        self.model_list = model_list
        self.pipe = pipe
        self.param_trans_pipe = param_trans_pipe
        
    def run(self):
        timestamp('worker', 'start')

        # Warm up CUDA and get shared cache
        torch.randn(1024, device='cuda')
        time.sleep(1)
        torch.cuda.recv_shared_cache() # pylint: disable=no-member
        timestamp('worker', 'share_gpu_memory')
        
        while True:  # dispatch workers for task execution
            agent, model_name, para_cache_info, comp_cache_info = self.pipe.recv()
            model_summary = ModelSummary(model_name, para_cache_info, comp_cache_info, self.param_trans_pipe)
            timestamp('worker', 'import models')
            timestamp('worker', 'after importing models')

            # start doing training
            with torch.cuda.stream(model_summary.cuda_stream_for_computation):
                output = model_summary.execute()
                print('output: {}'.format(output))
#                print ('Training time: {} ms'.format(output))
                del output

                self.pipe.send('FNSH')
                agent.send(b'FNSH')
                torch.cuda.clear_shared_cache()

            timestamp('worker_comp_thd', 'complete')
            model_summary.reset_initialized(model_summary.model)
