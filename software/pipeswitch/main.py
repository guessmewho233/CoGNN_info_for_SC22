import sys
import time
from queue import Queue

import torch
import torch.multiprocessing as mp

from cognn.frontend_tcp import FrontendTcpThd
from cognn.frontend_schedule import FrontendScheduleThd
from cognn.worker import WorkerProc
from util.util import timestamp, TcpAgent, TcpServer

def main():
    timestamp('frontend', 'start')

    # Load model list (task & data)
    model_list_file_name = sys.argv[1]
    model_list = []
    with open(model_list_file_name) as f:
        for line in f.readlines():
            if len(line.split()) != 3:
                continue
            model_list.append([line.split()[0], line.split()[1], line.split()[2]])

    # Warm up CUDA and allocate shared cache
    torch.randn(1024, device='cuda')
    torch.cuda.allocate_shared_cache()

    # Create workers
    num_workers = 1
    worker_list = []
    for _ in range(num_workers):
        p_parent, p_child = mp.Pipe()
        param_trans_parent, param_trans_child = mp.Pipe()
        worker = WorkerProc(model_list, p_child, param_trans_child)
        worker.start()
        torch.cuda.send_shared_cache()
        worker_list.append((p_parent, worker, param_trans_parent))
        timestamp('frontend', 'create_worker')


    # Create request queue and scheduler thread
    requests_queue = Queue()
    t_sch = FrontendScheduleThd(model_list, requests_queue, worker_list)
    t_sch.start()
    timestamp('frontend', 'start_schedule')

    # Accept connections
    server = TcpServer('localhost', 12345)
    timestamp('tcp', 'listen')
    while True:
        conn, _ = server.accept()
        agent = TcpAgent(conn)
        timestamp('tcp', 'connected')
        t_tcp = FrontendTcpThd(requests_queue, agent)
        t_tcp.start()

    # Wait for end
    t_sch.join()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
