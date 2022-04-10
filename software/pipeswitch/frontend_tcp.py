import threading
import struct

from util.util import timestamp

class FrontendTcpThd(threading.Thread):
    def __init__(self, qout, agent):
        super(FrontendTcpThd, self).__init__()
        self.qout = qout
        self.agent = agent

    def run(self):
        while True:
            timestamp('tcp', 'listening')
            
            task_name_length_b = self.agent.recv(4)
            task_name_length = struct.unpack('I', task_name_length_b)[0]
            if task_name_length == 0:
                break
            task_name_b = self.agent.recv(task_name_length)
            task_name = task_name_b.decode()
            timestamp('tcp', 'get_task_name')

            data_length_b = self.agent.recv(4)
            data_length = struct.unpack('I', data_length_b)[0]
            data_b = self.agent.recv(data_length)
            data_name = data_b.decode()
            timestamp('tcp', 'get_data_name')

            layer_length_b = self.agent.recv(4)
            layer_length = struct.unpack('I', layer_length_b)[0]
            layer_b = self.agent.recv(layer_length)
            num_layers = layer_b.decode()
            timestamp('tcp', 'get_num_layers')

            self.qout.put((self.agent, task_name, data_name, num_layers))
            timestamp('tcp', 'enqueue_request')
