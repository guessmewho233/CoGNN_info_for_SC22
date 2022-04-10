import numpy as np
import collections

MEMORY_ALLOCATED = 26 * 1024 * 1024 * 1024
POLICY_OPTION = 0 # 0 BASE (in-order-maximum-memory-consumption), 1 LMCF (lowest-memory-consumption-first), 2 BMC (balanced-memory-consumption)


# in-order maximum memory consumption
def policy_base(num_workers, para_bytes, comp_bytes, job_list):
    reorder_job_list = []
    group_mc, group_counter, element_counter = 0, -1, num_workers
    for i in range(len(job_list)):
        hash_name = hash('{}_{}_{}'.format(job_list[i][1], job_list[i][2], job_list[i][3]))
        total_bytes = comp_bytes[hash_name] + para_bytes[hash_name]
        group_mc += total_bytes
        if group_mc >= MEMORY_ALLOCATED or element_counter == num_workers:
            reorder_job_list.append([])
            group_mc, element_counter = total_bytes, 0
            group_counter += 1
        reorder_job_list[group_counter].append(job_list[i])
        element_counter += 1

    return reorder_job_list


# lowest memory consumption first
def policy_smcf(num_workers, para_bytes, comp_bytes, job_list):
    total_bytes = np.zeros((len(job_list)), dtype='int64')
    for i in range(len(job_list)):
        hash_name = hash('{}_{}_{}'.format(job_list[i][1], job_list[i][2], job_list[i][3]))
        total_bytes[i] = comp_bytes[hash_name] + para_bytes[hash_name]
    ascending = np.argsort(total_bytes)
   
    reorder_job_list = []
    group_mc, group_counter, element_counter = 0, -1, num_workers
    for i in range(ascending.shape[0]): 
        group_mc += total_bytes[ascending[i]]
        if group_mc >= MEMORY_ALLOCATED or element_counter == num_workers:
            reorder_job_list.append([])
            group_mc, element_counter = total_bytes[ascending[i]], 0
            group_counter += 1

        reorder_job_list[group_counter].append(job_list[ascending[i]])
        element_counter += 1

    return reorder_job_list


# balanced memory consumption
def policy_bmc(num_workers, para_bytes, comp_bytes, job_list):
    total_bytes = np.zeros((len(job_list)), dtype='int64')
    for i in range(len(job_list)):
        hash_name = hash('{}_{}_{}'.format(job_list[i][1], job_list[i][2], job_list[i][3]))
        total_bytes[i] = comp_bytes[hash_name] + para_bytes[hash_name]
    ascending = np.argsort(total_bytes)

    ascDeque = collections.deque()
    for i in range(ascending.shape[0]):
        ascDeque.append(ascending[i])
    
    reorder_job_list = []
    group_mc, group_counter, element_counter = 0, -1, num_workers
    for i in range(ascending.shape[0]):
        tmpIdx = ascDeque.pop() if i % 2 == 1 else ascDeque.popleft()
        group_mc += total_bytes[tmpIdx]
        if group_mc >= MEMORY_ALLOCATED or element_counter == num_workers:
            reorder_job_list.append([])
            group_mc, element_counter = total_bytes[tmpIdx], 0
            group_counter += 1

        reorder_job_list[group_counter].append(job_list[tmpIdx])
        element_counter += 1

    return reorder_job_list
