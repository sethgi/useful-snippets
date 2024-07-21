"""
file: torch_worker_pool.py
author: Seth Isaacson (sethgi@umich.edu, sethisaacson.me)
Description: Allows parallelizing arbitrary pytorch tasks across multiple GPUs.
It's useful in the case that you want to call one function many times on a lot of data.
It works by implementing a simple pool of workers, one per GPU. Each worker 
pulls tasks off a job queue, and puts the results in the result queue.
First, implement my_function to do the work you're trying to split across GPUs. 
Note that my_function should always put data on GPU 0, since this works by setting
CUDA_VISIBLE_DEVICES.
Then, populate all_function_arg_sets with some data. Each list element should correspond
to one job to be done.
You can also populate additional_args and additional_kwargs with any arguments that are
shared across all jobs.
"""


import torch.multiprocessing as mp
import torch
import os


# This script is really optimized for expensive functions, such as rendering an image from a NeRF or even training
# a model. Here, just a trivial task is shown. But you could imagine calling an entire training routine here. 
def my_function(tensor1: torch.Tensor, tensor2: torch.Tensor, offset_arg, offset_kwarg=None) -> torch.Tensor:
    # Move data to GPU 0 here... it'll actually move to whatever GPU this thread is assigned
    return (tensor1.to('cuda:0') + tensor2.to('cuda:0')).sum().item() + offset_arg + offset_kwarg

# Used to tell workers the queue is empty and they can exit
class StopSignal:
    pass

# Runs 
def _gpu_worker(fn_handle, job_queue: mp.Queue, result_queue: mp.Queue, *additional_args, **additional_kwargs):
    while not job_queue.empty():
        data = job_queue.get()
        if isinstance(data, StopSignal):
            result_queue.put(StopSignal())
            break
        
        output_data = fn_handle(*data, *additional_args, **additional_kwargs)        
        result_queue.put(output_data)
    
    # Hang until the process is terminated
    while True:
        continue

def run_pool(fn_handle, job_data, gpu_ids = None, additional_args = (), additional_kwargs = {}):
    job_queue = mp.Queue()
    result_queue = mp.Queue()
    
    if gpu_ids is None:
        gpu_ids = len(range(torch.cuda.get_device_count()))
       
    num_gpus = len(gpu_ids)
    
    # Load the jobs into the queue
    for args in job_data:
        job_queue.put(args)
        
    # One None corresponds to sending a stop signal to one of the workers
    for _ in range(num_gpus):
        job_queue.put(StopSignal())
        
    # Create and start the processes
    gpu_worker_processes = []
    for gpu_id in gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        gpu_worker_processes.append(mp.Process(target = _gpu_worker, args=(fn_handle, job_queue, result_queue, *additional_args), kwargs=additional_kwargs))
        gpu_worker_processes[-1].start()
    
    # Wait for everything to start
    stop_recv = 0
    results = []
    while stop_recv < num_gpus:
        result = result_queue.get()
        if isinstance(result, StopSignal):
            stop_recv += 1
            continue
        results.append(result)
    # Sync
    for process in gpu_worker_processes:
        process.terminate()
        
    return results

if __name__ == "__main__":
    ### This is very important!!!
    mp.set_start_method('spawn')
    
    # One list element corresponds to one job
    all_function_arg_sets = []
    
    # Args to pass to the GPU workers
    additional_args = (17,)
    additional_kwargs = {"offset_kwarg": 14}
    
    for _ in range(100):
        all_function_arg_sets.append((torch.rand(10,), torch.rand(10,)))
    
    results = run_pool(my_function, all_function_arg_sets, None, additional_args, additional_kwargs)
    print([r for r in results])