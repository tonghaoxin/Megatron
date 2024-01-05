#include"adam_thread.h"
#include<thread>
#include<future>
#include<queue>
#include<vector>
#include<mutex>
#include<iostream>
#include <fstream>
#include <stdlib.h>
#include <unistd.h>

using namespace std;

adam_compute_thread::adam_compute_thread(int num):threads(0),compute_task_queue(),mu(),num(num)
{
    start();   
}

void adam_compute_thread::start()
{
    std::lock_guard<std::mutex> lk(mu);
    while(threads.size()<(unsigned int)num){
        std::thread* t = new std::thread(&adam_compute_thread::worker,this);
        threads.push_back(t);
    }
}

void adam_compute_thread::worker()
{
    std::unique_lock<std::mutex> lk(mu);
    step_process = false;
    while(!done){
        if(compute_task_queue.empty()){
            cond.wait(lk);
            step_process = true;
            continue;
        }    
        cudaSetDevice(get_cuda_device());
        shared_ptr<adam_compute_task> task=compute_task_queue.front();
        compute_task_queue.pop();
        if(task){
            lk.unlock();
            task->process();
            lk.lock();
            done_tasks++;
            if(done_tasks == get_total_flat_params()){
                step_done = true;
	        done_tasks = 0;
                step_process = false;
                cond.notify_all();
            }
        }

    }

}


void adam_compute_thread::submit(shared_ptr<adam_compute_task> task)
{
    std::lock_guard<std::mutex> lk(mu);
    compute_task_queue.push(task);
    cond.notify_all();
}

void adam_compute_thread::stop()
{
    std::lock_guard<std::mutex> lk(mu);
    done = true;
    cond.notify_all();
}

void adam_compute_thread::wait_thread_complated() 
{
    std::unique_lock<std::mutex> lk(mu);
    while(!step_done){
        if(done_tasks < get_total_flat_params()){
            cond.wait(lk);
            continue;
        }
    }
    step_done = false;
    submit_tasks = 0;
}

adam_compute_thread::~adam_compute_thread()
{
    stop();
    for(int i=0;i<num;i++){
        thread* t=threads.back();
        threads.pop_back();
        t->join();
        delete t;
    }
}


adam_copy_thread::adam_copy_thread():threads(0),copy_task_queue(),mu()
{
    start();
}

void adam_copy_thread::start()
{
    std::lock_guard<std::mutex> lk(mu);
    std::thread* t = new std::thread(&adam_copy_thread::worker,this);
    threads.push_back(t);
}

void adam_copy_thread::worker()
{
    std::unique_lock<std::mutex> lk(mu);
    while(!done){
        if(copy_task_queue.empty()){
            cond.wait(lk);
            continue;
        }
        cudaSetDevice(get_cuda_device());
        shared_ptr<adam_copy_task> task=copy_task_queue.front();
        copy_task_queue.pop();
        if(task){
            lk.unlock();
            task->process();
            lk.lock();

            done_tasks++;
            if(done_tasks == get_total_flat_params()){
                step_done = true;
                done_tasks = 0;
                cond.notify_all();
            }
        }
    }
}

void adam_copy_thread::submit(shared_ptr<adam_copy_task> task)
{
    std::lock_guard<std::mutex> lk(mu);
    copy_task_queue.push(task);
    cond.notify_all();
}

void adam_copy_thread::stop()
{
    std::lock_guard<std::mutex> lk(mu);
    done = true;
    cond.notify_all();
}

void adam_copy_thread::wait_thread_complated() 
{
    std::unique_lock<std::mutex> lk(mu);
    while(!step_done){
        if(done_tasks < get_total_flat_params()){
            cond.wait(lk);
            continue;
        }
    }
    step_done = false;
}

adam_copy_thread::~adam_copy_thread()
{
    stop();
    
    thread* t=threads.back();
    threads.pop_back();
    t->join();
    delete t;
}

