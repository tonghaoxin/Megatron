#ifndef ADAM_THREAD_H
#define ADAM_THREAD_H
#include <thread>
#include <mutex>
#include <queue>
#include <vector>
#include <condition_variable>
#include "adam_task.h"
#include "cpu_adam.h"

class adam_thread{
public:
    virtual void start() = 0;
    virtual void stop() = 0;
    virtual void worker() = 0;
    virtual void wait_thread_complated() = 0;
  
private:
    int total_tasks = 0;
    int done_tasks = 0;
    bool done = false;
};

class adam_compute_thread: public adam_thread {
public:
    adam_compute_thread(int num);
    ~adam_compute_thread();
    void submit(std::shared_ptr<adam_compute_task> task);
    void start();
    void stop();
    void worker();
    void release();
    void wait_thread_complated();
private:
    std::vector<std::thread*> threads;
    std::queue<std::shared_ptr<adam_compute_task>> compute_task_queue;
    std::mutex mu;
    std::condition_variable cond;
    int num;
    int total_tasks = 0;
    int done_tasks = 0;
    bool done = false;
    bool step_done = false;
    bool step_process = false;
    int submit_tasks = 0;
};

class adam_copy_thread: public adam_thread {
public:
    void submit(std::shared_ptr<adam_copy_task> task);
    adam_copy_thread();
    ~adam_copy_thread();
    void start();
    void stop();
    void worker();
    void release();
    void wait_thread_complated();
private:
    std::vector<std::thread*> threads;
    std::queue<std::shared_ptr<adam_copy_task>> copy_task_queue;
    std::mutex mu;
    std::condition_variable cond;
    int num;
    int total_tasks = 0;
    int done_tasks = 0;
    bool done = false;
    bool step_done = false;
};

#endif
