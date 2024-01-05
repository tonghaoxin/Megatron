#ifndef ADAM_TASK_H
#define ADAM_TASK_H

#include<queue>
#include<future>
#include<utility>
#include<iostream>
#include<cpu_adam.h>
#include "custom_cuda_layers.h"

class adam_task{
public:
    int done_tasks;
    int all_tasks;
    virtual void process() = 0;
};

class adam_compute_task: public adam_task{
public:
    adam_compute_task(int optimizer_id, size_t step, float lr, float beta1,
                      float beta2, float epsilon, float weight_decay, 
                      bool bias_correction, bool half_precision, size_t param_size,
                      float* params_ptr, float* grads_ptr, float* exp_avg_ptr, 
                      float* exp_avg_sq_ptr,ds_half_precision_t* gpu_params_ptr,float*gpu_tmp_grads_ptr);
    ~adam_compute_task();
    void process();
private:
    int optimizer_id;
    size_t step;
    float lr;
    float beta1;
    float beta2;
    float epsilon;
    float weight_decay;
    bool bias_correction;
    bool half_precision;
    size_t param_size;
    float* params_ptr;
    float* grads_ptr;
    float* exp_avg_ptr;
    float* exp_avg_sq_ptr;
    ds_half_precision_t* gpu_params_ptr;
    float*gpu_tmp_grads_ptr;
    int done_tasks;
    int all_tasks;
};

class adam_copy_task: public adam_task{
public:
    int done_tasks;
    int all_tasks;
    adam_copy_task(float* _params, size_t _param_size, ds_half_precision_t* gpu_params_ptr,
                   float* grads,float* _exp_avg,float* _exp_avg_sq,float*gpu_tmp_grads_ptr);
    ~adam_copy_task();
    void process();
private:
    size_t _param_size;
    float* _params;
    ds_half_precision_t* gpu_params_ptr;

    float* grads;
    float* _exp_avg;
    float* _exp_avg_sq;
    float*gpu_tmp_grads_ptr;
    cudaStream_t stream;
};

#endif
