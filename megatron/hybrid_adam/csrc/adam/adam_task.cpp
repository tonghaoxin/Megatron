#include "adam_task.h"
#include "cpu_adam.h"
#include <iostream>
#include <fstream>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include "cuda.h"
#include "custom_cuda_layers.h"
#include "simd.h"
#include <unistd.h>

adam_compute_task::adam_compute_task(int optimizer_id, size_t step, float lr, float beta1,
                      float beta2, float epsilon, float weight_decay, 
                      bool bias_correction, bool half_precision ,size_t param_size,
                      float* params_ptr, float* grads_ptr, float* exp_avg_ptr, 
                      float* exp_avg_sq_ptr, ds_half_precision_t* gpu_params_ptr,float*gpu_tmp_grads_ptr)
    : optimizer_id(optimizer_id),
      step(step),
      lr(lr),
      beta1(beta1),
      beta2(beta2),
      epsilon(epsilon),
      weight_decay(weight_decay),
      bias_correction(bias_correction),
      half_precision(half_precision),
      param_size(param_size),
      params_ptr(params_ptr),
      grads_ptr(grads_ptr),
      exp_avg_ptr(exp_avg_ptr),
      exp_avg_sq_ptr(exp_avg_sq_ptr),
      gpu_params_ptr(gpu_params_ptr),
      gpu_tmp_grads_ptr(gpu_tmp_grads_ptr) 
{
 
}

adam_compute_task::~adam_compute_task()
{

}

void adam_compute_task::process()
{
    cpu_adam_process_async(optimizer_id, step, lr, beta1, beta2, epsilon, weight_decay,
                       bias_correction, half_precision, param_size,
                       params_ptr, grads_ptr, exp_avg_ptr, exp_avg_sq_ptr, gpu_params_ptr,gpu_tmp_grads_ptr);
}


adam_copy_task::adam_copy_task(float* _params, size_t _param_size, ds_half_precision_t* gpu_params_ptr,
                               float* grads,float* _exp_avg,float* _exp_avg_sq,float*gpu_tmp_grads_ptr)
:_params(_params),_param_size(_param_size),gpu_params_ptr(gpu_params_ptr),
 grads(grads),_exp_avg(_exp_avg),_exp_avg_sq(_exp_avg_sq),gpu_tmp_grads_ptr(gpu_tmp_grads_ptr)
{
    stream = TrainingContext::Instance().GetNewStream();
}

adam_copy_task::~adam_copy_task()
{

}

void adam_copy_task::process()
{
    size_t new_rounded_size = 0;
    new_rounded_size = _param_size;
    for (size_t t = 0; t < new_rounded_size; t += TILE) {
        size_t copy_size = TILE;
        if ((t + TILE) > new_rounded_size) copy_size = new_rounded_size - t;
        size_t offset = copy_size + t;
        if ((t / TILE) >= 2) { cudaStreamSynchronize(stream); }
        
        launch_param_update(
           _params + t, gpu_params_ptr + t, copy_size, stream);
    }

    cudaStreamSynchronize(stream);
}

