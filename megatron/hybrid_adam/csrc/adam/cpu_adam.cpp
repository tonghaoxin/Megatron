#include "cpu_adam.h"
#include <torch/extension.h>
#include <cassert>
#include <iostream>
#include <memory>
#include <type_traits>
#include <unordered_map>

#if defined(__ENABLE_CUDA__)
#include <cuda_runtime_api.h>
#include "cublas_v2.h"
#include "cuda.h"
#include "curand.h"
#include "custom_cuda_layers.h"
#endif

#include "adam_task.h"
#include "adam_thread.h"

static std::unordered_map<int, std::shared_ptr<void>> s_optimizers;
static std::unordered_map<int, std::shared_ptr<adam_compute_thread>> adam_compute_thread_t;
static std::unordered_map<int, std::shared_ptr<adam_copy_thread>> adam_copy_thread_t;

static int total_flat_params = 0;
static int cpu_adam_threads = 1;

static int global_proc_cuda_device = 0;

static bool start_adam_threads_flag = false;
// C++ interface

void Adam_Optimizer::Step_1(float* _params,
                            float* grads,
                            float* _exp_avg,
                            float* _exp_avg_sq,
                            size_t _param_size,
                            ds_half_precision_t* dev_params,
                            bool half_precision)
{
    size_t rounded_size = 0;
#if defined(__AVX512__) or defined(__AVX256__)
    Step_AVX<1>(&rounded_size,
                _params,
                grads,
                _exp_avg,
                _exp_avg_sq,
                _param_size,
                dev_params,
                half_precision);
#endif
    if (_param_size > rounded_size) {
        float betta1_minus1 = 1 - _betta1;
        float betta2_minus1 = 1 - _betta2;

        float step_size = -1 * _alpha / _bias_correction1;
        float w_decay = -1 * _alpha * _weight_decay;
        ds_half_precision_t* grads_cast_h = nullptr;
        ds_half_precision_t* params_cast_h = nullptr;
        if (half_precision) {
            grads_cast_h = reinterpret_cast<ds_half_precision_t*>(grads);
            params_cast_h = reinterpret_cast<ds_half_precision_t*>(_params);
        }

        for (size_t t = rounded_size; t < _param_size; t += TILE) {
            size_t copy_size = TILE;
            if ((t + TILE) > _param_size) copy_size = _param_size - t;
            size_t offset = copy_size + t;
#if defined(__ENABLE_CUDA__)
            if ((t / TILE) >= 2) { cudaStreamSynchronize(_streams[_buf_index]); }
#endif
#pragma omp parallel for
            for (size_t k = t; k < offset; k++) {
                float grad = half_precision ? (float)grads_cast_h[k] : grads[k];
                float param = half_precision ? (float)params_cast_h[k] : _params[k];
                float momentum = _exp_avg[k];
                float variance = _exp_avg_sq[k];
                if (_weight_decay > 0 && !_adamw_mode) { grad = param * _weight_decay + grad; }
                momentum = momentum * _betta1;
                momentum = grad * betta1_minus1 + momentum;

                variance = variance * _betta2;
                grad = grad * grad;
                variance = grad * betta2_minus1 + variance;

                grad = sqrt(variance);
                grad = grad * _bias_correction2 + _eps;
                grad = momentum / grad;
                if (_weight_decay > 0 && _adamw_mode) { param += w_decay * param; }
                param = grad * step_size + param;
#if defined(__ENABLE_CUDA__)
                if (dev_params) _doubled_buffer[_buf_index][k - t] = param;
#endif
                if (half_precision)
                    params_cast_h[k] = (ds_half_precision_t)param;
                else
                    _params[k] = param;
                _exp_avg[k] = momentum;
                _exp_avg_sq[k] = variance;
            }
#if defined(__ENABLE_CUDA__)
            if (dev_params) {
                launch_param_update(
                    _doubled_buffer[_buf_index], dev_params + t, (copy_size), _streams[_buf_index]);

                _buf_index = !_buf_index;
            }
#endif
        }
    }
}

void Adam_Optimizer::Step_4(float* _params,
                            float* grads,
                            float* _exp_avg,
                            float* _exp_avg_sq,
                            size_t _param_size,
                            ds_half_precision_t* dev_params,
                            bool half_precision)
{
    size_t rounded_size = 0;
#if defined(__AVX512__) or defined(__AVX256__)
    Step_AVX<4>(&rounded_size,
                _params,
                grads,
                _exp_avg,
                _exp_avg_sq,
                _param_size,
                dev_params,
                half_precision);
#endif
    if (_param_size > rounded_size)
        Step_1((_params + rounded_size),
               (grads + rounded_size),
               (_exp_avg + rounded_size),
               (_exp_avg_sq + rounded_size),
               (_param_size - rounded_size),
               (dev_params != nullptr ? (dev_params + rounded_size) : dev_params),
               half_precision);
}

int create_adam_optimizer(int optimizer_id,
                          float alpha = 1e-3,
                          float betta1 = 0.9,
                          float betta2 = 0.999,
                          float eps = 1e-8,
                          float weight_decay = 0,
                          bool adamw_mode = true,
                          bool should_log = false)
{
    auto opt =
        std::make_shared<Adam_Optimizer>(alpha, betta1, betta2, eps, weight_decay, adamw_mode);
    s_optimizers[optimizer_id] = opt;

    if (start_adam_threads_flag == true) {
        auto compute_thread = std::make_shared<adam_compute_thread>(cpu_adam_threads);
        adam_compute_thread_t[optimizer_id] = compute_thread;

        auto copy_thread = std::make_shared<adam_copy_thread>();
        adam_copy_thread_t[optimizer_id] = copy_thread;
    }
    if (should_log) {
        std::string avx_type = "";
#if defined(__AVX512__)
        avx_type = "AVX512";
#else
#if defined(__AVX256__)
        avx_type = "AVX2";
#else
        avx_type = "scalar";
#endif
#endif

        printf("Adam Optimizer #%d is created with %s arithmetic capability.\n",
               optimizer_id,
               avx_type.c_str());
        printf("Config: alpha=%f, betas=(%f, %f), weight_decay=%f, adam_w=%d\n",
               alpha,
               betta1,
               betta2,
               weight_decay,
               (int)adamw_mode);
    }

    return 0;
}

void Adam_Optimizer::Step_8(float* _params,
                            float* grads,
                            float* _exp_avg,
                            float* _exp_avg_sq,
                            size_t _param_size,
                            ds_half_precision_t* dev_params,
                            bool half_precision)
{
    size_t rounded_size = 0;
#if defined(__AVX512__) or defined(__AVX256__)
    Step_AVX<8>(&rounded_size,
                _params,
                grads,
                _exp_avg,
                _exp_avg_sq,
                _param_size,
                dev_params,
                half_precision);
#endif
    if (_param_size > rounded_size)
        Step_4((_params + rounded_size),
               (grads + rounded_size),
               (_exp_avg + rounded_size),
               (_exp_avg_sq + rounded_size),
               (_param_size - rounded_size),
               (dev_params != nullptr ? (dev_params + rounded_size) : dev_params),
               half_precision);
}

int ds_adam_step(int optimizer_id,
                 size_t step,
                 float lr,
                 float beta1,
                 float beta2,
                 float epsilon,
                 float weight_decay,
                 bool bias_correction,
                 torch::Tensor& params,
                 torch::Tensor& grads,
                 torch::Tensor& exp_avg,
                 torch::Tensor& exp_avg_sq)
{
    auto params_c = params.contiguous();
    auto grads_c = grads.contiguous();
    auto exp_avg_c = exp_avg.contiguous();
    auto exp_avg_sq_c = exp_avg_sq.contiguous();

    assert(params.options().dtype() == grads.options().dtype());

    float* params_ptr = (float*)params_c.data_ptr();
    float* grads_ptr = (float*)grads_c.data_ptr();
    float* exp_avg_ptr = (float*)exp_avg_c.data_ptr();
    float* exp_avg_sq_ptr = (float*)exp_avg_sq_c.data_ptr();

    std::shared_ptr<Adam_Optimizer> opt =
        std::static_pointer_cast<Adam_Optimizer>(s_optimizers[optimizer_id]);
    opt->IncrementStep(step, beta1, beta2);
    opt->update_state(lr, epsilon, weight_decay, bias_correction);

    opt->Step_8(params_ptr,
                grads_ptr,
                exp_avg_ptr,
                exp_avg_sq_ptr,
                params_c.numel(),
                nullptr,
                (params.options().dtype() == at::kHalf));

#if defined(__ENABLE_CUDA__)
    opt->SynchronizeStreams();
#endif
    return 0;
}

void cpu_adam_process_async(int optimizer_id, size_t step, float lr, float beta1,
                      float beta2, float epsilon, float weight_decay, 
                      bool bias_correction, bool half_precision, size_t param_size,
                      float* params_ptr, float* grads_ptr, float* exp_avg_ptr, 
                      float* exp_avg_sq_ptr, ds_half_precision_t*gpu_params_ptr,float*gpu_tmp_grads_ptr)
{
    float *params_ptr_start = params_ptr;
    std::shared_ptr<Adam_Optimizer> opt =
        std::static_pointer_cast<Adam_Optimizer>(s_optimizers[optimizer_id]);
    assert(opt != nullptr);
    opt->IncrementStep(step, beta1, beta2);
    opt->update_state(lr, epsilon, weight_decay, bias_correction);

    opt->Step_8(params_ptr,
                grads_ptr,
                exp_avg_ptr,
                exp_avg_sq_ptr,
                param_size,
                nullptr,
                half_precision);

#if defined(__ENABLE_CUDA__)
    opt->SynchronizeStreams();
#endif
   if (gpu_params_ptr != nullptr){
       std::shared_ptr<adam_copy_thread> copy_thread =
          std::static_pointer_cast<adam_copy_thread>(adam_copy_thread_t[optimizer_id]);
       auto task = std::make_shared<adam_copy_task>(params_ptr_start, param_size, gpu_params_ptr,
                                                    grads_ptr,exp_avg_ptr,exp_avg_sq_ptr,gpu_tmp_grads_ptr);
       copy_thread->submit(task);
   }
}

int ds_adam_step_async(int optimizer_id,
                 size_t step,
                 float lr,
                 float beta1,
                 float beta2,
                 float epsilon,
                 float weight_decay,
                 bool bias_correction,
                 torch::Tensor& params,
                 torch::Tensor& grads,
                 torch::Tensor& exp_avg,
                 torch::Tensor& exp_avg_sq,
                 torch::Tensor& gpu_params,
                 torch::Tensor& gpu_tmp_grads,
                 int cuda_device)
{
    global_proc_cuda_device = cuda_device;
    auto gpu_tmp_grads_c = gpu_tmp_grads.contiguous();
    auto params_c = params.contiguous();
    auto gpu_params_c = gpu_params.contiguous();
    auto grads_c = grads.contiguous();
    auto exp_avg_c = exp_avg.contiguous();
    auto exp_avg_sq_c = exp_avg_sq.contiguous();
    size_t param_size = params_c.numel();
    bool half_precision = (params.options().dtype() == at::kHalf);

    assert(params.options().dtype() == grads.options().dtype());

    float* gpu_tmp_grads_ptr = (float*)gpu_tmp_grads_c.data_ptr();
    float* params_ptr = (float*)params_c.data_ptr();
    float* grads_ptr = (float*)grads_c.data_ptr();
    float* exp_avg_ptr = (float*)exp_avg_c.data_ptr();
    float* exp_avg_sq_ptr = (float*)exp_avg_sq_c.data_ptr();
    ds_half_precision_t* gpu_params_ptr = (ds_half_precision_t*)gpu_params_c.data_ptr();

    std::shared_ptr<adam_compute_thread> compute_thread =
        std::static_pointer_cast<adam_compute_thread>(adam_compute_thread_t[optimizer_id]);

    auto task = std::make_shared<adam_compute_task>(optimizer_id, step, lr, beta1, beta2, epsilon,
                                                    weight_decay, bias_correction, half_precision, param_size,
                                                    params_ptr, grads_ptr, exp_avg_ptr, exp_avg_sq_ptr, gpu_params_ptr,gpu_tmp_grads_ptr); 
    compute_thread->submit(task);
    return 0;
}

int ds_adam_step_plus_copy(int optimizer_id,
                           size_t step,
                           float lr,
                           float beta1,
                           float beta2,
                           float epsilon,
                           float weight_decay,
                           bool bias_correction,
                           torch::Tensor& params,
                           torch::Tensor& grads,
                           torch::Tensor& exp_avg,
                           torch::Tensor& exp_avg_sq,
                           torch::Tensor& gpu_params)
{
#if defined(__ENABLE_CUDA__)
    auto params_c = params.contiguous();
    auto gpu_params_c = gpu_params.contiguous();
    auto exp_avg_c = exp_avg.contiguous();
    auto exp_avg_sq_c = exp_avg_sq.contiguous();
    auto grads_c = grads.contiguous();

    float* params_ptr = (float*)params_c.data_ptr();
    float* grads_ptr = (float*)grads_c.data_ptr();
    ds_half_precision_t* gpu_params_ptr = (ds_half_precision_t*)gpu_params_c.data_ptr();
    float* exp_avg_ptr = (float*)exp_avg_c.data_ptr();
    float* exp_avg_sq_ptr = (float*)exp_avg_sq_c.data_ptr();

    std::shared_ptr<Adam_Optimizer> opt =
        std::static_pointer_cast<Adam_Optimizer>(s_optimizers[optimizer_id]);
    opt->IncrementStep(step, beta1, beta2);
    opt->update_state(lr, epsilon, weight_decay, bias_correction);
    opt->Step_8(params_ptr,
                grads_ptr,
                exp_avg_ptr,
                exp_avg_sq_ptr,
                params_c.numel(),
                gpu_params_ptr,
                (params.options().dtype() == at::kHalf));

    opt->SynchronizeStreams();
#else
    assert(false);
#endif
    return 0;
}

int set_total_flat_params(int totals)
{
    total_flat_params = totals;  
    return 0;
}

int get_total_flat_params()
{
    return total_flat_params;
}


int get_cuda_device()
{
    return global_proc_cuda_device;
}

int set_cuda_device(int cuda_device)
{
    global_proc_cuda_device = cuda_device;
    return 0;
}

int set_cpu_adam_threads(int threads)
{
    cpu_adam_threads = threads;
    return 0;
}

int enable_adam_threads(bool enable)
{
    start_adam_threads_flag = enable;
    return 0;
}

int wait_async(int optimizer_id)
{
    std::shared_ptr<adam_compute_thread> compute_thread =
        std::static_pointer_cast<adam_compute_thread>(adam_compute_thread_t[optimizer_id]);
    compute_thread->wait_thread_complated();
    std::shared_ptr<adam_copy_thread> copy_thread =
        std::static_pointer_cast<adam_copy_thread>(adam_copy_thread_t[optimizer_id]);
    copy_thread->wait_thread_complated();
    return 0;
}

int destroy_adam_optimizer(int optimizer_id)
{
    s_optimizers.erase(optimizer_id);

    return 0;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("adam_update", &ds_adam_step, "Megatron CPU Adam update (C++)");
    m.def("adam_update_copy",
          &ds_adam_step_plus_copy,
          "Megatron CPU Adam update and param copy (C++)");
    m.def("adam_update_async", &ds_adam_step_async, "Megatron CPU Adam update (C++)");
    m.def("adam_set_total_flat_params", &set_total_flat_params, "Megatron CPU Adam update (C++)");
    m.def("adam_set_cpu_adam_threads", &set_cpu_adam_threads, "Megatron CPU Adam update (C++)");
    m.def("adam_enable_adam_threads", &enable_adam_threads, "Megatron CPU Adam update (C++)");
    m.def("adam_wait_async", &wait_async, "Megatron CPU Adam update (C++)");
    m.def("create_adam", &create_adam_optimizer, "Megatron CPU Adam (C++)");
    m.def("destroy_adam", &destroy_adam_optimizer, "Megatron CPU Adam destroy (C++)");
}
