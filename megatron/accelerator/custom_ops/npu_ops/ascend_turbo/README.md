- AscendSpeed安装：

STEP 1：在 ascendspeed/initialize.py 中按照如下代码段进行修改

```python
from ascendspeed.mpu.ascend_turbo.initialize import initialize_cfg_from_args

# ...

def initialize_megatron(extra_args_provider=None, args_defaults=None,
                        ignore_unknown_args=False, allow_no_cuda=False):
    # ...
    def finish_mpu_init():
        # ...
        _set_random_seed(args.seed)
        initialize_cfg_from_args(args)
```

STEP 2：将 ascend_turbo 文件夹放在 ascendspeed/mpu 下；

STEP 3：将 ascend_turbo 中 initialize.py 文件中 import 库的部分对应代码改成如下代码；

```python
from ascendspeed.mpu import ColumnParallelLinear, RowParallelLinear
from ascendspeed.core.parallel_state import (
   get_tensor_model_parallel_group,
   get_tensor_model_parallel_world_size
)
```

- Megatron 安装

STEP 1：在 megatron/initialize.py 按照如下代码段进行修改

```python
from megatron.core.tensor_parallel.ascend_turbo.initialize import initialize_cfg_from_args

def initialize_megatron(extra_args_provider=None, args_defaults={},
                        ignore_unknown_args=False, allow_no_cuda=False):
    def finish_mpu_init():
        ...
        _set_random_seed(args.seed, args.data_parallel_random_init)
        initialize_cfg_from_args(args)
```

STEP 2：将 ascend_turbo 文件夹放在 megatron/core/tensor_parallel/ 下；

STEP 3：将 ascend_turbo 中 initialize.py 文件中 import 库的部分对应代码改成如下代码；

```python
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_world_size
)
```
### 环境变量设置

在使用计算通信并行时，需要开启FFTS和内存复用，分别对应如下这两个环境变量的设置。

```shell
export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE # 开启FFTS
export MULTI_STREAM_MEMORY_REUSE=1 # 内存复用

```
