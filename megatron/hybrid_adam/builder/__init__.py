"""
    Used for compiling to the dynamic library of cpu_adam.so
"""
import sys
import os
import pkgutil
import importlib

# Do not remove, required for abstract accelerator to detect if we have a megatron or 3p op_builder
__megatron__ = True

# List of all available op builders from megatron op_builder
try:
    import megatron.hybrid_adam.builder  # noqa: F401
    op_builder_dir = "megatron.hybrid_adam.builder"
except ImportError:
    op_builder_dir = "builder"

__op_builders__ = []

this_module = sys.modules[__name__]

class_dict = {}

def _lazy_init_class_dict():
    """
    Import the module of CpuAdamBuilder.
    """
    if len(class_dict) != 0:
        return
    else:
        op_builder_module = importlib.import_module(op_builder_dir)
        for _, module_name, _ in pkgutil.iter_modules([os.path.dirname(op_builder_module.__file__)]):
            if module_name != 'all_ops' and module_name != 'builder' and module_name != 'cpu':
                module = importlib.import_module("{}.{}".format(op_builder_dir, module_name))
                for member_name in module.__dir__():
                    if member_name.endswith(
                        'Builder'
                    ) and member_name != "OpBuilder" and \
                      member_name != "CUDAOpBuilder" and member_name != "TorchCPUOpBuilder": 
                        if not member_name in class_dict:
                            class_dict[member_name] = getattr(module, member_name)



def get_op_builder(class_name):
    """
    Get the module of CpuAdamBuilder.
    """
    _lazy_init_class_dict()
    if class_name in class_dict:
        return class_dict[class_name]
    else:
        return None

# reflect builder names and add builder closure, such as 'CPUBuilder()' creates op builder wrt current accelerator
for _, module_name, _ in pkgutil.iter_modules([os.path.dirname(this_module.__file__)]):
    if module_name != 'all_ops' and module_name != 'builder':
        module = importlib.import_module(f".{module_name}", package=op_builder_dir)
        for member_name in module.__dir__():
            if member_name.endswith('Builder') and member_name != "OpBuilder" and member_name != "CUDAOpBuilder":
                # assign builder name to variable with same name
                # the following is equivalent to i.e. TransformerBuilder = "TransformerBuilder"
                this_module.__dict__[member_name] = get_op_builder(member_name)
