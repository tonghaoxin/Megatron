"""
    MultiThreadHybridAdam used for cpu adam and gpu adam
"""
import pathlib
import os
import torch
from megatron import get_args
from apex.multi_tensor_apply import multi_tensor_applier

class MultiThreadHybridAdam(torch.optim.Optimizer):
    """
    MultiThreadHybridAdam: An object that contains cpu adam and gpu adam.
    """

    optimizer_id = 0

    def __init__(self,
                 param_groups,
                 lr=1e-03,
                 bias_correction=True,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0,
                 amsgrad=False,
                 adamw_mode=True,
                 set_grad_none=True):
        """
        """

        default_args = dict(lr=lr,
                            betas=betas,
                            eps=eps,
                            weight_decay=weight_decay,
                            bias_correction=bias_correction,
                            amsgrad=amsgrad)
        super(MultiThreadHybridAdam, self).__init__(param_groups, default_args)
        args = get_args() 
 
        self.opt_id = MultiThreadHybridAdam.optimizer_id
        MultiThreadHybridAdam.optimizer_id = MultiThreadHybridAdam.optimizer_id + 1
        self.adam_w_mode = adamw_mode
        self.fp32_optimizer_states = True

        
        self.memory_tracer = False
        self.hybrid_adam_param_groups_init = False
   
        self.total_cpu_flat_params = 0
        self.cpu_have_optmizer = False
        self.cuda_have_optmizer = False
      
        load_dir = pathlib.Path(__file__).parent.absolute()
        buildpath = load_dir / 'builder'
        library_path = load_dir / 'cpu_adam.so'

        if os.path.exists(buildpath):
            from megatron.hybrid_adam.builder.cpu_adam_builder import CPUAdamBuilder
            self.ds_opt_adam = CPUAdamBuilder().load()
        else:
            from . import cpu_adam
            self.ds_opt_adam = cpu_adam

        self.ds_opt_adam.adam_enable_adam_threads(True)
        if args.cpu_adam_threads:
            self.ds_opt_adam.adam_set_cpu_adam_threads(args.cpu_adam_threads) 
        self.ds_opt_adam.create_adam(self.opt_id, lr, betas[0], betas[1], eps, weight_decay, adamw_mode,
                                     False) 

        self.cuda_step = 0
        self.set_grad_none = set_grad_none
        if multi_tensor_applier.available:
            import amp_C
            # Skip buffer
            self._dummy_overflow_buf = torch.cuda.IntTensor([0])
            self.multi_tensor_adam = amp_C.multi_tensor_adam
        else:
            raise RuntimeError('apex.optimizers.FusedAdam requires cuda extensions')

    def set_optimizer_state(self, cpu_have_optmizer, cuda_have_optmizer):
        """
        set optimizer_state of cpu_adam and fuse_adam.

        Args:
            cpu_have_optmizer: Are there any parameters in cpu memory.
            cuda_have_optmizer: Are there any parameters in cuda memory.
        """
        self.cpu_have_optmizer = cpu_have_optmizer
        self.cuda_have_optmizer = cuda_have_optmizer
        self.hybrid_adam_param_groups_init = True

    def get_optimizer_state(self):
        """
        get optimizer_state of cpu_adam and fuse_adam.
        """
        cpu_have_optmizer = False
        cuda_have_optmizer = False

        for group_id, group in enumerate(self.param_groups):
            for param_id, p in enumerate(group['params']):
                if p.device == torch.device('cpu'):
                    cpu_have_optmizer = True
                    self.total_cpu_flat_params = self.total_cpu_flat_params + 1
                else:
                    cuda_have_optmizer = True
        return cpu_have_optmizer, cuda_have_optmizer

    @torch.no_grad()
    def step(self, copy_cpu_grads, fp16_param_groups, closure=None):
        """
        Update the model parameters.
        Args:
            closure (callable, optional): closure to compute the loss.
                Defaults to ``None``.
            fp16_param_groups: FP16 GPU parameters to update. Performing the
                copy here reduces communication time. Defaults to ``None``.

        Returns:
            loss: if ``closure`` is provided. Otherwise ``None``.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        args = get_args()
        if not self.hybrid_adam_param_groups_init:
            cpu_have_optmizer, cuda_have_optmizer = self.get_optimizer_state()
            self.set_optimizer_state(cpu_have_optmizer, cuda_have_optmizer)
            self.ds_opt_adam.adam_set_total_flat_params(self.total_cpu_flat_params)

        param_flat_id = 0
        cuda_device = torch.cuda.current_device()        
        for group, grad_group, fp16_group in zip(self.param_groups, copy_cpu_grads, fp16_param_groups):
            bias_correction = 1 if group['bias_correction'] else 0
            beta1, beta2 = group['betas']
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            g_32, p_32, m_32, v_32 = [], [], [], []
            for p, copy_grad, fp16 in zip(group['params'], grad_group, fp16_group):
                state = self.state[p]
                device = torch.device('cpu')

                if p.device == device:
                    p.grad.copy_(copy_grad)
                    if len(state) == 0:
                        state['step'] = 0
                        #use full precision by default unless self.fp32_optimizer_states is off
                        state_dtype = torch.float if self.fp32_optimizer_states else p.dtype

                        # gradient momentums
                        state['exp_avg'] = torch.zeros_like(p.data, dtype=state_dtype, device=device)

                        # gradient variances
                        state['exp_avg_sq'] = torch.zeros_like(p.data, dtype=state_dtype, device=device)
                    if 'step' in state:
                        state['step'] += 1
                    else:
                        state['step'] = 1
                    if args.overlap_offload_optimizer:
                        self.ds_opt_adam.adam_update_async(self.opt_id, group['step'], group['lr'], 
                                                           beta1, beta2, group['eps'], group['weight_decay'], 
                                                           group['bias_correction'], p.data, p.grad.data, 
                                                           state['exp_avg'], state['exp_avg_sq'], fp16.data, 
                                                           copy_grad.data, cuda_device)
                        param_flat_id = param_flat_id + 1
                    else:
                        self.ds_opt_adam.adam_update(self.opt_id, group['step'], group['lr'], beta1, beta2, 
                                                     group['eps'], group['weight_decay'], group['bias_correction'], 
                                                     p.data, p.grad.data, state['exp_avg'], state['exp_avg_sq'])
                else:
                    p.grad = copy_grad 
                    if len(state) == 0:
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p.data)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p.data)

                    g_32.append(p.grad.data)
                    p_32.append(p.data)
                    m_32.append(state['exp_avg'])
                    v_32.append(state['exp_avg_sq'])

            if(len(g_32) > 0):
                # assume same step across group now to simplify things
                # per parameter step can be easily support by making it tensor, or pass list into kernel
                multi_tensor_applier(self.multi_tensor_adam,
                                     self._dummy_overflow_buf,
                                     [g_32, p_32, m_32, v_32],
                                     group['lr'],
                                     beta1,
                                     beta2,
                                     group['eps'],
                                     group['step'],
                                     self.adam_w_mode,
                                     bias_correction,
                                     group['weight_decay'])

        if args.overlap_offload_optimizer and self.cpu_have_optmizer:
            self.ds_opt_adam.adam_wait_async(self.opt_id)
        return loss
