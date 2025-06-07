import os
import torch
import numpy as np
import copy
import math

@torch.no_grad()
def pseudo_quantize_tensor(tensor, n_bits=8, zero_point=False, q_group_size=-1, per_tensor=False, inplace=False):
    """
    The basic quantization function for weight, activation and KV cache.
    """

    org_tensor_shape = tensor.shape
    org_tensor = copy.deepcopy(tensor)
    if q_group_size > 0:
        # q_group_size = min(org_tensor_shape[-1], q_group_size)
        tensor = tensor.reshape(-1, q_group_size)
    if len(org_tensor_shape)==4:
        # per out channel in Conv2d
        tensor = tensor.reshape(tensor.shape[0],-1)
    if per_tensor:
        tensor = tensor.reshape(1, -1)
    assert tensor.dim() == 2
    if zero_point:
        max_val = tensor.amax(dim=1, keepdim=True)
        min_val = tensor.amin(dim=1, keepdim=True)
        max_int = 2**n_bits - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales))
    else:
        max_val = tensor.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (n_bits - 1) - 1
        min_int = -(2 ** (n_bits - 1))
        scales = max_val / max_int
        zeros = 0

    if inplace:
        (
            (tensor.div_(scales).round_().add_(zeros)).clamp_(min_int, max_int).sub_(zeros)
        ).mul_(scales)
    else:
        tensor = (torch.clamp(torch.round(tensor / scales + zeros), min_int, max_int) - zeros) * scales
    
    if torch.isnan(tensor).sum() > 0:
        import ipdb; ipdb.set_trace()
    assert torch.isnan(tensor).sum() == 0

    tensor = tensor.reshape(org_tensor_shape)

    return tensor


@torch.no_grad()
def pseudo_nf_quantize_tensor(tensor, n_bits=4, zero_point=False, q_group_size=-1, per_tensor=False, inplace=False):
    """
    The fpx quantization function for weight, activation.
    """
    org_tensor_shape = tensor.shape
    if q_group_size > 0:
        # q_group_size = min(org_tensor_shape[-1], q_group_size)
        tensor = tensor.reshape(-1, q_group_size)
    if len(org_tensor_shape)==4:
        # per out channel in Conv2d
        tensor = tensor.reshape(tensor.shape[0],-1)
    if per_tensor:
        tensor = tensor.reshape(1, -1)
    assert tensor.dim() == 2

    fp_list = torch.Tensor(np.loadtxt('./quant/format_ref/NF/nf{}.txt'.format(n_bits))).to(tensor.device)

    
    if zero_point:
        max_val = tensor.amax(dim=1, keepdim=True)
        min_val = tensor.amin(dim=1, keepdim=True)
        max_fp = 1
        min_fp = -1
        scales = (max_val - min_val).clamp(min=1e-5) / (max_fp-min_fp)
        zeros = (-((max_val+min_val)/2)/scales)
    else:
        max_val = tensor.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_fp = 1
        min_fp = -1
        scales = max_val / max_fp
        zeros = 0

    tensor = tensor / scales + zeros
    outputs = []
    interval = 1
    num = math.ceil(tensor.shape[0]/interval)
    for i in range(num):
        ind_tensor = tensor[i*interval:min((i+1)*interval,tensor.shape[0])]
        diff = (ind_tensor[...,None]-fp_list).abs()
        index = torch.argmin(diff,-1)
        x = fp_list[None,:].repeat(ind_tensor.shape[0], ind_tensor.shape[1], 1)
        pf_tensor = x.gather(-1, index[...,None]).squeeze(-1)
        if inplace:
            pf_tensor = ((pf_tensor.clamp_(min_fp, max_fp))-zeros) * scales[i*interval:min((i+1)*interval,tensor.shape[0])]
        else:
            pf_tensor = (torch.clamp(pf_tensor, min_fp, max_fp) - zeros) * scales[i*interval:min((i+1)*interval,tensor.shape[0])]

        output_tensor = pf_tensor

        assert torch.isnan(output_tensor).sum() == 0
        outputs.append(output_tensor)

    output_tensor = torch.cat(outputs)
    output_tensor = output_tensor.reshape(org_tensor_shape)

    return output_tensor


@torch.no_grad()
def pseudo_pn_quantize_tensor(tensor, n_bits=4, zero_point=False, q_group_size=-1, per_tensor=False, inplace=False, pn_type='offset'):
    """
    The fpx quantization function for weight, activation.
    """
    org_tensor_shape = tensor.shape
    if q_group_size > 0:
        # q_group_size = min(org_tensor_shape[-1], q_group_size)
        tensor = tensor.reshape(-1, q_group_size)
    if len(org_tensor_shape)==4:
        # per out channel in Conv2d
        tensor = tensor.reshape(tensor.shape[0],-1)
    if per_tensor:
        tensor = tensor.reshape(1, -1)
    assert tensor.dim() == 2
    fp_list = torch.Tensor(np.loadtxt('./quant/format_ref/PN/pn{}_nf_{}.txt'.format(n_bits,pn_type))).to(tensor.device)
    
    if zero_point:
        max_val = tensor.amax(dim=1, keepdim=True)
        min_val = tensor.amin(dim=1, keepdim=True)
        max_fp = 1
        min_fp = -1
        scales = (max_val - min_val).clamp(min=1e-5) / (max_fp-min_fp)
        zeros = (-((max_val+min_val)/2)/scales)
    else:
        max_val = tensor.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_fp = fp_list.max()
        min_fp = fp_list.min()
        scales = max_val/max_fp
        zeros = 0

    tensor = tensor / scales + zeros
    outputs = []
    interval = 1
    num = math.ceil(tensor.shape[0]/interval)
    # import ipdb; ipdb.set_trace()
    for i in range(num):
        ind_tensor = tensor[i*interval:min((i+1)*interval,tensor.shape[0])]
        diff = (ind_tensor[...,None]-fp_list).abs()
        index = torch.argmin(diff,-1)
        x = fp_list[None,:].repeat(ind_tensor.shape[0], ind_tensor.shape[1], 1)
        pf_tensor = x.gather(-1, index[...,None]).squeeze(-1)
        if inplace:
            pf_tensor = ((pf_tensor.clamp_(min_fp, max_fp))-zeros) * scales[i*interval:min((i+1)*interval,tensor.shape[0])]
        else:
            pf_tensor = (torch.clamp(pf_tensor, min_fp, max_fp) - zeros) * scales[i*interval:min((i+1)*interval,tensor.shape[0])]

        output_tensor = pf_tensor

        assert torch.isnan(output_tensor).sum() == 0
        outputs.append(output_tensor)

    output_tensor = torch.cat(outputs)
    output_tensor = output_tensor.reshape(org_tensor_shape)

    return output_tensor


@torch.no_grad()
def quantize_weight_per_channel_absmax(w, n_bits=8, mode='int', model_name=None, module_name=None ,pn_type='offset'):
    """
    The basic quantization function for weight, activation and KV cache.
    """
    if mode=='int':
        tensor = pseudo_quantize_tensor(w, n_bits=n_bits, zero_point=False, q_group_size=-1, per_tensor=False, inplace=False)
    elif mode=='fp':
        tensor = pseudo_fp_quantize_tensor(w, n_bits=n_bits, zero_point=False, q_group_size=-1, per_tensor=False, inplace=False)
    elif mode=='nf':
        tensor = pseudo_nf_quantize_tensor(w, n_bits=n_bits, zero_point=False, q_group_size=-1, per_tensor=False, inplace=False)
    elif mode=='apot':
        tensor = pseudo_apot_quantize_tensor(w, n_bits=n_bits, zero_point=False, q_group_size=-1, per_tensor=False, inplace=False)
    elif mode=='pn':
        tensor = pseudo_pn_quantize_tensor(w, n_bits=n_bits, zero_point=False, q_group_size=-1, per_tensor=False, inplace=False, pn_type=pn_type)
    elif mode=='dynamic_pn':
        tensor = dynamic_pseudo_pn_quantize_tensor(w, n_bits=n_bits, zero_point=False, q_group_size=-1, per_tensor=False, inplace=False, model_name=model_name, module_name=module_name)
    
    return tensor
    
@torch.no_grad()
def quantize_activation_per_token_absmax(t, n_bits=8, mode='int' ,pn_type='offset'):
    t_shape = t.shape
    # t = t.contiguous().view(-1, t_shape[-1])
    t = t.contiguous().view(t_shape[0], -1)
    if mode=='int':
        t = pseudo_quantize_tensor(t, n_bits=n_bits, zero_point=False, q_group_size=-1, per_tensor=False, inplace=False)
    elif mode=='fp':
        t = pseudo_fp_quantize_tensor(t, n_bits=n_bits, zero_point=False, q_group_size=-1, per_tensor=False, inplace=False)
    elif mode=='nf':
        t = pseudo_nf_quantize_tensor(t, n_bits=n_bits, zero_point=False, q_group_size=-1, per_tensor=False, inplace=False)
    elif mode=='pn':
        t = pseudo_pn_quantize_tensor(t, n_bits=n_bits, zero_point=False, q_group_size=-1, per_tensor=False, inplace=False, pn_type=pn_type)
    return t.reshape(t_shape)
    
@torch.no_grad()
def quantize_weight_per_tensor_absmax(w, n_bits=8, mode='int', model_name=None, module_name=None ,pn_type='offset'):
    """
    The basic quantization function for weight, activation and KV cache.
    """
    if mode=='int':
        tensor = pseudo_quantize_tensor(w, n_bits=n_bits, zero_point=False, q_group_size=-1, per_tensor=True, inplace=False)
    elif mode=='fp':
        tensor = pseudo_fp_quantize_tensor(w, n_bits=n_bits, zero_point=False, q_group_size=-1, per_tensor=True, inplace=False)
    elif mode=='nf':
        tensor = pseudo_nf_quantize_tensor(w, n_bits=n_bits, zero_point=False, q_group_size=-1, per_tensor=True, inplace=False)
    elif mode=='pn':
        tensor = pseudo_pn_quantize_tensor(w, n_bits=n_bits, zero_point=False, q_group_size=-1, per_tensor=True, inplace=False, pn_type=pn_type)
    elif mode=='apot':
        tensor = pseudo_apot_quantize_tensor(w, n_bits=n_bits, zero_point=False, q_group_size=-1, per_tensor=True, inplace=False)
    elif mode=='dynamic_pn':
        tensor = dynamic_pseudo_pn_quantize_tensor(w, n_bits=n_bits, zero_point=False, q_group_size=-1, per_tensor=True, inplace=False, model_name=model_name, module_name=module_name)
    
    return tensor
    
@torch.no_grad()
def quantize_activation_per_tensor_absmax(t, n_bits=8, mode='int',pn_type='offset'):
    t_shape = t.shape
    # t = t.contiguous().view(-1, t_shape[-1])
    t = t.contiguous().view(t_shape[0], -1)
    if mode=='int':
        t = pseudo_quantize_tensor(t, n_bits=n_bits, zero_point=False, q_group_size=-1, per_tensor=True, inplace=False)
    elif mode=='fp':
        t = pseudo_fp_quantize_tensor(t, n_bits=n_bits, zero_point=False, q_group_size=-1, per_tensor=True, inplace=False)
    elif mode=='nf':
        t = pseudo_nf_quantize_tensor(t, n_bits=n_bits, zero_point=False, q_group_size=-1, per_tensor=True, inplace=False)
    elif mode=='pn':
        t = pseudo_pn_quantize_tensor(t, n_bits=n_bits, zero_point=False, q_group_size=-1, per_tensor=True, inplace=False, pn_type=pn_type)
    return t.reshape(t_shape)
