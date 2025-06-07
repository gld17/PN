import torch
from functools import partial

def quantize_model(model, args, model_name=None, exclude_layers=[]):
    # Weight-Activation quantization (ours)
    if args.w_bit is not None and args.w_bit > 0 and args.w_bit < 16 and args.a_bit is not None and args.a_bit > 0 and args.a_bit < 16:
        from quant.qlinear.sqwa import WALinear, WAConv2d, Appro_WAConv2d
        from utils import get_module_by_name_suffix
        # Replace original Linear/Conv2d module
        for name, module in model.named_modules():
            # TODO: how to tackle the last linear layer?
            if isinstance(module, torch.nn.Linear) and 'lm_head' not in name and 'output_layer' not in name and 'head' not in name and name not in exclude_layers:
                new_linear = WALinear.from_float(module, weight_quant='per_group', act_quant='per_token', w_bit=args.w_bit, a_bit=args.a_bit, weight_group=args.w_group_size, 
                                                 quantize_output=False, w_mode=args.w_mode, a_mode=args.a_mode, model_name=model_name, module_name=name, pn_type=args.pn_type)
                father_module = get_module_by_name_suffix(model, '.'.join(name.split('.')[:-1]))
                setattr(father_module, name.split('.')[-1], new_linear)
                del new_linear, module
                torch.cuda.empty_cache()
            elif isinstance(module, torch.nn.Conv2d) and module.groups==1:
                new_conv2d = WAConv2d.from_float(module, weight_quant='per_channel', act_quant='per_token', w_bit=args.w_bit, a_bit=args.a_bit, weight_group=args.w_group_size, 
                                                 quantize_output=False, w_mode=args.w_mode, a_mode=args.a_mode, model_name=model_name, module_name=name, pn_type=args.pn_type)
                father_module = get_module_by_name_suffix(model, '.'.join(name.split('.')[:-1]))
                setattr(father_module, name.split('.')[-1], new_conv2d)
                del new_conv2d, module
                torch.cuda.empty_cache()
            elif isinstance(module, torch.nn.Conv2d) and module.groups>1:
                # import ipdb; ipdb.set_trace()
                new_conv2d = Appro_WAConv2d.from_float(module, weight_quant='per_channel', act_quant='per_token', w_bit=args.w_bit, a_bit=args.a_bit, weight_group=args.w_group_size, 
                                                       quantize_output=False, w_mode=args.w_mode, a_mode=args.a_mode, model_name=model_name, module_name=name)
                father_module = get_module_by_name_suffix(model, '.'.join(name.split('.')[:-1]))
                setattr(father_module, name.split('.')[-1], new_conv2d)
                del new_conv2d, module
                torch.cuda.empty_cache()
                
    return model
