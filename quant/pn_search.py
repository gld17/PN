import torch
import argparse
import copy
import math
import numpy as np
from quant_funcs import *

x1_list = [0 if i%2==0 else 1 for i in range(2**8)]
x2_list = [0 if math.floor(i/2)%2==0 else 1 for i in range(2**8)]
x3_list = [0 if math.floor(i/4)%2==0 else 1 for i in range(2**8)]
x4_list = [0 if math.floor(i/8)%2==0 else 1 for i in range(2**8)]
x5_list = [0 if math.floor(i/16)%2==0 else 1 for i in range(2**8)]
x6_list = [0 if math.floor(i/32)%2==0 else 1 for i in range(2**8)]
x7_list = [0 if math.floor(i/64)%2==0 else 1 for i in range(2**8)]
x8_list = [0 if math.floor(i/128)%2==0 else 1 for i in range(2**8)]
x_list = [x1_list, x2_list, x3_list, x4_list, x5_list, x6_list, x7_list, x8_list]

parser = argparse.ArgumentParser()
parser.add_argument('--res', type=int, default=4)
parser.add_argument('--gpu_id', type=int, default=0)
params = parser.parse_args()

@torch.no_grad()
def pn_pseudo_quantize_tensor(tensor, n_bits=4, per_tensor=False, list=None):
    """
    The basic quantization function for weight, activation and KV cache.
    """
    org_tensor_shape = tensor.shape
    if per_tensor:
        tensor = tensor.reshape(1, -1)
    assert tensor.dim() == 2

    max_val = tensor.abs().amax(dim=1, keepdim=True)
    max_val = max_val.clamp(min=1e-5)
    max_pn = list.abs().max()
    min_pn = list.min()
    scales = max_val / max_pn
    zeros = 0

    pim_quantile_list = list

    tensor = tensor / scales + zeros
    diff = (tensor[...,None]-pim_quantile_list.cuda()).abs()
    index = torch.argmin(diff,-1)
    x = pim_quantile_list.cuda()[None,:].repeat(tensor.shape[0], tensor.shape[1], 1)
    tensor = x.gather(-1, index[...,None]).squeeze(-1)
    tensor = (torch.clamp(tensor, min_pn, max_pn) - zeros) * scales

    return tensor


def forward(x, i, res):
    output = 0
    for ind in range(res):
        output += x[ind] * x_list[ind][i]
    return output

if __name__=='__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = str(params.gpu_id)
    device = torch.device("cuda")


    activation = torch.randn(1,256).cuda()
    weight = torch.randn(256,256).cuda()
    gt_mv_results = torch.mm(activation, weight)
    res = params.res # pn8_search not work well

    criterion = torch.nn.MSELoss()
    loss_fn = torch.nn.MSELoss()

    gt = torch.Tensor(np.loadtxt('./quant/format_ref/NF/nf{}.txt'.format(res))).float().cuda()

    x=torch.arange(res).float()
    x.requires_grad=True
    gt = (gt+1)*0.5

    optimizer = torch.optim.Adam([{"params": x}], lr=0.5)

    for epoch in range(500):
        output = torch.zeros_like(gt, requires_grad=False)
        for i in range(gt.shape[0]):
            output[i]=forward(x,i,res)
        output = output.sort()[0].float()
        output = output/output.max()
        loss1 = loss_fn(output, gt)
        loss2 = loss_fn((output[1:]-output[:-1]), (gt[1:]-gt[:-1]))
        if res <= 4:
            total_loss = loss1
        elif res > 4:
            total_loss = loss1+loss2
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # lr_scheduler.step()
        print("Loss={}".format(total_loss.item()))

    num_list = []
    for i in range(2**res):
        num = forward(x, i, res)
        num_list.append(num.item())

    num_list = torch.Tensor(num_list).cuda()
    num_list = num_list.sort()[0]

    print("\n")
    print("------ Searched Optimal Values ------")
    print(x.tolist())
    
    print("\n")
    print("------ Quantile Point ------")
    print(num_list.tolist())

    print("\n")
    print("------ Quantization Interval ------")
    print((num_list[1:]-num_list[:-1]).tolist())

    sym_quantile_point = num_list/num_list.max()*2-1
    sym_quantile_point[sym_quantile_point.abs().argmin()] = 0
    print("\n")
    print(sym_quantile_point.tolist())

    ###### INT Approximation for PN #######
    gt = x.detach()/x.abs().min().item()

    loss = 1e5
    optimal_i = 1
    upper_limit = math.floor(256/gt.abs().max().item())
    # for i in range(1,upper_limit):
    #     appro_list=torch.round(gt*i)/i
    #     if criterion(appro_list, gt)<loss:
    #         loss = criterion(appro_list,gt)
    #         optimal_i = i
    for i in range(1,upper_limit):
        appro_list=torch.round(gt*i)
        factor = appro_list[0] / x[0]
        appro_int_quantile_list = []
        for j in range(2**res):
            num = forward(appro_list, j, res)
            appro_int_quantile_list.append(num.item())
        if len(list(set(appro_int_quantile_list)))<2**res:
            continue
        appro_int_quantile_list = torch.Tensor(appro_int_quantile_list).cuda().sort()[0] / factor
        appro_int_quantile_list /= appro_int_quantile_list.max()
        if criterion(appro_int_quantile_list, sym_quantile_point)<loss:
            loss = criterion(appro_list,gt)
            optimal_i = i
    
    int_num = torch.round(gt*optimal_i)
    print("\n")
    print("------ Searched Optimal Approximate INT8 Values ------")
    print(int_num.tolist())
    factor = int_num[0] / x[0]

    appro_int_quantile_list = []
    for i in range(2**res):
        num = forward(int_num, i, res)
        appro_int_quantile_list.append(num.item())
    appro_int_quantile_list = torch.Tensor(appro_int_quantile_list).cuda().sort()[0] / factor
    
    print("\n")
    print("------ Quantile Point ------")
    print(appro_int_quantile_list.tolist())

    print("\n")
    print("----- Quantization Interval ------")
    print((appro_int_quantile_list[1:]-appro_int_quantile_list[:-1]).tolist())

    sym_approx_quantile_point = appro_int_quantile_list/appro_int_quantile_list.max()*2-1
    sym_approx_quantile_point[sym_approx_quantile_point.abs().argmin()] = 0
    print("\n")
    print("------ The quantile points for symmetric weight quantization (range [-1,1]) ------")
    print(sym_approx_quantile_point.tolist())

    ###### Double '3bit-INT' Approximation for PN #######
    gt = x.detach()/x.abs().min().item()

    loss = 1e5
    optimal_i = 1
    upper_limit = math.floor(256/gt.abs().max().item())

    interval = [1,2,4,8,16,32,64,128]
    feasible_num_list = []
    for i in range(7):
        for j in range(7):
            for k in range(7):
                for t in range(7):
                    feasible_num_list.append(interval[i]+interval[j]+interval[k]+interval[t])
    feasible_num_list = torch.Tensor(list(set(feasible_num_list)))
    feasible_num_list = torch.cat((feasible_num_list, -feasible_num_list))

    # for i in range(1,upper_limit):
    #     appro_list=gt*i
    #     diff = (appro_list[...,None]-feasible_num_list).abs()
    #     index = torch.argmin(diff,-1)
    #     appro_list = feasible_num_list[None,:].repeat(appro_list.shape[0], 1)
    #     appro_list = appro_list.gather(-1, index[...,None]).squeeze(-1)

    #     appro_list/=i
    #     if criterion(appro_list, gt)<loss:
    #         loss = criterion(appro_list,gt)
    #         optimal_i = i

    for i in range(1,upper_limit):
        appro_list=gt*i
        diff = (appro_list[...,None]-feasible_num_list).abs()
        index = torch.argmin(diff,-1)
        appro_list = feasible_num_list[None,:].repeat(appro_list.shape[0], 1)
        appro_list = appro_list.gather(-1, index[...,None]).squeeze(-1)

        factor = appro_list[0] / x[0]
        double_appro_int_quantile_list = []
        for j in range(2**res):
            num = forward(appro_list, j, res)
            double_appro_int_quantile_list.append(num.item())
        if len(list(set(double_appro_int_quantile_list)))<2**res:
            continue
        double_appro_int_quantile_list = torch.Tensor(double_appro_int_quantile_list).cuda().sort()[0] / factor
        double_appro_int_quantile_list /= double_appro_int_quantile_list.max()
        if criterion(double_appro_int_quantile_list, sym_quantile_point)<loss:
            loss = criterion(appro_list,gt)
            optimal_i = i

    int_num = gt*optimal_i
    diff = (int_num[...,None]-feasible_num_list).abs()
    index = torch.argmin(diff,-1)
    int_num = feasible_num_list[None,:].repeat(int_num.shape[0], 1)
    int_num = int_num.gather(-1, index[...,None]).squeeze(-1)
    print("\n")
    print("------ Searched Optimal Double Approximate INT Values ------")
    print(int_num.tolist())
    factor = int_num[0] / x[0]

    double_appro_int_quantile_list = []
    for i in range(2**res):
        num = forward(int_num, i, res)
        double_appro_int_quantile_list.append(num.item())
    double_appro_int_quantile_list = torch.Tensor(double_appro_int_quantile_list).cuda().sort()[0]

    print("\n")
    print("------ Quantile Point ------")
    print(double_appro_int_quantile_list.tolist())

    sym_double_approx_quantile_point = double_appro_int_quantile_list/double_appro_int_quantile_list.max()*2-1
    sym_double_approx_quantile_point[sym_double_approx_quantile_point.abs().argmin()] = 0

    print("\n")
    print("----- Quantization Interval ------")
    print((double_appro_int_quantile_list[1:]-double_appro_int_quantile_list[:-1]).tolist())

    print("\n")
    print("------ The double-approximate quantile points for symmetric weight quantization (range [-1,1]) ------")
    print(sym_double_approx_quantile_point.tolist())

    ###### INT W4A8 Quantization ######
    print("\n")
    print("------ INT W{}A8 Quantization ------".format(res))
    quantized_activation = quantize_activation_per_tensor_absmax(activation, n_bits=8, mode='int')
    quantized_weight = pseudo_quantize_tensor(weight, n_bits=res, zero_point=False, q_group_size=-1, per_tensor=False, inplace=False)
    int_mv_results = torch.mm(quantized_activation, quantized_weight)
    int_quan_error = criterion(int_mv_results, gt_mv_results)
    print("Weight Quantization Error: {}".format(criterion(quantized_weight, weight)))
    print("Quantization Error: {}".format(int_quan_error))

    ###### PN W4A8 Quantization ######
    print("------ PN W{}A8 Quantization ------".format(res))
    quantized_activation = quantize_activation_per_tensor_absmax(activation, n_bits=8, mode='int')
    quantized_weight = pn_pseudo_quantize_tensor(weight, n_bits=res, list=sym_quantile_point)
    pn_mv_results = torch.mm(quantized_activation, quantized_weight)
    pn_quan_error = criterion(pn_mv_results, gt_mv_results)
    print("Weight Quantization Error: {}".format(criterion(quantized_weight, weight)))
    print("Quantization Error: {}".format(pn_quan_error))
    
    ###### Appro-PN W4A8 Quantization ######
    print("------ Appro-PN W{}A8 Quantization ------".format(res))
    quantized_activation = quantize_activation_per_tensor_absmax(activation, n_bits=8, mode='int')
    quantized_weight = pn_pseudo_quantize_tensor(weight, n_bits=res, list=sym_approx_quantile_point)
    pn_mv_results = torch.mm(quantized_activation, quantized_weight)
    pn_quan_error = criterion(pn_mv_results, gt_mv_results)
    print("Weight Quantization Error: {}".format(criterion(quantized_weight, weight)))
    print("Quantization Error: {}".format(pn_quan_error))

    ###### Double-Appro-PN W4A8 Quantization ######
    print("------ Double-Appro-PN W{}A8 Quantization ------".format(res))
    quantized_activation = quantize_activation_per_tensor_absmax(activation, n_bits=8, mode='int')
    quantized_weight = pn_pseudo_quantize_tensor(weight, n_bits=res, list=sym_double_approx_quantile_point)
    pn_mv_results = torch.mm(quantized_activation, quantized_weight)
    pn_quan_error = criterion(pn_mv_results, gt_mv_results)
    print("Weight Quantization Error: {}".format(criterion(quantized_weight, weight)))
    print("Quantization Error: {}".format(pn_quan_error))