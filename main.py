import argparse
import torch
from tqdm import tqdm
import os
import sys
import torchvision.models as models
from torchvision import transforms
import torchvision
from torch.autograd import Variable

from quant.quant_wrapper import quantize_model
from utils import get_cnn_model, get_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, help="path of the hf model")
parser.add_argument("--dataset", type=str, default='cifar100', help="the evaluation dataset")
parser.add_argument("--w_mode", type=str, default='int', help="the mode of weight quantization")
parser.add_argument("--a_mode", type=str, default='int', help="the mode of activation quantization")
parser.add_argument("--pn_type", type=str, default='offset', help="the type of pn format")
parser.add_argument("--w_group_size", type=int, default=128)
parser.add_argument("--w_bit", type=int, default=4)
parser.add_argument("--a_group_size", type=int, default=128)
parser.add_argument("--a_bit", type=int, default=8)
parser.add_argument('--gpu_id', type=int, default=0)
params = parser.parse_args()

def cnn_evaluate(model, dataloader):
    model.eval()
    total_images = 0
    correct_1 = 0
    correct_5 = 0
    iter = 0
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader, 0)):
            images, labels = data
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
            outputs = model(images)
            _, predicts_1 = torch.max(outputs.data, 1)
            _, predicts_5 = outputs.data.topk(5, 1, True, True)
            total_images += labels.size(0)
            correct_1 += (predicts_1 == labels).sum().item()
            correct_5 += (predicts_5 == labels.unsqueeze(1).expand_as(predicts_5)).sum().item()

    return 100 * correct_1 / total_images, 100*correct_5 / total_images

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = str(params.gpu_id)
    device = torch.device("cuda")
    
    model, exclude_layers = get_cnn_model(params.model_name)
    model = model.to(device)

    test_dataset = get_dataset(params.dataset, params.model_name)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    baseline_top1_acc, baseline_top5_acc = cnn_evaluate(model, test_loader)
    print('Baseline-acc-top1={} Baseline-acc-top5={}'.format(baseline_top1_acc, baseline_top5_acc))

    # quantize model
    model = quantize_model(model, params, model_name=params.model_name, exclude_layers=exclude_layers)

    quantized_top1_acc, quantized_top5_acc = cnn_evaluate(model, test_loader)
    print('Quantized-acc-top1={} Quantized-acc-top5={}'.format(quantized_top1_acc, quantized_top5_acc))

    if params.w_mode == 'pn':
        log_path = './results/{}_{}_{}{}W{}{}A{}.txt'.format(params.model_name, params.dataset, params.w_mode, params.pn_type, params.w_bit, params.a_mode, params.a_bit)
    else:
        log_path = './results/{}_{}_{}W{}{}A{}.txt'.format(params.model_name, params.dataset, params.w_mode, params.w_bit, params.a_mode, params.a_bit)
    with open(log_path, 'a', encoding='utf-8') as file:
        file.write('Quantized-acc-top1={} Quantized-acc-top5={} \n'.format(quantized_top1_acc, quantized_top5_acc))
    




if __name__ == "__main__":
    main()