# Fixed PN parameter search
python quant/pn_search.py

# Model Quantization
python main.py --model_name mobilenetv2 --dataset imagenet --gpu_id 3 --w_bit 4 --a_bit 8 --w_mode pn --a_mode int