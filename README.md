# Towards Floating Point-Based AI Acceleration: Hybrid PIM with Non-Uniform Data Format and Reduced Multiplications

**[[Code]](https://github.com/gld17/PN.git)**

We provide example scripts for the PN parameter search and model quantization based on PN format.

* Fixed PN parameter search

  ```
    python quant/pn_search.py
  ```
* Model Quantization and Evaluation

  ```
  python main.py --model_name mobilenetv2 --dataset imagenet --gpu_id 3 --w_bit 4 --a_bit 8 --w_mode pn --a_mode int
  ```
