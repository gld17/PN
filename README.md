# Towards Floating Point-Based AI Acceleration: Hybrid PIM with Non-Uniform Data Format and Reduced Multiplications

## 🛠️ PIM-oriented Non-uniformed Data Format (PN)

In this paper, we design a non-uniform data format for PIM architecture, named PN. Ideally, the PN format can be adjusted by changing its bit scaling factors to fit different types of non-uniform data formats (FP, NF, ...).

Here, we provide example scripts for the fixed PN parameter search targeting NF format.

* Fixed PN parameter search

  ```
  python quant/pn_search.py
  ```

## 🧪 Algorithm Evaluation
After searching for the appropriate PN format parameters, the model weights can be quantized offline and evaluated on the target dataset.
* Model Quantization and Evaluation

  ```
  python main.py --model_name mobilenetv2 --dataset imagenet --gpu_id 3 --w_bit 4 --a_bit 8 --w_mode pn --a_mode int
  ```
