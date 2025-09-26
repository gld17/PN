# Towards Floating Point-Based AI Acceleration: Hybrid PIM with Non-Uniform Data Format and Reduced Multiplications

<p align="center">
🌐 &nbsp&nbsp📑 <a href="https://nicsefc.ee.tsinghua.edu.cn/%2Fnics_file%2Fpdf%2F42f6260d-133b-46b4-a3d5-2386b76608c8.pdf"><b>Paper</b></a>&nbsp&nbsp
</p>

In this work, we propose the PIM-oriented exponent-free non-uniform (PN) data format. The proposed PN format can be flexibly adjusted to fit the non-uniform distribution and approach FP-
based algorithm accuracy using bit-slicing-based full INT operations.

[<img src="assets/PN definition.png" width="">]()

### News

- [25/08] The extention work was accepted by ACM TODAES journal
- [24/06] The conference work was accepted by ICCAD'25, named **Towards Floating Point-Based Attention-Free LLM: Hybrid PIM with Non-Uniform Data Format and Reduced Multiplications**


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
