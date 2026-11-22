# Project 1

安装：

    python -m pip install -r requirements.txt

检查（在project1-numpy-student目录）：

    python check.py --stage all

生成输出：

    python run.py --output output_tokens.npz

说明：只改了image_encoder.py，补完TODO-01到TODO-06；weights/image_encoder.npz和assets/reference_outputs.npz是已有的，没有改动。全部用NumPy float32计算。