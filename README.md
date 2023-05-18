# Int-and-Int
Int&Int: A Two-Pathway Network for Skeleton-Based Action Recognition.
Accepted by IEEE Conference on Industrial Electronics and Applications (ICIEA), 2023.
## Introduction
In this work, we propose a two-pathway Int&Int network(Intensity&Integrity) for skeleton-based action recognition to satisfy both aspects, where the great complementarity between the two pathways further enhances the performance. Besides, for Integrity pathway, we apply the uniform sampling strategy. For Intensity pathway, we introduce the intensity-dependent sampling, where a clip composed of consecutive frames around the frame with the largest motion intensity is sampled. Moreover, we explain various definitions of the motion intensity containing different semantic information based on the extracted 2D human poses. For each pathway, the poses are represented by a 3D heatmap volume and 3D-CNNs of both pathways have the same architecture. Late fusion is used to ensemble them. The model is evaluated on two action recognition datasets, FineGYM-99 and HMDB-51.

![](https://github.com/SarahQi666/Int-and-Int/blob/master/demo/1.png)
## Installation
```
git clone https://github.com/SarahQi666/Int-and-Int.git
pip install -r requirements.txt
pip install -e .
```
## Training & Testing
You can use following commands for training and testing. Basically, we support distributed training on a single server with multiple GPUs.
```
# Training
bash tools/dist_train.sh {config_name} {num_gpus} {other_options}
# Testing (single-pathway)
bash tools/dist_test.sh {config_name} {checkpoint} {num_gpus} --out {output_file} --eval top_k_accuracy mean_class_accuracy
# Testing (two-pathway)
bash tools/dist_test_2s.sh {config_name_Integrity} {config_name_Intensity} {checkpoint_Integrity} {checkpoint_Intensity} {num_gpus} --eval top_k_accuracy mean_class_accuracy --out {output_file}
```
## Contact
For any question, feel free to contact: sarah.qixiangyuan@gmail.com
