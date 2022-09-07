# 百度网盘AI大赛-文档图像方向识别赛方案

## 概述

## 快速开始
模型训练

~~~
python train.py \
    --env mobilevitv2_050 \
    --train_set ../datasets/trainset_v3/ \
    --train_list ../datasets/trainset_v3/train_list.txt \
    --valid_set ../datasets/test_A/images \
    --valid_list ../datasets/test_A/label.txt \
    --model mobilevitv2_050 \
    --epochs 120 \
    --batch_size 128 \
    --lr 0.001 \
    --min_lr 1e-5 \
    --warmup \
    --warmup_epochs 10 \
    --rot_loss \
    --rotLoss_weight 0.1 \
    --label_smooth 0.1 \
    --fp16 \
    --copypaste \
    --weight_decay 1e-5 
~~~

导出onnx模型
~~~
python export_onnx.py \
    --checkpoint ./log/mobilevitv2_050/models/model_best.pth
~~~

转换为FP16半精度模型
~~~
python Tofp16.py
~~~

模型预测
~~~
python predict6.py ./datasets/test_A/images/ ./predict.txt
~~~