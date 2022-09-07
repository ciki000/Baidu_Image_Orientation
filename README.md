# 百度网盘AI大赛-文档图像方向识别赛方案

## 概述
我们使用了MobileVitv2_050作为我们方法的backbone，使用了我们提出的自监督ROT Loss和Cross Entropy作为损失函数进行训练。我们将模型量化为FP16半精度模型，最终模型大小为2.2M。

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

## 训练数据
+ 文档图片

    英文文档图片 DocBacnk
    
    中文文档图片 CDLA_DATASET

    百度网盘AI大赛——模糊文档图像恢复赛道训练数据的GroundTruth图片

    使用爬虫爬取的百度百科截图
+ 普通图片

    VOC2007
    
    VOC2012
    
    food-101
    
    Animals_with_Attributes2
    
    WIDER

## 模型参数
+ 数据增强

    随机翻转，随机灰度化，随机调整锐度，Random Erasing，Copy-Paste