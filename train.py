import os
import time
import datetime
import random
import argparse
import json
from tqdm import tqdm
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data 
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import timm
from timm.utils import NativeScaler


from warmup_scheduler import GradualWarmupScheduler
import utils
from utils.dataloader import  train_set
from utils.dataloader import load_checkpoint

parser = argparse.ArgumentParser(description='Model training')
parser.add_argument('--env', default='_', type=str,  help='env')
parser.add_argument('--model', default='mobilenetv3_small_050', type=str,  help='model')
parser.add_argument('--pretrain', default=None, type=str)
parser.add_argument('--train_set', default='../datasets/text_image_orientation/', type=str, help='Directory of train set')
parser.add_argument('--train_list', default='../datasets/text_image_orientation/extra_v2_train_list.txt', type=str, help='List of train set')
parser.add_argument('--valid_set', default='../datasets/text_image_orientation/', type=str, help='Directory of valid set')
parser.add_argument('--valid_list', default='../datasets/text_image_orientation/extra_v2_test_list.txt', type=str, help='List of valid set')


parser.add_argument('--image_size', default=256, type=int)
parser.add_argument('--rot_loss', default=False, action='store_true')
parser.add_argument('--rotLoss_weight', default=0.1, type=float)
parser.add_argument('--label_smooth', default=0.0, type=float)


parser.add_argument('--gpu_index', default='0', type=str, help='GPUs')
parser.add_argument('--num_workers', default=16, type=int, help='number of data loading workers')

parser.add_argument('--copypaste', default=False, action='store_true')
parser.add_argument('--fp16', default=False, action='store_true')
parser.add_argument('--optimizer', default ='Adamw', type=str, help='optimizer for training')
parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--min_lr', default=1e-6, type=float, help='min learning rate')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--warmup', default=False, action='store_true', help='warm up') 
parser.add_argument('--warmup_epochs', default=5, type=int, help='epochs for warm up')
parser.add_argument('--epochs', default=60, type=int, help='number of total epochs to run')
parser.add_argument('--batch_size', default=256, type=int, help='number of batch size')
# parser.add_argument('--eval_resize', default=False, action='store_true', help='resize valid image') 
parser.add_argument('--eval_freq', default=1, type=int, help='Frequency of validation')

def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)


class CrossEntropy(nn.Module):
    def __init__(self):
        super(CrossEntropy, self).__init__()

    def forward(self, outputs, labels):
        loss = torch.nn.KLDivLoss(reduction='batchmean')
        outputs = outputs.float()
        return loss(F.log_softmax(outputs, dim=1), labels)

class RotLoss(nn.Module):
    def __init__(self):
        super(RotLoss, self).__init__()

    def forward(self, outputs1, outputs2):
        loss = torch.nn.KLDivLoss(reduction='batchmean')
        outputs1 = outputs1.float()
        outputs2 = outputs2.float()
        return loss(F.log_softmax(outputs1, dim=1), F.softmax(outputs2, dim=1))
        # return loss(F.log_softmax(outputs1, dim=1), F.log_softmax(outputs2, dim=1))

def rotsinglelabel(label):
    #print(label.shape)
    newlabel = label.clone()
    newlabel[3] = label[2]
    newlabel[2] = label[1]
    newlabel[1] = label[0]
    newlabel[0] = label[3]
    return newlabel
def rotlabel(labels, rot):
    newlabels = labels.clone()
    b = newlabels.shape[0]

    for i in range(b):
        for j in range(rot):
            #print('old:', newlabels[i])
            newlabels[i] = rotsinglelabel(newlabels[i])
            #print('new:', newlabels[i])
    
    return newlabels
def main(args):

    print(args)
    # Set GPUs
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_index
    torch.backends.cudnn.benchmark = True

    # Set Logs
    dir_name = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(dir_name, 'log', args.env)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    model_dir  = os.path.join(log_dir, 'models')
    utils.mkdir(model_dir)

    # Set Seeds
    random.seed(12)
    np.random.seed(12)
    torch.manual_seed(12)
    torch.cuda.manual_seed_all(12)

    # DataLoader
    transform_train = transforms.Compose([
        transforms.RandomCrop(args.image_size, padding=0),
        transforms.RandomHorizontalFlip(),
        #transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125, hue=0.125),]), p=0.3),
        #transforms.ColorJitter(brightness=0.125, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.2),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.RandomErasing(value=128),
    ])
    train_dataset = train_set(args.train_set, args.train_list, transform=transform_train, crop_mode=-1, img_size=args.image_size, use_rotLoss=args.rot_loss, label_smooth=args.label_smooth)
    if args.rot_loss:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=(args.batch_size // 4), shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    transform_valid = transforms.Compose([
        #transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    valid_dataset = train_set(args.valid_set, args.valid_list, transform=transform_valid, crop_mode=1, img_size=args.image_size)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False, drop_last=False)

    # model

    # model_zoo = timm.list_models('resnet*')
    # print(model_zoo)

    print("------------------------------------------------------------------")
    print('MODEL:', args.model)

    if args.model=='edgeformer':
        setattr(args, "model.classification.dropout", 0.2)
        setattr(args, "model.classification.edge.scale", "scale_xs")
        setattr(args, "model.classification.edge.mode", "outer_frame_v1")
        setattr(args, "model.classification.edge.kernel", "gcc_ca")
        setattr(args, "model.classification.edge.fusion", "add")
        setattr(args, "model.classification.edge.instance_kernel", "crop")
        setattr(args, "model.classification.edge.use_pe", True)
        setattr(args, "model.classification.activation.name", "swish")

        setattr(args, "model.normalization.name", "batch_norm_2d")
        setattr(args, "model.normalization.momentum", 0.1)
        setattr(args, "model.activation.name", "swish")

        setattr(args, "model.layer.global_pool", "mean")
        setattr(args, "model.layer.conv_init", "kaiming_normal")
        setattr(args, "model.layer.linear_init", "trunc_normal")
        setattr(args, "model.layer.linear_init_std_dev", 0.02)

        setattr(args, "model.classification.n_classes", 4)
        import models.EdgeFormer 
        from models.EdgeFormer.build import build_classification_model
        model = build_classification_model(args)
    else:
        model = timm.create_model( 
            args.model,
            pretrained=True,
            num_classes=4,
        )

    if (args.pretrain!=None):
        print('Load checkpoint:', args.pretrain)
        load_checkpoint(model, args.pretrain)
    model_size = getModelSize(model)[-1]
    print('MODEL SIZE: {:.3f}MB'.format(model_size))
    print("------------------------------------------------------------------")
    # DataParallel 
    model = torch.nn.DataParallel(model)
    model.cuda()

    # Optimizer 
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)
    # Scheduler
    if args.warmup:
        print("Scheduler: Using warmup and cosine strategy")
        warmup_epochs = args.warmup_epochs
        scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs-warmup_epochs, eta_min=args.min_lr)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
        scheduler.step()
    else:
        step = 50
        print("Scheduler: Using stepLR, step={}".format(step))
        scheduler = StepLR(optimizer, step_size=step, gamma=0.5)
        scheduler.step()


    ########## Train ##########
    criterion = CrossEntropy().cuda()
    criterion_rot = RotLoss().cuda()

    loss_scaler = NativeScaler()
    torch.cuda.empty_cache()

    best_acc = 0

    if (args.rot_loss):
        #init_alpha = 1-args.rotLoss_weight
        # init_beta = args.rotLoss_weight
        alpha = 1-args.rotLoss_weight
        beta = args.rotLoss_weight
        assert alpha>=0. and alpha<=1.
        print('alpha:', alpha, 'beta:', beta)

        
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        epoch_loss = 0

        # if (args.rot_loss):
        #     beta = args.rotLoss_weight * (epoch / args.epochs)
        #     alpha = 1 -  beta
        #     assert alpha>=0. and alpha<=1.
        #     print('alpha: %.2f'%alpha, 'beta: %.2f'%beta)

        for i, data in enumerate(train_loader, 0): 
            optimizer.zero_grad()

            if (args.rot_loss):
                if args.copypaste:
                    copypaste_imgs = data[0].clone()
                    b, c, h, w = copypaste_imgs.shape
                    
                    for src_img in range(b):
                        for copy_img in range(b):
                            if (src_img == copy_img):
                                continue
                            if (random.random() < 0.02):
                                patch_h = random.randint(16, 128)
                                patch_w = random.randint(16, 128)

                                row_copy = random.randint(0, h-patch_h)
                                col_copy = random.randint(0, w-patch_w)
                                row_paste = random.randint(0, h-patch_h)
                                col_paste = random.randint(0, w-patch_w)
                                copypaste_imgs[src_img, :, row_paste:row_paste+patch_h, col_paste:col_paste+patch_w] = data[0][copy_img, :, row_copy:row_copy+patch_h, col_copy:col_copy+patch_w]
                    data[0] = copypaste_imgs.clone()

                image_0 = data[0].cuda()
                image_1 = torch.rot90(data[0], -1, (2,3)).cuda()
                image_2 = torch.rot90(data[0], -2, (2,3)).cuda()
                image_3 = torch.rot90(data[0], -3, (2,3)).cuda()

                label_0 = data[1].cuda()
                label_1 = rotlabel(data[1], 1).cuda()
                label_2 = rotlabel(data[1], 2).cuda()
                label_3 = rotlabel(data[1], 3).cuda()
                if args.fp16:
                    with torch.cuda.amp.autocast():
                        pre_0 = model(image_0)
                        pre_1 = model(image_1)
                        pre_2 = model(image_2)
                        pre_3 = model(image_3)

                        loss_0 = alpha*criterion(pre_0, label_0) + beta * criterion_rot(rotlabel(pre_0.cpu(), 1).cuda(), pre_1.detach())# pre_1.detach())
                        loss_1 = alpha*criterion(pre_1, label_1) + beta * criterion_rot(rotlabel(pre_1.cpu(), 1).cuda(), pre_2.detach())
                        loss_2 = alpha*criterion(pre_2, label_2) + beta * criterion_rot(rotlabel(pre_2.cpu(), 1).cuda(), pre_3.detach())
                        loss_3 = alpha*criterion(pre_3, label_3) + beta * criterion_rot(rotlabel(pre_3.cpu(), 1).cuda(), pre_0.detach())
                        loss = loss_0 + loss_1 + loss_2 + loss_3
                else:
                    pre_0 = model(image_0)
                    pre_1 = model(image_1)
                    pre_2 = model(image_2)
                    pre_3 = model(image_3)

                    loss_0 = alpha*criterion(pre_0, label_0) + beta * criterion_rot(rotlabel(pre_0.cpu(), 1).cuda(), pre_1.detach())
                    loss_1 = alpha*criterion(pre_1, label_1) + beta * criterion_rot(rotlabel(pre_1.cpu(), 1).cuda(), pre_2.detach())
                    loss_2 = alpha*criterion(pre_2, label_2) + beta * criterion_rot(rotlabel(pre_2.cpu(), 1).cuda(), pre_3.detach())
                    loss_3 = alpha*criterion(pre_3, label_3) + beta * criterion_rot(rotlabel(pre_3.cpu(), 1).cuda(), pre_0.detach())
                    loss = loss_0 + loss_1 + loss_2 + loss_3
            else:
                image = data[0].cuda()
                label = data[1].cuda()

                if args.fp16:
                    with torch.cuda.amp.autocast():
                        pre = model(image)
                        loss = criterion(pre, label)
                else:
                    pre = model(image)
                    loss = criterion(pre, label)

            loss_scaler(loss, optimizer, parameters=model.parameters())
            epoch_loss += loss.item()
        
        if (epoch % args.eval_freq == 0 or epoch == args.epochs -1):
            with torch.no_grad():
                model.eval()
                total = 0
                acc_num = 0

                for i, data in enumerate((valid_loader), 0):
                    image = data[0].cuda()
                    label = data[1].cuda()

                    if args.fp16:
                        with torch.cuda.amp.autocast():
                            pre = model(image)
                    else:
                        pre = model(image)
                    
                    gt_label = torch.topk(label, 1)[1]
                    pr_label = torch.topk(pre, 1)[1]
                    total += image.shape[0]
                    for j in range(image.shape[0]): 
                        if (gt_label[j]==pr_label[j]):
                            acc_num = acc_num+1
                
                acc = acc_num/total
                if (acc > best_acc):
                    best_acc = acc
                    best_epoch = epoch
                    torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict()}, os.path.join(model_dir, "model_best.pth"))

                print("[Epoch %d\t acc: %.4f\t] ---- [best_Epoch %d best_acc %.4f] " % (epoch, acc, best_epoch, best_acc))
                model.train()
                torch.cuda.empty_cache()
        
        scheduler.step()
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(),'optimizer' : optimizer.state_dict()}, os.path.join(model_dir, "model_latest.pth"))
        print("------------------------------------------------------------------")
        print("Epoch: {}\t Time: {:.4f}\t Loss: {:.4f}\t LearningRate {:.6f}".format(epoch, time.time()-epoch_start_time, epoch_loss, scheduler.get_lr()[0]))
        print("------------------------------------------------------------------")

if __name__ == '__main__':
 
    args = parser.parse_args()
    main(args)