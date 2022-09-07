import os
import argparse
import torch
import timm

from utils.dataloader import load_checkpoint

parser = argparse.ArgumentParser(description='Exporting Model')
parser.add_argument('--model', default ='mobilevitv2_050', type=str)
parser.add_argument('--checkpoint', default ='./log/mobilevitv2_050/models/model_best.pth', type=str)
parser.add_argument('--save_dir', default ='./', type=str)
parser.add_argument('--gpu_index', default='0', type=str, help='GPUs')

def main(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_index
    model = timm.create_model( 
            args.model,
            pretrained=True,
            num_classes=4,
        )

    if 'best' not in args.checkpoint:
        from collections import OrderedDict
        state_dict = torch.load(args.checkpoint)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'module.module.' in k:
                name = k[14:]
            elif 'module.' in k:
                name = k[7:]
            else:
                name = k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)
    else:
        load_checkpoint(model, args.checkpoint)
    model.eval()

    x = torch.randn((1, 3, 256, 256))
    torch.onnx.export(model,  # model being run
                    x,  # model input (or a tuple for multiple inputs)
                    os.path.join(args.save_dir, "model_b256.onnx"),  # where to save the model (can be a file or file-like object)
                    export_params=True,  # store the trained parameter weights inside the model file
                    opset_version=10,  # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names=['input'],  # the model's input names
                    output_names=['output'],  # the model's output names
                    dynamic_axes={'input': {0:'batch_size'}})

if __name__ == '__main__':
 
    args = parser.parse_args()
    main(args)