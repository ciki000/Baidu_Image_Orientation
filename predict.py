import os
import sys
import glob
from PIL import Image
import numpy as np
import onnxruntime
import paddle
from paddle.vision import transforms as T

#mport time

class TestSet(paddle.io.Dataset):
	def __init__(self, Dataset_dir, transforms):
		super(TestSet, self).__init__()       
		self.image_paths = glob.glob(os.path.join(Dataset_dir, "*.jpg"))
		self.transforms = transforms
	def __getitem__(self, index):
		image_path = self.image_paths[index]
		image_name = image_path.split('/')[-1]
		image = Image.open(image_path).convert('RGB')
		h = image.size[1]
		w = image.size[0]
		long_edge = max(h, w)
		image = T.pad(image, ((long_edge-w)//2, (long_edge-h)//2, (long_edge-w+1)//2, (long_edge-h+1)//2), padding_mode='constant', fill=(128,128,128))
		image = T.resize(image, (256, 256))
		image = self.transforms(image)
		return image, image_name
				
	def __len__(self):
		return len(self.image_paths)



if __name__ == '__main__':
	src_image_dir = sys.argv[1]
	output_filename = sys.argv[2]
	current_path = os.path.dirname(__file__)
	paddle.device.set_device("cpu")

	sess = onnxruntime.InferenceSession('model_bfp16.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
		
	transform_test = T.Compose([
		T.ToTensor(),
		T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
	])
	test_dataset  = TestSet(src_image_dir, transform_test)
	test_loader = paddle.io.DataLoader(test_dataset, 
										batch_size=256,
										num_workers=0,
										)

	with open(os.path.join(current_path, output_filename), 'w') as f:
		for i, (image, image_name) in enumerate(test_loader()):
			image = image.numpy().astype(np.float16)
						
			outputs = sess.run(None, {'input':image})
			outputs = outputs[0]
			for j in range(outputs.shape[0]): 
				pred_label = np.argmax(outputs[j])
				f.write(f'{image_name[j]} {pred_label}\n')
		f.close()
    
	# CUDA_VISIBLE_DEVICES=1 python predict6.py ./datasets/test_A/images/ ./predict.txt
    #print(time.time()-bg_time)