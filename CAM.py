from PIL import Image
import torch
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import json

import models
# 读取 imagenet数据集的类别标签
json_path = 'CAM/labels.json'
with open(json_path, 'r') as load_f:
    load_json = json.load(load_f)
classes = {int(key): value for (key, value)
           in load_json.items()}

img_path = 'CAM/9933031-large.jpeg'
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# 图片预处理
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

img_pil = Image.open(img_path)
img_tensor = preprocess(img_pil)
img_variable = Variable(img_tensor.unsqueeze(0))

model_id = 1
if model_id == 1:
    net = models.swin_transformer.SwinTransformer()
    
    pthfile = r'pth/pth/swin_tiny_patch4_window7_224.pth'
    net.load_state_dict(torch.load(pthfile))
    finalconv_name = 'features'  # 获取卷积层的特征
elif model_id == 2:
    net = models.resnet18(pretrained=False)
    finalconv_name = 'layer4'
elif model_id == 3:
    net = models.densenet161(pretrained=False)
    finalconv_name = 'features'
net.eval()	# 使用eval()属性
print(net)



features_blobs = []     # 后面用于存放特征图

def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

print(11111111)
print(net._modules.get(finalconv_name))
# 获取 features 模块的输出
net._modules.get(finalconv_name).register_forward_hook(hook_feature)

# 获取权重
net_name = []
params = []
for name, param in net.named_parameters():
    net_name.append(name)
    params.append(param)
print(net_name[-1], net_name[-2])	# classifier.1.bias classifier.1.weight
print(len(params))		# 52
weight_softmax = np.squeeze(params[-2].data.numpy())	# shape:(1000, 512)

logit = net(img_variable)				# 计算输入图片通过网络后的输出值
print(logit.shape)						# torch.Size([1, 1000])
print(params[-2].data.numpy().shape)	# 权重有1000种 (1000, 512, 1, 1)
print(features_blobs[0].shape)			# 特征图大小为　(1, 512, 13, 13)

# 结果有1000类，进行排序，并获得排序索引
h_x = F.softmax(logit, dim=1).data.squeeze()
print(h_x.shape)						# torch.Size([1000])
probs, idx = h_x.sort(0, True)
probs = probs.numpy()					# 概率值排序
idx = idx.numpy()						# 类别索引排序，概率值越高，索引越靠前

# 取概率值为前5的类别看看类别名和概率值
for i in range(0, 5):
    print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))
'''
0.678 -> mountain bike, all-terrain bike, off-roader
0.088 -> bicycle-built-for-two, tandem bicycle, tandem
0.042 -> unicycle, monocycle
0.038 -> horse cart, horse-cart
0.019 -> lakeside, lakeshore

'''

# 定义计算CAM的函数
def returnCAM(feature_conv, weight_softmax, class_idx):
    # 类激活图上采样到 256 x 256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    # 将权重赋给卷积层：这里的weigh_softmax.shape为(1000, 512)
    # 				feature_conv.shape为(1, 512, 13, 13)
    # weight_softmax[class_idx]由于只选择了一个类别的权重，所以为(1, 512)
    # feature_conv.reshape((nc, h * w))后feature_conv.shape为(512, 169)
    cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h * w)))
    print(cam.shape)		# 矩阵乘法之后，为各个特征通道赋值。输出shape为（1，169）
    cam = cam.reshape(h, w) # 得到单张特征图
    # 特征图上所有元素归一化到 0-1
    cam_img = (cam - cam.min()) / (cam.max() - cam.min())
    # 再将元素更改到　0-255
    cam_img = np.uint8(255 * cam_img)
    output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

# 对概率最高的类别产生类激活图
CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])
# 融合类激活图和原始图片
img = cv2.imread(img_path)
height, width, _ = img.shape
heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
result = heatmap * 0.3 + img * 0.7
cv2.imwrite('CAM0.jpg', result)
