import torch
import torchvision
from PIL import Image
from torch import nn

# 传入图片处理
img_path = "./imgs/dog.jpg"
img = Image.open(img_path)
img = img.convert("RGB")
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor()
])
img = transform(img)
img = torch.reshape(img, (1, 3, 32, 32))
img = img.cuda()

# 开始识别
model = torch.load("tudui_0.pth")
model.cuda()
model.eval()
with torch.no_grad():
    output = model(img)
print(output.argmax(1))