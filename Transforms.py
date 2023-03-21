from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image

img_path = "dataset/train/ants_image/0013035.jpg"
img = Image.open(img_path)
writer = SummaryWriter("logs")

tensor_trans = transforms.ToTensor() # 创建工具
tensor_img = tensor_trans(img) # 使用工具

writer.add_image("Tensor_img", tensor_img)

writer.close()