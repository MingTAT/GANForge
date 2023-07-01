import torch
from torch import nn
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

# 生成器模型定义
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Tanh()
        )
    
    def forward(self, x):
        output = self.model(x)
        output = output.view(x.size(0), 1, 28, 28)
        return output



# 判别器模型定义
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = x.view(x.size(0), 784)
        output = self.model(x)
        return output

# 加载模型
def load_models():
    generator = Generator()
    discriminator = Discriminator()

    generator.load_state_dict(torch.load('generator.pth'))
    discriminator.load_state_dict(torch.load('discriminator.pth'))

    generator.eval()
    discriminator.eval()

    return generator, discriminator

# 生成新的手写数字
def generate_image(generator):
    z = torch.randn(1, 100)
    fake_image = generator(z)
    fake_image = (fake_image + 1) / 2.0

    plt.imshow(fake_image.detach().numpy().reshape(28, 28), cmap='gray')
    plt.show()


# 评估一个图片
def evaluate_image(discriminator, image):
    output = discriminator(image)
    print('The discriminator output:', output.item())

# 主函数
def main():
    generator, discriminator = load_models()

    generate_image(generator)

    image = Image.open('./test.png')  # 这应该是一个图像文件的路径
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    image = transform(image).unsqueeze(0)
    evaluate_image(discriminator, image)

if __name__ == '__main__':
    main()
