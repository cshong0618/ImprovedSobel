import argparse
import sys

from PIL import Image
import PIL
import torch.nn as nn
import torch
import numpy as np
from torchvision import transforms
from torch.autograd import Variable


class Sobel(nn.Module):
    def __init__(self, removal_scale=5):
        super(Sobel, self).__init__()
        self.removal_scale = removal_scale
        self.x3 = np.array([
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1]
        ])

        self.conv_x3 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_x3.weight = nn.Parameter(torch.from_numpy(self.x3).float().unsqueeze(0).unsqueeze(0))

        self.y3 = self.x3.T

        self.conv_y3 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_y3.weight = nn.Parameter(torch.from_numpy(self.y3).float().unsqueeze(0).unsqueeze(0))

        self.x5 = np.array([
            [-2/8, -1/5, 0, 1/5, 2/8],
            [-2/5, -1/2, 0, 1/2, 2/5],
            [-2/4, -1/1, 0, 1/1, 2/4],
            [-2/5, -1/2, 0, 1/2, 2/5],
            [-2/8, -1/5, 0, 1/5, 2/8],
        ])

        self.y5 = self.x5.T

        self.conv_x5 = nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv_x5.weight = nn.Parameter(torch.from_numpy(self.x5).float().unsqueeze(0).unsqueeze(0))

        self.conv_y5 = nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv_y5.weight = nn.Parameter(torch.from_numpy(self.y5).float().unsqueeze(0).unsqueeze(0))

        self.x7 = np.array([
            [-3/18, -2/13, -1/10, 0, 1/10, 2/13, 3/18],
            [-3/13, -2/8 , -1/5 , 0, 1/5 , 2/8 , 3/13],
            [-3/10, -2/5 , -1/2 , 0, 1/2 , 2/5 , 3/10],
            [-3/9 , -2/4 , -1/1 , 0, 1/1 , 2/4 , 3/9 ],
            [-3/10, -2/5 , -1/2 , 0, 1/2 , 2/5 , 3/10],
            [-3/13, -2/8 , -1/5 , 0, 1/5 , 2/8 , 3/13],
            [-3/18, -2/13, -1/10, 0, 1/10, 2/13, 3/18],
        ])

        self.y7 = self.x7.T

        self.conv_x7 = nn.Conv2d(1, 1, kernel_size=7, stride=1, padding=3, bias=False)
        self.conv_x7.weight = nn.Parameter(torch.from_numpy(self.x7).float().unsqueeze(0).unsqueeze(0))

        self.conv_y7 = nn.Conv2d(1, 1, kernel_size=7, stride=1, padding=3, bias=False)
        self.conv_y7.weight = nn.Parameter(torch.from_numpy(self.y7).float().unsqueeze(0).unsqueeze(0))


    def forward(self, X):
        G_x3 = self.conv_x3(X).data.view(1, X.size(2), X.size(3))
        G_y3 = self.conv_y3(X).data.view(1, X.size(2), X.size(3))

        G_x5 = self.conv_x5(X).data.view(1, X.size(2), X.size(3))
        G_y5 = self.conv_y5(X).data.view(1, X.size(2), X.size(3))

        G_x7 = self.conv_x7(X).data.view(1, X.size(2), X.size(3))
        G_y7 = self.conv_y7(X).data.view(1, X.size(2), X.size(3))

        G = (torch.sqrt(torch.pow(G_x7, 2) + torch.pow(G_y7, 2)) / 7 + torch.sqrt(torch.pow(G_x5, 2) + torch.pow(G_y5, 2)) / 5 + torch.sqrt(torch.pow(G_x3, 2) + torch.pow(G_y3, 2)) / 3) / self.removal_scale
        #G = torch.sqrt(torch.pow(G_x3, 2) + torch.pow(G_y3, 2))

        return G

parser = argparse.ArgumentParser()
parser.add_argument("--input", help="Input file", type=str, required=True)
parser.add_argument("--output", help="Output file", type=str, required=True)
parser.add_argument("--invert", help="Invert color to blank line white background", default=False, action='store_true')
parser.add_argument("--removal_scale", help="Magnitude of details removal", default=5, type=float)
parser.add_argument("--gpu", help="Use GPU", default=False, action='store_true')

args = parser.parse_args(sys.argv[1:])

img = Image.open(args.input).convert('LA')

T = transforms.Compose([transforms.ToTensor()])
P = transforms.Compose([transforms.ToPILImage()])

img_tensor = T(img)

x = img_tensor[0].view(1, 1, img_tensor[0].size(0), img_tensor[0].size(1))

sobel = Sobel(args.removal_scale)

if args.gpu:
    sobel.cuda()
    x = Variable(x).cuda()
else:
    x = Variable(x)

X = sobel(x)

if args.gpu:
    X = P(X.cpu())
else:
    X = P(X)

if args.invert:
    X = PIL.ImageOps.invert(X)

X.save(args.output)