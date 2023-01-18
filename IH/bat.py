import torch
from model import Hide, Reveal
from utils import DatasetFromFolder
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import cv2
import numpy as np
import math
device = torch.device("cpu")
def psnr1(img1, img2):
    mse = np.mean((img1 / 1.0 - img2 / 1.0) ** 2)
    if mse < 1.0e-10:
        return 100

    return 10 * math.log10(255.0 ** 2 / mse)
# 将模型进行导入
def hide_moxing(hide):
    hide_net = Hide()
    hide_net.eval()

    hide_net.to(device)

    hide_net.load_state_dict(torch.load(hide,map_location='cpu'))
    return hide_net
def reveal_moxing(reveal):
    reveal_net = Reveal()
    reveal_net.eval()
    reveal_net.to(device)

    reveal_net.load_state_dict(torch.load(reveal,map_location='cpu'))
    return reveal_net
if __name__ == '__main__':
    dataset = DatasetFromFolder('./data1', crop_size=256)
    # 然后利用torch.utils.data.DataLoader将整个数据集分成多个批次。将400数据集以每组32个进行导出运算
    dataloader = DataLoader(dataset, 1, shuffle=True, num_workers=4)
    hide = './checkpoint/epoch_1080_hide.pkl'
    hide_net = hide_moxing(hide)
    reveal = './checkpoint/epoch_1080_reveal.pkl'
    reveal_net = reveal_moxing(reveal)
    psnrsum1=0
    psnrsum2=0
    inputnum=0
    max1=0
    max2=0
    for i, (secret, cover) in enumerate(dataloader):
        secret = Variable(secret).to(device)
        cover = Variable(cover).to(device)

        output = hide_net(secret, cover)
        reveal_secret = reveal_net(output)

        save_image(secret.cpu().data[:4], fp='./result/secret.png')
        save_image(cover.cpu().data[:4], fp='./result/cover.png')
        save_image(reveal_secret.cpu().data[:4], fp='./result/reveal_secret.png')
        save_image(output.cpu().data[:4], fp='./result/output.png')
        gt1 = cv2.imread('./result/secret.png')
        img1 = cv2.imread('./result/reveal_secret.png')
        gt2 = cv2.imread('./result/cover.png')
        img2 = cv2.imread('./result/output.png')
        print(psnr1(gt1, img1))
        print(psnr1(gt2, img2))
        if max1<psnr1(gt1,img1):
            max1=psnr1(gt1,img1)
        if max2<psnr1(gt2,img2):
            max2=psnr1(gt2,img2)
        psnrsum1=psnrsum1+psnr1(gt1,img1)
        psnrsum2=psnrsum2+psnr1(gt2,img2)
        inputnum=inputnum+1
    print("AvgcoverPsnr:",psnrsum1/inputnum)
    print("MaxcoverPsnr:",max1)
    print("AvgsecretPsnr:",psnrsum2/inputnum)
    print("MaxsecretPsnr:",max2)