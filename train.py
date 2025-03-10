import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import cv2
import time
import numpy as np
import argparse
from DataLoader import StereoDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import stackhourglass, basic  # Припускаємо, що ці моделі вже є

# Налаштування для тренування
parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int ,default=192, help='maxium disparity')
parser.add_argument('--model', default='stackhourglass', help='select model')
parser.add_argument('--datapath', default='/media/jiaren/ImageNet/SceneFlowData/', help='datapath')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
parser.add_argument('--loadmodel', default=None, help='load model')
parser.add_argument('--savemodel', default='./', help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
args = parser.parse_args()

# Визначення пристрою для обчислень (GPU або CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Встановлюємо випадкові значення для відтворюваності
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

# Завантаження даних
train_transform = transforms.Compose([
    transforms.ToTensor(),
    # Додайте інші трансформації, якщо потрібно
])

all_left_img = '/path/to/left/images'
all_right_img = '/path/to/right/images'
all_left_disp = '/path/to/disp/images'

TrainImgLoader = DataLoader(
    StereoDataset(all_left_img, all_right_img, all_left_disp, transform=train_transform),
    batch_size=12, shuffle=True, num_workers=8, drop_last=False)

TestImgLoader = DataLoader(
    StereoDataset(all_left_img, all_right_img, all_left_disp, transform=train_transform),
    batch_size=8, shuffle=False, num_workers=4, drop_last=False)

# Вибір моделі
if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    print('No model')

# Переміщуємо модель на відповідний пристрій
model = model.to(device)

# Якщо є попередньо навчена модель
if args.loadmodel is not None:
    print('Load pretrained model')
    pretrain_dict = torch.load(args.loadmodel)
    model.load_state_dict(pretrain_dict['state_dict'])

# Оптимізатор
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

# Функція тренування
def train(imgL, imgR, disp_L):
    model.train()

    # Переміщуємо зображення на правильний пристрій (GPU або CPU)
    imgL, imgR, disp_true = imgL.to(device), imgR.to(device), disp_L.to(device)

    mask = disp_true < args.maxdisp
    mask.detach_()

    optimizer.zero_grad()

    if args.model == 'stackhourglass':
        output1, output2, output3 = model(imgL, imgR)
        output1 = torch.squeeze(output1, 1)
        output2 = torch.squeeze(output2, 1)
        output3 = torch.squeeze(output3, 1)
        loss = 0.5 * F.smooth_l1_loss(output1[mask], disp_true[mask]) + \
               0.7 * F.smooth_l1_loss(output2[mask], disp_true[mask]) + \
               F.smooth_l1_loss(output3[mask], disp_true[mask])
    elif args.model == 'basic':
        output = model(imgL, imgR)
        output = torch.squeeze(output, 1)
        loss = F.smooth_l1_loss(output[mask], disp_true[mask])

    loss.backward()
    optimizer.step()

    return loss.data

# Функція тестування
def test(imgL, imgR, disp_true):
    model.eval()

    imgL, imgR, disp_true = imgL.to(device), imgR.to(device), disp_true.to(device)

    mask = disp_true < 192

    if imgL.shape[2] % 16 != 0:
        times = imgL.shape[2] // 16
        top_pad = (times + 1) * 16 - imgL.shape[2]
    else:
        top_pad = 0

    if imgL.shape[3] % 16 != 0:
        times = imgL.shape[3] // 16
        right_pad = (times + 1) * 16 - imgL.shape[3]
    else:
        right_pad = 0

    imgL = F.pad(imgL, (0, right_pad, top_pad, 0))
    imgR = F.pad(imgR, (0, right_pad, top_pad, 0))

    with torch.no_grad():
        output3 = model(imgL, imgR)
        output3 = torch.squeeze(output3)

    if top_pad != 0:
        img = output3[:, top_pad:, :]
    else:
        img = output3

    if len(disp_true[mask]) == 0:
        loss = 0
    else:
        loss = F.l1_loss(img[mask], disp_true[mask])

    return loss.data.cpu()

# Функція для регулювання навчальної швидкості
def adjust_learning_rate(optimizer, epoch):
    lr = 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Основна функція
def main():
    start_full_time = time.time()
    for epoch in range(0, args.epochs):
        print(f'This is {epoch}-th epoch')
        total_train_loss = 0
        adjust_learning_rate(optimizer, epoch)

        # Тренування
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):
            loss = train(imgL_crop, imgR_crop, disp_crop_L)
            print(f'Iter {batch_idx} training loss = {loss:.3f}')
            total_train_loss += loss
        print(f'epoch {epoch} total training loss = {total_train_loss/len(TrainImgLoader):.3f}')

        # Збереження моделі
        savefilename = os.path.join(args.savemodel, f'checkpoint_{epoch}.tar')
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'train_loss': total_train_loss / len(TrainImgLoader),
        }, savefilename)

    print(f'full training time = {(time.time() - start_full_time) / 3600:.2f} HR')

    # Тестування
    total_test_loss = 0
    for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
        test_loss = test(imgL, imgR, disp_L)
        print(f'Iter {batch_idx} test loss = {test_loss:.3f}')
        total_test_loss += test_loss

    print(f'total test loss = {total_test_loss / len(TestImgLoader):.3f}')

    # Збереження результатів тестування
    savefilename = os.path.join(args.savemodel, 'testinformation.tar')
    torch.save({
        'test_loss': total_test_loss / len(TestImgLoader),
    }, savefilename)

if __name__ == "__main__":
    main()
