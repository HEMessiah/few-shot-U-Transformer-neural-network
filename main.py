import csv
import os
import cv2
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
from pcc import pcc_calculation, pcc_calculation_t
import swin_transformer_unet_SLE_decoder


config = {
    'n_epochs': 100,  # maximum number of epochs
    'batch_size': 200,  # mini-batch size for dataloader
    'optimizer': 'Adam',  # optimization algorithm (optimizer in torch.optim)
    'optim_hparas': {  # hyper-parameters for the optimizer (depends on which optimizer you are using)
        'lr': 0.00005,  # learning rate of SGD
        # 'momentum': 0.9              # momentum for SGD
    },
    'model_save_path': 'model\\unet.pth',  # your model will be saved here
    'model_save_epoch': 'model_epoch\\',
    'train_data_path': 'D:\\wuzheyu\\pythonProject\\u-net cnn\\train_image\\',
    'train_loss_record_path': 'D:\\wuzheyu\\pythonProject\\u-net cnn\\train_image\\train_loss_record.csv',
    'dev_data_path': 'D:\\wuzheyu\\pythonProject\\u-net cnn\\dev_image\\',
    'dev_loss_record_path': 'D:\\wuzheyu\\pythonProject\\u-net cnn\\dev_image\\dev_loss_record.csv'
}
train_image_save_path = 'train_image'
dev_image_save_path = 'dev_image'

transform_sanban = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.265363, 0.117120)
])

transform_yuantu = transforms.Compose([
    transforms.ToTensor(),
])


class MyDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.name = os.listdir(os.path.join(path, 'sanban'))

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        sanban_name = self.name[index]
        sanban_path = os.path.join(self.path, 'sanban\\', sanban_name)
        yuantu_path = os.path.join(self.path, 'yuantu\\', sanban_name.replace('csv', 'png'))
        sanban_image = cv2.imread(sanban_path, cv2.IMREAD_GRAYSCALE)
        yuantu_image = cv2.imread(yuantu_path, cv2.IMREAD_GRAYSCALE)
        yuantu_image = Image.open(yuantu_path)
        yuantu_image = yuantu_image.convert('L')
        return transform_sanban(sanban_image), transform_yuantu(yuantu_image)


def prep_dataloader(path, mode, batch_size, n_jobs=2):
    ''' Generates a dataset, then is put into a dataloader. '''
    dataset = MyDataset(path)  # Construct dataset
    data_loader = DataLoader(
        dataset, batch_size,
        shuffle=(mode != 'test'), drop_last=False,
        num_workers=n_jobs, pin_memory=True)  # Construct dataloader
    return data_loader

tr_set = prep_dataloader(config['train_data_path'], 'train', config['batch_size'])
dev_set = prep_dataloader(config['dev_data_path'], 'dev', config['batch_size'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    net = swin_transformer_unet_SLE_decoder.SwinTransformerSys_SLE_decoder().to(device)
    net.device = device
    model = 1
    if os.path.exists(config['model_save_path']) and model == 1:
        net.load_state_dict(torch.load(config['model_save_path']))
        print('success')
    else:
        print('false')

    optimizer = optim.Adam(net.parameters(), **config['optim_hparas'])
    layer = nn.Sigmoid()
    loss_f = nn.BCEWithLogitsLoss()

    n_epochs = config['n_epochs']
    train_record = {'BCE_loss': [], 'pcc': []}
    dev_record = {'BCE_loss': [], 'pcc': []}
    delta_loss = 0

    for epoch in range(n_epochs):
        if epoch == 0:
            dev_BCE_loss = []
            dev_pcc = []
            net.eval()
            for batch_number, (dev_sanban_image, dev_yuantu_image) in tqdm(enumerate(dev_set)):
                dev_sanban_image, dev_yuantu_image = dev_sanban_image.to(device), dev_yuantu_image.to(device)

                with torch.no_grad():
                    dev_out_image = net(dev_sanban_image)
                dev_out_image_l = layer(dev_out_image)
                BCE_loss = loss_f(dev_out_image, dev_yuantu_image)

                pcc_mean = pcc_calculation_t(dev_out_image_l, dev_yuantu_image)  # 一个batch的平均

                dev_BCE_loss.append(BCE_loss.item())
                dev_pcc.append(pcc_mean)

            dev_BCE_loss = sum(dev_BCE_loss) / len(dev_BCE_loss)
            dev_pcc = sum(dev_pcc) / len(dev_pcc)

            dev_record['BCE_loss'].append(dev_BCE_loss)
            dev_record['pcc'].append(dev_pcc)
            if epoch != 0:
                delta_loss = dev_record['BCE_loss'][epoch] - dev_record['BCE_loss'][epoch - 1]
            print(
                f"                                        [ Dev | {epoch:03d}/{n_epochs:03d} ] loss = {dev_BCE_loss:.5f}, pcc = {dev_pcc:.5f}, delta_loss={delta_loss:.5f}")

        train_BCE_loss = []
        train_pcc = []
        net.train()
        for batch_number, (train_sanban_image, train_yuantu_image) in tqdm(enumerate(tr_set)):
            train_sanban_image, train_yuantu_image = train_sanban_image.to(device), train_yuantu_image.to(device)
            train_out_image = net(train_sanban_image)
            train_out_image_l = layer(train_out_image)

            BCE_loss = loss_f(train_out_image, train_yuantu_image)

            optimizer.zero_grad()
            BCE_loss.backward()
            optimizer.step()

            pcc_mean = pcc_calculation(train_out_image_l, train_yuantu_image)  # 一个batch的平均

            train_BCE_loss.append(BCE_loss.item())
            train_pcc.append(pcc_mean)

        train_pcc = sum(train_pcc) / len(train_pcc)
        train_BCE_loss = train_BCE_loss[0]

        train_record['BCE_loss'].append(train_BCE_loss)
        train_record['pcc'].append(train_pcc)
        if epoch != 0:
            delta_loss = train_record['BCE_loss'][epoch]-train_record['BCE_loss'][epoch-1]
        print(f"[ Train | {epoch:03d}/{n_epochs:03d} ] loss = {train_BCE_loss:.5f}, pcc = {train_pcc:.5f}, delta_loss={delta_loss:.5f}")

        dev_BCE_loss = []
        dev_pcc = []
        net.eval()
        for batch_number, (dev_sanban_image, dev_yuantu_image) in tqdm(enumerate(dev_set)):
            dev_sanban_image, dev_yuantu_image = dev_sanban_image.to(device), dev_yuantu_image.to(device)

            with torch.no_grad():
                dev_out_image = net(dev_sanban_image)
            dev_out_image_l = layer(dev_out_image)
            BCE_loss = loss_f(dev_out_image, dev_yuantu_image)

            pcc_mean = pcc_calculation_t(dev_out_image_l, dev_yuantu_image)  # 一个batch的平均

            dev_BCE_loss.append(BCE_loss.item())
            dev_pcc.append(pcc_mean)

        dev_BCE_loss = sum(dev_BCE_loss) / len(dev_BCE_loss)
        dev_pcc = sum(dev_pcc) / len(dev_pcc)

        dev_record['BCE_loss'].append(dev_BCE_loss)
        dev_record['pcc'].append(dev_pcc)
        if epoch != 0:
            delta_loss = dev_record['BCE_loss'][epoch]-dev_record['BCE_loss'][epoch-1]
        print(f"                                        [ Dev | {epoch + 1:03d}/{n_epochs:03d} ] loss = {dev_BCE_loss:.5f}, pcc = {dev_pcc:.5f}, delta_loss={delta_loss:.5f}")

        # save
        if epoch % 5 == 0:
            torch.save(net.state_dict(), os.path.join(config['model_save_epoch'], f'{epoch}.pth'))

            with open('loss_record\\train_loss_record.csv', "w") as csv_file:
                writer = csv.writer(csv_file)
                for key, value in train_record.items():
                    writer.writerow([key, value])
            with open('loss_record\\dev_loss_record.csv', "w") as csv_file:
                writer = csv.writer(csv_file)
                for key, value in dev_record.items():
                    writer.writerow([key, value])
        epoch += 1

    torch.save(net.state_dict(), os.path.join(config['model_save_path']))
