import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd
from PIL import Image
import my_net_model
import lenet

class MyDataset1(Dataset):
    def __init__(self, csv_path, img_dir, trans, train=False):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.trans = trans
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # 注意这里iloc是中括号
        img_path = self.df.iloc[index,0]
        if self.train:
            label = self.df.iloc[index,1]
            label = torch.tensor(label)
            img = Image.open(img_path)
            if self.trans:
                img = self.trans(img)
            return img, label
        else:
            img = Image.open(img_path)
            if self.trans:
                img = self.trans(img)
            return img


def test_load_data():
    transform = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # TODO label-mapping
    train_datasets = MyDataset1("train.csv", "images", transform, train=True)
    test_datasets = MyDataset1("test.csv", "images", transform)

    train_dataloader = DataLoader(train_datasets, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_datasets, batch_size=32, shuffle=False)
    # print(train_dataloader)
    # print(test_dataloader)
    #
    # print(test_datasets.__getitem__(0))
    print(train_datasets.__getitem__(0)[0].shape)
    # print(train_datasets.__getitem__(1))
    # print(train_datasets.__getitem__(10))
    return train_dataloader,test_dataloader


def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu().

    Defined in :numref:`sec_use_gpu`"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


if __name__ == "__main__":
    train_loader,test_loader = test_load_data()
    net1 = my_net_model.get_my_lenet()
    lenet.train_ch6(net1,train_loader,test_loader,10,0.1,try_gpu())
    # df1 = pd.read_csv("train.csv")
    # print(df1.columns)
    # print(df1.sample(10))
