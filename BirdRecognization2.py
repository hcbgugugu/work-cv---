import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt  # plt 用于显示图片
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.models as model
from torchvision.transforms import ToPILImage
import torchvision.transforms as transforms
import os
from dataset import CUB
import transforms
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
torch.cuda.empty_cache()

# 1.1 实现load_data函数加载图片名称与标签的加载，并使用torch.utils.data接口将其封装成程序可用的数据集类OwnDataset。
def load_dir(directory, labstart=0):  # 获取所有directory中的所有图与标签
    # 返回path指定的文件夹所包含的文件或文件名的名称列表
    strlabels = os.listdir(directory)
    # 对标签进行排序，以便训练和验证按照相同的顺序进行:在不同的操作系统中，加载文件夹的顺序可能不同。目录不同的情况会导致在不同的操作系统中，模型的标签出现串位的现象。所以需要对文件夹进行排序，保证其顺序的一致性。
    strlabels.sort()
    # 创建文件标签列表
    file_labels = []
    for i, label in enumerate(strlabels):
        print(label)
        jpg_names = glob.glob(os.path.join(directory, label, "*.jpg"))
        print(jpg_names)
        # 加入列表
        file_labels.extend(zip(jpg_names, [i + labstart] * len(jpg_names)))
    return file_labels, strlabels

def load_data(dataset_path):  # 定义函数load_data函数完成对数据集中图片文件名称和标签的加载。
    # 该函数可以实现两层文件夹的嵌套结构。其中，外层结构使用load_data函数进行遍历，内层结构使用load_dir函进行遍历。
    sub_dir = sorted(os.listdir(
        dataset_path))  # 跳过子文件夹:在不同的操作系统中，加载文件夹的顺序可能不同。目录不同的情况会导致在不同的操作系统中，模型的标签出现串位的现象。所以需要对文件夹进行排序，保证其顺序的一致性。
    start = 1  # 第0类是none
    tfile_lables, tstrlabels = [], ['none']  # 在制作标签时，人为地在前面添加了一个序号为0的none类。这是一个训练图文类模型的技巧，为了区分模型输出值是0和预测值是0这两种情况。
    for i in sub_dir:
        directory = os.path.join(dataset_path, i)
        if os.path.isdir(directory) == False:  # 只处理文件夹中的数据
            print(directory)
            continue
        file_labels, strlables = load_dir(directory, labstart=start)
        tfile_lables.extend(file_labels)
        tstrlabels.extend(strlables)
        start = len(strlables)
    # 将数据路径与标签解压缩,把数据路径和标签解压缩出来
    filenames, labels = zip(*tfile_lables)
    return filenames, labels, tstrlabels


# 1.2 实现自定义数据集OwnDataset
def default_loader(path):  # 定义函数加载图片
    return Image.open(path).convert('RGB')

# 1.3 测试数据集：在完成数据集的制作之后，编写代码对其进行测试。

dataset_path = r'./cub-200-2011/Caltech-UCSD Birds-200-2011/CUB_200_2011/'  # 加载数据集路径
filenames, labels, classes = load_data(dataset_path)  # 调用load_data函数对数据集中图片文件名称和标签进行加载，其返回对象classes中包含全部的类名。
# 打乱数据顺序
# 对数据文件列表的序号进行乱序划分，分为测试数据集和训练数集两个索引列表。该索引列表会传入OwnDataset类做成指定的数据集。
np.random.seed(0)
label_shuffle_index = np.random.permutation(len(labels))
label_train_num = (len(labels) // 10) * 8  # 划分训练数据集和测试数据集
train_list = label_shuffle_index[0:label_train_num]
test_list = label_shuffle_index[label_train_num:]

IMAGE_SIZE = 448
TRAIN_MEAN = [0.48560741861744905, 0.49941626449353244, 0.43237713785804116]
TRAIN_STD = [0.2321024260764962, 0.22770540015765814, 0.2665100547329813]
TEST_MEAN = [0.4862169586881995, 0.4998156522834164, 0.4311430419332438]
TEST_STD = [0.23264268069040475, 0.22781080253662814, 0.26667253517177186]

path = './cub-200-2011/Caltech-UCSD Birds-200-2011/CUB_200_2011'
# 定义数据的预处理方法
train_transforms = transforms.Compose([
    transforms.ToCVImage(),
    transforms.RandomResizedCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(TRAIN_MEAN, TRAIN_STD)
])

test_transforms = transforms.Compose([
    transforms.ToCVImage(),
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(TEST_MEAN, TEST_STD)
])

train_dataset = CUB(
    path,
    train=True,
    transform=train_transforms,
    target_transform=None
)
# print(len(train_dataset))
train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    num_workers=0,
    shuffle=True
)

test_dataset = CUB(
    path,
    train=False,
    transform=test_transforms,
    target_transform=None
)

val_loader = DataLoader(
    test_dataset,
    batch_size=2,
    num_workers=0,
    shuffle=True
)

# 1.4 获取并改造ResNet模型：获取ResNet模型，并加载预训练模型的权重。将其最后一层（输出层）去掉，换成一个全新的全连接层，该全连接层的输出节点数与本例分类数相同。
# 指定设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# get_ResNet函数，获取预训练模型，可指定pretrained=True来实现自动下载预训练模型，也可指定loadfile来从本地路径加载预训练模型。
def get_ResNet(classes, pretrained=True, loadfile=None):
    ResNet = model.resnet101(pretrained)  # 自动下载官方的预训练模型
    if loadfile != None:
        ResNet.load_state_dict(torch.load(loadfile))  # 加载本地模型
    # 将所有的参数层进行冻结：设置模型仅最后一层可以进行训练，使模型只针对最后一层进行微调。
    for param in ResNet.parameters():
        param.requires_grad = False
    # 输出全连接层的信息
    print(ResNet.fc)
    x = ResNet.fc.in_features  # 获取全连接层的输入
    ResNet.fc = nn.Linear(x, len(classes))  # 定义一个新的全连接层
    print(ResNet.fc)  # 最后输出新的模型
    return ResNet


ResNet = get_ResNet(classes)  # 实例化模型
ResNet.to(device=device)

# 1.5 定义损失函数、训练函数及测试函数，对模型的最后一层进行微调。
criterion = nn.CrossEntropyLoss()
# 指定新加的全连接层的学习率
optimizer = torch.optim.Adam([{'params': ResNet.fc.parameters()}], lr=0.01)


def train(model, device, train_loader, epoch, optimizer):  # 定义训练函数
    model.train()
    allloss = []
    for batch_idx, data in enumerate(train_loader):
        x, y = data
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        allloss.append(loss.item())
        optimizer.step()
    print('Train Epoch:{}\t Loss:{:.6f}'.format(epoch, np.mean(allloss)))  # 输出训练结果


def test(model, device, val_loader):  # 定义测试函数
    model.eval()
    test_loss = []
    correct = []
    with torch.no_grad():  # 使模型在运行时不进行梯度跟踪，可以减少模型运行时对内存的占用。
        for i, data in enumerate(val_loader):
            x, y = data
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)
            test_loss.append(criterion(y_hat, y).item())  # 收集损失函数
            pred = y_hat.max(1, keepdim=True)[1]  # 获取预测结果
            correct.append(pred.eq(y.view_as(pred)).sum().item() / pred.shape[0])  # 收集精确度
    print('\nTest:Average loss:{:,.4f},Accuracy:({:,.0f}%)\n'.format(np.mean(test_loss),
                                                                     np.mean(correct) * 100))  # 输出测试结果
    if np.mean(correct) > maxacc:
        maxacc = np.mean(correct)
        torch.save(ResNet.state_dict(), secondmodepth)

maxacc = 0.72 # 记得改！！！！！！！！
# 迁移学习的两个步骤如下
if __name__ == '__main__':
    # 迁移学习步骤①：固定预训练模型的特征提取部分，只对最后一层进行训练，使其快速收敛。
    firstmodepth = './cub-200-2011/firstmodepth_1.pth'  # 定义模型文件的地址
    if os.path.exists(firstmodepth) == False:
        print("—————————固定预训练模型的特征提取部分，只对最后一层进行训练，使其快速收敛—————————")

        for epoch in range(1, 2):  # 迭代两次
            train(ResNet, device, train_loader, epoch, optimizer)
            test(ResNet, device, val_loader)
        """
        for epoch in range(1, 21):
            train(ResNet, device, train_loader, epoch, optimizer)
        test(ResNet, device, val_loader)
        """
        # 保存模型
        torch.save(ResNet.state_dict(), firstmodepth)
    # 1.6 使用退化学习率对模型进行全局微调
    # 迁移学习步骤②：使用较小的学习率，对全部模型进行训练，并对每层的权重进行细微的调节，即将模型的每层权重都设为可训练，并定义带有退化学习率的优化器。（1.6部分）
    secondmodepth = './cub-200-2011/firstmodepth_2.pth'
    optimizer2 = optim.SGD(ResNet.parameters(), lr=0.001,
                           momentum=0.9)  # 第198行代码定义带有退化学习率的SGD优化器。该优化器常用来对模型进行手动微调。有实验表明，使用经过手动调节的SGD优化器，在训练模型的后期效果优于Adam优化器。
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer2, step_size=2,
                                           gamma=0.9)  # 由于退化学习率会在训练过程中不断地变小，为了防止学习率过小，最终无法进行权重需要对其设置最小值。当学习率低于该值时，停止对退化学习率的操作。
    for param in ResNet.parameters():  # 所有参数设计为可训练
        param.requires_grad = True
    if os.path.exists(secondmodepth):
        ResNet.load_state_dict(torch.load(secondmodepth))  # 加载本地模型
    else:
        ResNet.load_state_dict(torch.load(firstmodepth))  # 加载本地模型
    print("____使用较小的学习率，对全部模型进行训练，定义带有退化学习率的优化器______")
    for epoch in range(1, 25):
        train(ResNet, device, train_loader, epoch, optimizer2)
        if optimizer2.state_dict()['param_groups'][0]['lr'] > 0.00001:
            exp_lr_scheduler.step()
            print("___lr:", optimizer2.state_dict()['param_groups'][0]['lr'])
        test(ResNet, device, val_loader)

    # 保存模型
    torch.save(ResNet.state_dict(), secondmodepth)
torch.cuda.empty_cache()
