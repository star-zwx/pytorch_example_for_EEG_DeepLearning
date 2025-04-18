"""
Split data into train and test sets
将数据按照一定比例随机划分为训练集/测试集/验证集
"""
import torch
from torch.utils.data import Dataset, DataLoader, random_split


def data_split(dataset, train_ratio=0.8, test_ratio=0.1, val_ratio=0.1, random_state=42):
    # 设置随机种子以确保划分的可重复性
    generator = torch.Generator().manual_seed(random_state)

    # 计算每个子集的大小
    len_train = int(len(dataset) * train_ratio)
    len_test = int(len(dataset) * test_ratio)
    len_val = len(dataset) - (len_test + len_train)
    lengths = [len_train, len_val, len_test]
    # 使用 random_split 划分数据集
    train_dataset, val_dataset, test_dataset = random_split(dataset, lengths, generator=generator)

    return train_dataset, val_dataset, test_dataset
