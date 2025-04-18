import torch
import numpy as np
from torch.utils.data import Dataset
import os


class EEGDataset(Dataset):
    def __init__(self, data_name, data_root_path, subject_No):
        """
        :param data_name: 数据集名称
        :param data_root_path: 数据集地址
        :param:subject_No: 被试编号
        描述：根据传入的数据集名称和数据集路径加载数据（单被试）
        假设每个数据集有相同的存储结构：
            -- /dataset  -数据集文件夹
                -- dataname1  -dataname1数据集
                ···
                    -- S1  - 被试1的数据
                    ···
                        -- session1  - 第一次实验的数据
                        ···
                            --trail1.npy  - 第一个trail的文件：包含数据信息和标签

        """
        self.data_name = data_name
        self.data_root_path = data_root_path
        self.subject_No = subject_No
        # 构造单被试的数据文件路径
        self.data_path_E = os.path.join(self.data_root_path, self.data_name, self.subject_No)
        self.sessionList = []
        self.data_list = []
        self.trail_list = []
        # 遍历这个被试文件夹下的的实验
        try:
            # 获取路径下的所有条目
            entries = os.listdir(self.data_path_E)
            # 过滤出文件夹
            self.sessionList = [entry for entry in entries if os.path.isdir(os.path.join(self.data_path_E, entry))]
            for session in self.sessionList:
                self.trail_list.append(os.path.join(self.data_path_E, session))
        except Exception as e:
            print("Error: {}".format("该被试文件不存在"))

        # 将所有数据的路径加入 data_list 列表
        if len(self.trail_list) != 0:
            for session in self.trail_list:
                try:
                    # 遍历目录下的所有条目
                    for root, dirs, files in os.walk(session):
                        for file in files:
                            if file.endswith('.npy'):
                                # 构建完整路径并添加到 datalist
                                full_path = os.path.join(root, file)
                                self.data_list.append(full_path)
                except Exception as e:
                    print(f"Error: {e}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):

        EEG_npy_data = np.load(self.data_list[index],allow_pickle=True).item()  # 加载一个文件的npy

        EEG_data = EEG_npy_data["data"]  # 获取EEG数据
        EEG_label = EEG_npy_data["label"]  # 获取数据对应的标签
        # 将数据和标签转换为torch.Tensor
        EEG_data = torch.tensor(EEG_data, dtype=torch.float32)
        EEG_label = torch.tensor(EEG_label, dtype=torch.long)

        return EEG_data, EEG_label
