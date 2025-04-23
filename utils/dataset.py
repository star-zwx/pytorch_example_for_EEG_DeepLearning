import torch
import numpy as np
from torch.utils.data import Dataset
import os
from sklearn.model_selection import KFold
import glob


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
                            if file.endswith('.npz'):
                                # 构建完整路径并添加到 datalist
                                full_path = os.path.join(root, file)
                                self.data_list.append(full_path)
                except Exception as e:
                    print(f"Error: {e}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):

        EEG_npy_data = np.load(self.data_list[index], allow_pickle=True)  # 加载一个文件的npz

        EEG_data = EEG_npy_data["data"].T  # 获取EEG数据
        EEG_label = EEG_npy_data["label"] - 1  # 获取数据对应的标签，我的数据标签是从1开始的，所以要减一从零开始
        # 将数据和标签转换为torch.Tensor
        EEG_data = torch.tensor(EEG_data, dtype=torch.float32)
        EEG_label = torch.tensor(EEG_label, dtype=torch.long)

        return EEG_data, EEG_label


# 重写一个用于交叉验证的数据集类(这个暂时可以不用，脑子抽了没设计好)
class EEGDatasetCrossValidation(EEGDataset):
    def __init__(self, data_name, data_root_path, subject_No, ttv_selection, current_fold, num_folds=5):
        super().__init__(data_name, data_root_path, subject_No)
        self.num_folds = num_folds
        self.__get_index_of_cross_val()  # 产生交叉验证的文件索引
        self.ttv_selection = ttv_selection  # 选择加载训练集还是测试集还是验证集
        self.txt_files_list = []
        self.current_fold = current_fold
        if self.ttv_selection == "train":
            pattern = os.path.join(self.data_path_E, 'train*.txt')
            self.txt_files_list = glob.glob(pattern)

        elif self.ttv_selection == "test":
            pattern = os.path.join(self.data_path_E, 'test*.txt')
            self.txt_files_list = glob.glob(pattern)

        else:
            pattern = os.path.join(self.data_path_E, 'val*.txt')
            self.txt_files_list = glob.glob(pattern)

        current_file = self.txt_files_list[self.current_fold]
        self.this_data_list = self._load_data_paths(current_file)

    def __getitem__(self, index):
        assert len(self.txt_files_list) == self.num_folds

        one_data_path = self.this_data_list[index]
        EEG_npy_data = np.load(one_data_path, allow_pickle=True).item()  # 加载一个文件的npy

        EEG_data = EEG_npy_data["data"]  # 获取EEG数据
        EEG_label = EEG_npy_data["label"] - 1  # 获取数据对应的标签
        # 将数据和标签转换为torch.Tensor
        EEG_data = torch.tensor(EEG_data, dtype=torch.float32)
        EEG_label = torch.tensor(EEG_label, dtype=torch.long)

        return EEG_data, EEG_label

    def __len__(self):
        return len(self.this_data_list)

    def _load_data_paths(self, file_path):
        with open(file_path, 'r') as f:
            data_paths = f.readlines()
        # 去除每行末尾的换行符
        data_paths = [path.strip() for path in data_paths]
        return data_paths

    def __get_index_of_cross_val(self):
        # 产生k折交叉验证的数据路径索引文件
        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)

        # 遍历每一折
        for fold, (train_val_index, test_index) in enumerate(kf.split(self.data_list)):
            # 进一步将训练集和验证集划分
            train_index, val_index = train_val_index[:int(0.8 * len(train_val_index))], train_val_index[int(0.8 * len(
                train_val_index)):]

            # 获取对应的数据路径
            train_paths = np.array(self.data_list)[train_index]
            val_paths = np.array(self.data_list)[val_index]
            test_paths = np.array(self.data_list)[test_index]

            # 保存到txt文件
            with open(f'{self.data_path_E}\\train_fold_{fold}.txt', 'w') as f:
                for path in train_paths:
                    f.write(f"{path}\n")

            with open(f'{self.data_path_E}\\val_fold_{fold}.txt', 'w') as f:
                for path in val_paths:
                    f.write(f"{path}\n")

            with open(f'{self.data_path_E}\\test_fold_{fold}.txt', 'w') as f:
                for path in test_paths:
                    f.write(f"{path}\n")

        print("数据地址索引已生成并保存到txt文件中。")
