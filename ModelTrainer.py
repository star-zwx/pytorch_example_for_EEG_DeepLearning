from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import *
from models import EEGNet, MCNN, EEGInception, TransNet
from tqdm import tqdm

from sklearn.metrics import accuracy_score, recall_score, f1_score


class ModelTrainer():
    def __init__(self, para_config_dir):
        # 先对超参数做一些声明
        self.para_config_dir = para_config_dir
        self.model_name = self.para_config_dir["model_name"]  # 获取模型名称
        self.data_name = self.para_config_dir["data_name"]  # 获取数据集名称
        self.data_root_path = self.para_config_dir["data_root_path"]  # 获取数据集根地址
        self.learning_rate = self.para_config_dir["learning_rate"]  # 获取初始学习率
        self.batch_size = self.para_config_dir["batch_size"]  # 获取batch_size
        self.epochs = self.para_config_dir["epochs"]  # 获取epochs
        self.repeat_time = self.para_config_dir["repeat_time"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设置训练设备
        self.optimizer_name = self.para_config_dir["optimizer"]  # 获取优化器名称
        self.loss_name = self.para_config_dir["loss_function"]  # 获取损失函数名称
        self.activation_func_name = self.para_config_dir["activation_function"]  # 获取激活函数名称
        self.early_stopping = self.para_config_dir["early_stopping"]  # 获取是否使用早停策略
        self.learning_rate_decay = self.para_config_dir["learning_rate_decay"]  # 获取是否采取学习率衰减
        self.init_seed = self.para_config_dir["init_seed"]  # 获取随机种子
        self.data_split = self.para_config_dir["data_split"]  # 获取数据划分的比例
        self.Subject_sigal = self.para_config_dir["Subject_sigal"]  # 获取是否用单个被试或者全被试训练
        self.data_logs_path = self.para_config_dir["save_logs_path"]  # 训练过程中的数据保存位置
        self.save_model_path = self.para_config_dir["save_model_path"]  # 模型保存地址
        self.Cross_validation = self.para_config_dir["Cross_validation"]  # 是否选择交叉验证的策略的配置
        self.EEGDataInfo = self.para_config_dir["EEGDataInfo"] # 获取脑电数据的信息

        # 声明数据加载器，训练集，测试集，验证集
        self.dataLoader_train = None
        self.dataLoader_test = None
        self.dataLoader_valid = None
        self.model = None

        # 设置随机种子
        torch.manual_seed(self.init_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.init_seed)

        # 这里需要修改为实际的模型名称,注意路径关系
        self.model_dict = {
            "EEGNet": EEGNet.EEGNet,
            "EEGNet2": EEGNet.EEGNet,
            "MCNN": MCNN.MCNN,
            "EEGInception": EEGInception.Inception_EEG,
            "TransNet": TransNet.TransNet,
        }  #在这里添加需要在上面导入模块那导入对应的python模块

    def data_loader(self):
        if self.Subject_sigal["sigal"] == "yes":
            all_dataset = EEGDataset(self.data_name, self.data_root_path, self.Subject_sigal["subject_name"])  # 加载数据集
            train_dataset, val_dataset, test_dataset = data_split(all_dataset, self.data_split["train"],
                                                                  self.data_split["val"], self.data_split["test"])  #
            print(train_dataset, val_dataset, test_dataset)
            # 划分数据集
            self.dataLoader_train = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            self.dataLoader_valid = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)
            self.dataLoader_test = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

    def model_train(self):
        num_repeats = self.repeat_time
        # 获取当前时间，并格式化为字符串
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(self.data_logs_path, f"{self.model_name}-{current_time}")

        all_results = {}
        # 输出一些基本的信息
        print("模型名称：", self.model_name)
        print("数据集名称：", self.data_name)
        print("训练设备：", self.device)
        for repeat in range(1, num_repeats + 1):
            print(f"\n=======================Training Round  {repeat}  /  {self.repeat_time} =====================")
            self.model = self.initialize_model()
            criterion = self.initialize_loss()
            optimizer = self.initialize_optimizer(self.model)

            repeat_key = f"repeat_{repeat}"
            all_results[repeat_key] = {"epochs": [], "summary": {}}
            best_val_acc = 0
            best_model_state = None

            for epoch in range(1, self.epochs + 1):
                self.model.train()
                train_loss, correct, total = 0, 0, 0

                for data, labels in self.dataLoader_train:
                    data, labels = data.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = self.model(data)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                avg_train_loss = train_loss / len(self.dataLoader_train)
                train_acc = correct / total

                val_acc, val_loss = self.model_evaluation(self.dataLoader_test, criterion)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = self.model.state_dict()

                all_results[repeat_key]["epochs"].append({
                    "epoch": epoch,
                    "train_loss": avg_train_loss,
                    "val_loss": val_loss,
                    "train_acc": train_acc,
                    "val_acc": val_acc
                })

                print(
                    f"✅ Repeat {repeat}/{num_repeats} | 📈 Epoch [{epoch}/{self.epochs}] "
                    f"🔧 Train Loss: {avg_train_loss:.4f} | 🎯 Train Acc: {train_acc:.4f} | "
                    f"📊 Val Loss: {val_loss:.4f} | 💡 Val Acc: {val_acc:.4f}"
                )

                # 加载最佳模型评估测试集
            self.model.load_state_dict(best_model_state)
            test_acc, _ = self.model_evaluation(self.dataLoader_test, criterion)
            y_true, y_pred = self.model_test(self.model, self.dataLoader_test)
            recall = recall_score(y_true, y_pred, average="macro")
            f1 = f1_score(y_true, y_pred, average="macro")

            all_results[repeat_key]["summary"] = {
                "best_val_acc": best_val_acc,
                "final_test_acc": test_acc,
                "recall": recall,
                "f1": f1
            }

            # 保存最优模型
            self.model_save(best_model_state, self.save_model_path, repeat, current_time)

        # 保存整个 JSON 文件
        save_json_path = os.path.join(save_path, "logs.json")
        save_json(all_results, save_json_path)

    def model_test(self, model, dataloader):
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for data, labels in dataloader:
                data = data.to(self.device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                y_pred.extend(predicted.cpu().numpy())
                y_true.extend(labels.numpy())
        return y_true, y_pred

    def model_evaluation(self, dataloader, criterion):

        self.model.eval()
        total_loss, correct, total = 0, 0, 0

        with torch.no_grad():
            for data, labels in dataloader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(data)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(dataloader)
        acc = correct / total
        return acc, avg_loss

    def initialize_model(self):
        # 字典索引模型（说实话有点啰嗦了，但是很strong）
        model = self.model_dict[self.model_name]().to(self.device)
        return model

    def initialize_loss(self):
        # 初始化损失函数，假设损失函数为交叉熵损失
        if self.loss_name == "CrossEntropyLoss":
            criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss function: {self.loss_name}")
        return criterion

    def initialize_optimizer(self, model):
        # 初始化优化器，假设优化器为 Adam
        if self.optimizer_name == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")
        return optimizer

    def model_save(self, model_state_dict, save_path, repeat_index, current_time):
        filename = f"best_model_repeat{repeat_index}-{current_time}.pth"
        full_path = os.path.join(save_path, filename)
        torch.save(model_state_dict, full_path)
        print(f"💾 Best model for Repeat {repeat_index} saved to: {full_path}")


class ModelTrainCrossValidation(ModelTrainer):
    def __init__(self, para_config_dir):
        super().__init__(para_config_dir)
        self.dataset = None

    def data_loader(self):
        if self.Subject_sigal["sigal"] == "yes":
            all_dataset = EEGDataset(self.data_name, self.data_root_path, self.Subject_sigal["subject_name"])  # 加载数据集
            self.dataset = all_dataset  # 将整个数据集赋值给 self.dataset

    def model_train(self):
        num_repeats = self.repeat_time
        k_folds = self.Cross_validation["Fload_Num"]  # 设置k折交叉验证的折数
        kfold = KFold(n_splits=k_folds, shuffle=True)

        # 获取当前时间，并格式化为字符串
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(self.data_logs_path, f"{self.model_name}-{current_time}")

        all_results = {}

        for repeat in range(1, num_repeats + 1):
            print(f"\n=======================Training Round  {repeat}  /  {self.repeat_time} =====================")

            fold_results = {}
            for fold, (train_ids, test_ids) in enumerate(kfold.split(self.dataset)):
                print(f"\n----------------------- Fold {fold + 1} / {k_folds} -----------------------")

                # 创建数据加载器
                train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
                test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

                train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size,
                                                           sampler=train_subsampler)
                test_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size,
                                                          sampler=test_subsampler)

                self.model = self.initialize_model()
                criterion = self.initialize_loss()
                optimizer = self.initialize_optimizer(self.model)

                fold_key = f"fold_{fold + 1}"
                fold_results[fold_key] = {"epochs": [], "summary": {}}
                best_val_acc = 0
                best_model_state = None

                for epoch in range(1, self.epochs + 1):
                    self.model.train()
                    train_loss, correct, total = 0, 0, 0

                    for data, labels in train_loader:
                        data, labels = data.to(self.device), labels.to(self.device)
                        optimizer.zero_grad()
                        outputs = self.model(data)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        train_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                    avg_train_loss = train_loss / len(train_loader)
                    train_acc = correct / total

                    val_acc, val_loss = self.model_evaluation(test_loader, criterion)

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_model_state = self.model.state_dict()

                    fold_results[fold_key]["epochs"].append({
                        "epoch": epoch,
                        "train_loss": avg_train_loss,
                        "val_loss": val_loss,
                        "train_acc": train_acc,
                        "val_acc": val_acc
                    })

                    print(
                        f"✅ Repeat {repeat}/{num_repeats} | Fold {fold + 1}/{k_folds} | 📈 Epoch [{epoch}/{self.epochs}] "
                        f"🔧 Train Loss: {avg_train_loss:.4f} | 🎯 Train Acc: {train_acc:.4f} | "
                        f"📊 Val Loss: {val_loss:.4f} | 💡 Val Acc: {val_acc:.4f}"
                    )

                # 加载最佳模型评估测试集
                self.model.load_state_dict(best_model_state)
                test_acc, _ = self.model_evaluation(test_loader, criterion)
                y_true, y_pred = self.model_test(self.model, test_loader)
                recall = recall_score(y_true, y_pred, average="macro")
                f1 = f1_score(y_true, y_pred, average="macro")

                fold_results[fold_key]["summary"] = {
                    "best_val_acc": best_val_acc,
                    "final_test_acc": test_acc,
                    "recall": recall,
                    "f1": f1
                }

                # 保存最优模型
                self.model_save(best_model_state, self.save_model_path, repeat, current_time, fold + 1)

            all_results[f"repeat_{repeat}"] = fold_results

        # 保存整个 JSON 文件
        save_json_path = os.path.join(save_path, "logs.json")
        save_json(all_results, save_json_path)

    def model_save(self, model_state_dict, save_path, repeat_index, current_time, fold):
        filename = f"best_model_repeat{repeat_index}-{fold}-{current_time}.pth"
        full_path = os.path.join(save_path, filename)
        torch.save(model_state_dict, full_path)
        print(f"💾 Best model for Repeat {repeat_index} saved to: {full_path}")
