"""
这是画图的类
从训练过程中保存的json文件中读取训练信息
分别对不同类型的数据进行画图表示
"""
from utils import read_json
import numpy as np
import matplotlib.pyplot as plt
import sys


def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height}', ha='center', va='bottom')


class DrawData:
    def __init__(self, json_file_name, ):
        # 先声明一些需要的变量
        self.json_file = json_file_name
        self.repeat_num = None
        self.picture_type = None
        self.summary_repeat_num = []  # 绘制每一次实验的总的指标的实验轮次列表
        # 读取json文件
        self.json_data = read_json(json_file_name)

    def draw_picture(self, repeat_num=0, picture_type='train_loss'):
        self.repeat_num = repeat_num
        self.picture_type = picture_type
        # 绘制变化曲线
        repeat_name = []  # 一维数组
        repeat_value_list = []  # 二维数组
        if self.repeat_num == 0:  # 输入的训练轮次编号如果是0，则绘制全部的变化
            if type(self.json_data) is dict:
                data_list = list(self.json_data.items())
                for item1 in data_list:
                    epochs_dir = item1[1]
                    label_troup = item1[0]
                    repeat_name.append(label_troup)  # repeat的名称，代表训练编号
                    epochs_datas = epochs_dir['epochs']
                    repeat_value = []
                    for epoch_data in epochs_datas:
                        repeat_value.append(epoch_data[self.picture_type])

                    repeat_value_list.append(repeat_value)
        if self.repeat_num != 0:  # 如何非零，根据输入的repeat_num来指定绘制哪一次训练的曲线
            if type(self.json_data) is dict:
                data_list = list(self.json_data.items())
                counter_repeat = data_list[self.repeat_num - 1]
                repeat_name.append(counter_repeat[0])
                epochs_dir = counter_repeat[1]
                epochs_datas = epochs_dir['epochs']
                repeat_value = []
                for epoch_data in epochs_datas:
                    repeat_value.append(epoch_data[self.picture_type])

                repeat_value_list.append(repeat_value)

        # 绘制多个图
        for repeat_one_name, repeat_one_value in zip(repeat_name, repeat_value_list):
            self.drawing_sigal(repeat_one_value, repeat_one_name)

    def drawing_sigal(self, epoch_value, label_group):
        """
        绘制折线图，展示 epoch_value 随着 epochs 的变化。
        :param epoch_value: 数据列表，表示每个 epoch 的值。
        :param label_group: 图像的标题。
        """
        # 检查 self.picture_type 是否为空
        if not self.picture_type:
            self.picture_type = "Value"

        # 生成 epochs 列表
        epochs = list(range(1, len(epoch_value) + 1))

        # 创建图形
        plt.figure(figsize=(10, 6))

        # 绘制折线图
        plt.plot(epochs, epoch_value, marker='o', linestyle='-', color='b', label=self.picture_type)

        # 添加标题和标签
        plt.title(label_group)
        plt.xlabel('Epochs')
        plt.ylabel(self.picture_type)

        # 添加网格
        plt.grid(True, linestyle='--', alpha=0.7)

        # 添加图例
        plt.legend()

        # 可选：在每个点上显示具体的值
        if len(epoch_value) <= 20:  # 如果数据点较少，显示具体值
            for i, loss in enumerate(epoch_value):
                plt.text(epochs[i], epoch_value[i], f'{loss:.4f}', fontsize=9, ha='right', va='bottom')

        # 显示图像（可选）
        plt.show()

    def draw_summary_picture(self, summary_repeat_num=None):  # summary_repeat_num 参数不填的话默认全画
        all_summary_data = {"repeat_name": [], "best_val_acc": [], "final_test_acc": [], "recall": [], "f1": []}  #
        # 用来保存所有实验的最终数据
        if type(self.json_data) is dict:
            data_list = list(self.json_data.items())
            all_summary_data["repeat_name"] = list(range(1, len(data_list) + 1))  # 先产生绘图的横坐标的列表（全部数据的）
            for item2 in data_list:
                item2_data = item2[1]
                all_summary_data["best_val_acc"].append(item2_data["summary"]["best_val_acc"])
                all_summary_data["final_test_acc"].append(item2_data["summary"]["final_test_acc"])
                all_summary_data["recall"].append(item2_data["summary"]["recall"])
                all_summary_data["f1"].append(item2_data["summary"]["f1"])

        if summary_repeat_num is None:
            # 默认绘制所有实验的结果
            # print(all_summary_data)
            self.drawing_summay(all_summary_data)
        else:
            # 如果参数填入的是一个列表，代表绘制某几次实验的总结果
            if type(summary_repeat_num) is list:
                all_summary_data["repeat_name"] =summary_repeat_num
                all_summary_data["best_val_acc"] = [all_summary_data["best_val_acc"][i] for i in summary_repeat_num]
                all_summary_data["final_test_acc"] = [all_summary_data["final_test_acc"][i] for i in summary_repeat_num]
                all_summary_data["recall"] = [all_summary_data["recall"][i] for i in summary_repeat_num]
                all_summary_data["f1"] = [all_summary_data["f1"][i] for i in summary_repeat_num]
                print(all_summary_data)
            self.drawing_summay(all_summary_data)

    def drawing_summay(self, data):

        # 提取数据
        names = data["repeat_name"]
        best_val_acc = data["best_val_acc"]
        final_test_acc = data["final_test_acc"]
        f1 = data["f1"]
        recall = data["recall"]

        # 设置柱状图的宽度
        bar_width = 0.2

        # 设置柱状图的位置
        r1 = np.arange(len(names))
        r2 = [x + bar_width for x in r1]
        r3 = [x + bar_width for x in r2]
        r4 = [x + bar_width for x in r3]


        # 创建柱状图
        plt.figure(figsize=(10, 6))

        # 绘制柱状图
        bars1 = plt.bar(r1, best_val_acc, color='b', width=bar_width, edgecolor='white', label='Best Val Acc')
        bars2 = plt.bar(r2, final_test_acc, color='g', width=bar_width, edgecolor='white', label='Final Test Acc')
        bars3 = plt.bar(r3, f1, color='r', width=bar_width, edgecolor='white', label='F1')
        bars4 = plt.bar(r4, recall, color='y', width=bar_width, edgecolor='white', label='recall')

        add_labels(bars1)
        add_labels(bars2)
        add_labels(bars3)
        add_labels(bars4)

        # 添加标题和标签
        plt.title('Performance Metrics')
        plt.xlabel('Name')
        plt.ylabel('Values')
        plt.xticks([r + bar_width for r in range(len(names))], names)
        plt.legend()

        # 添加网格
        plt.grid(True, linestyle='--', alpha=0.7)

        # 显示图像
        plt.tight_layout()

        plt.show()  # 显示图片
