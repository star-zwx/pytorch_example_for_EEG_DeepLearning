import numpy as np
import os
from utils import *
import plotly.graph_objects as go
import pandas as pd
from dataset import *

def createdata():
    save_dir = "eeg_dataset"
    os.makedirs(save_dir, exist_ok=True)

    for i in range(10):
        sample = {
            "data": np.random.randn(36, 2500).astype(np.float32),  # 假设标准正态脑电信号
            "label": np.random.randint(0, 3)  # 标签为 0、1、2
        }
        file_path = os.path.join(save_dir, f"sample_{i}.npy")
        np.save(file_path, sample)

    print(f"✅ 10 个 EEG 样本保存到 {save_dir}/ 下")


def jsondata():
    data_list = []
    json_data = read_json(r"logs/EEGNet-20250417_174017/logs.json")
    print(json_data)
    if type(json_data) == dict:
        data_list = list(json_data.items())
        print(data_list)

    print(data_list[0][1]['epochs'][0]['train_loss'])
    print(data_list[0][1]['summary'])


def class_json():
    clss_draw = DrawData(r"logs/EEGNet-20250417_174017/logs.json")
    clss_draw.draw_picture(1, 'val_loss')
    clss_draw.draw_summary_picture([0,5,6])


# 在柱状图上方标注数值
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height}', ha='center', va='bottom')


def draw_table():
    import matplotlib.pyplot as plt
    import numpy as np

    # 数据
    data = {
        "Name": [1, 2, 3,4,5,6,7,8,9,10],
        "best_val_acc": [25, 30, 35,12,15,45,60,75,80,90],
        "final_test_acc": [1, 1.5, 3.5,3.6,4.5,5.5,6.5,7.5,8.5,9.5],
        "f1": [5, 8, 6.5,6.2,5.5,5.5,5.5,5.5,5.5,5.5]
    }

    # 提取数据
    names = data["Name"]
    best_val_acc = data["best_val_acc"]
    final_test_acc = data["final_test_acc"]
    f1 = data["f1"]

    # 设置柱状图的宽度
    bar_width = 0.2

    # 设置柱状图的位置
    r1 = np.arange(len(names))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    # 创建柱状图
    plt.figure(figsize=(10, 6))

    # 绘制柱状图
    bars1 = plt.bar(r1, best_val_acc, color='b', width=bar_width, edgecolor='white', label='Best Val Acc')
    bars2 = plt.bar(r2, final_test_acc, color='g', width=bar_width, edgecolor='white', label='Final Test Acc')
    bars3 = plt.bar(r3, f1, color='r', width=bar_width, edgecolor='white', label='F1')

    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)

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

def jiaocha():
    jioacha = EEGDatasetCrossValidation(data_name = "dataname1", data_root_path="./dataset/", subject_No = "S1", ttv_selection = "train", current_fold = 0, num_folds=5)


if __name__ == '__main__':
    # jsondata()
    # class_json()
    # draw_table()
