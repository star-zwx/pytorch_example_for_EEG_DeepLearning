"""
main.py执行逻辑：
首先在指定路径下读取模型训练的一系列参数 --config->model_config
实例化ModelTrainer类，将读取的参数字典作为参数传入类
调用ModelTrainer类中的函数进行训练/测试/
调用DrawData类中的函数进行绘图

"""

from ModelTrainer import *
from utils import *

# 读取json文件获取训练的超参数设置
json_file = './configs/model_config.json'
para_config_dir = read_json(json_file)


def model_trainer(para_config_dir):
    # 模型的训练
    my_model = ModelTrainer(para_config_dir)  # 实例化模型训练的类
    my_model.data_loader()  # 数据加载
    my_model.model_train()  # 模型训练


def model_trainer_cross_validation(para_config_dir):  # 适合交叉验证的
    my_model = ModelTrainCrossValidation(para_config_dir)
    my_model.data_loader()
    my_model.model_train()


def result_drawer(json_file):
    # 对保存的训练过程的数据画图，注意：这个过程不适合交叉验证产生的数据绘图
    draw_picture = DrawData(json_file)  # 参数是日志文件的路径
    """
        对draw_picture()方法的参数填写说明
        @ repeat_num：如果填写的是0，将会绘制所有实验的变化曲线（如损失的变化），如果填写其他数字，表示绘制某次实验的变化曲线（注意这里索引值是从1开始，如果填写1表示绘制第一次实验的曲线）
        @ picture_type ： 这个参数表示你需要绘制哪种曲线可选（train_loss,val_loss,val_acc,train_acc）
    """
    draw_picture.draw_picture(1, 'val_loss')  # 两个参数:repeat_num和picture_type

    """
    对.draw_summary_picture()方法的参数填写说明
    该方法是绘制一些实验结束后的指标，包括best_val_acc，final_test_acc，f1，recall
    参数有两种填法，如果不填，默认绘制所有的实验的数据
    如果填写一个列表，列表的数值分别代表第几次实验，按照列表的索引去绘制需要的实验的数据，索引从0开始
    """
    draw_picture.draw_summary_picture([0, 5, 6])


if __name__ == '__main__':
    # model_trainer(para_config_dir)  #由于需要在模型训练完才能保存训练数据，所以训练和画图的代码不能同时执行
    # model_trainer_cross_validation(para_config_dir)
    # result_drawer(r"logs/EEGNet-20250418_123734/logs.json")

    pass
