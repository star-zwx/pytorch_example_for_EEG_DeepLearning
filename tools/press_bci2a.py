import numpy as np
from scipy.io import loadmat
import io
import os
import glob


def press_bci2a():
    # 需要处理的数据被试文件夹
    root_file_path = r"..\dataset\BCI_2a"
    subject_num = 9

    # 构造被试的文件夹路径便于加载npy文件
    for subject in range(1, subject_num + 1):
        npy_files = []
        # 被试文件夹
        subject_file_path = os.path.join(root_file_path, 'S' + str(subject), "train")
        npy_files = glob.glob(os.path.join(subject_file_path, '*.npy'))

        print(npy_files)
        # 加载每个每个数据并重构
        for npy_file in npy_files:
            current_data = np.load(npy_file, allow_pickle=True)
            current_data = current_data.astype(np.float32)
            sample = {
                'data': current_data,
                'label': int(npy_file.split('\\')[-1][1])
            }
            print(sample)
            # print(**sample)
            npz_filename = npy_file.split('\\')[-1].split('.')[0]
            save_path = os.path.join(subject_file_path, npz_filename + '.npz')
            # 保存在原文件中
            np.savez(save_path, **sample)
def remove_npy():
    # 需要处理的数据被试文件夹
    root_file_path = r"..\dataset\BCI_2a"
    subject_num = 9

    # 构造被试的文件夹路径便于加载npy文件
    for subject in range(1, subject_num + 1):
        npy_files = []
        # 被试文件夹
        subject_file_path = os.path.join(root_file_path, 'S' + str(subject), "train")
        npy_files = glob.glob(os.path.join(subject_file_path, '*.npy'))

        print(npy_files)
        #
        for npy_file in npy_files:
            # 检查文件是否存在
            if os.path.exists(npy_file):
                # 删除文件
                os.remove(npy_file)
                print(f"文件 {npy_file} 已成功删除。")
            else:
                print(f"文件 {npy_file} 不存在。")


def view_press_bci2a():
    root_file_path = r"C:\Users\Administrator\Desktop\pytorch_example\dataset\BCI_2a\S1\test\c1_r8.npz"
    data = np.load(root_file_path, allow_pickle=True)

    print(type(data))
    print(data)
    print(data['data'].shape)


if __name__ == '__main__':
    # press_bci2a()
    # view_press_bci2a()
    # remove_npy()


    data = np.load(r'C:\Users\Administrator\Desktop\pytorch_example\dataset\dataname1\S1\session1\c1_r14.npz' , allow_pickle=True)
    print(data['data'].shape)
