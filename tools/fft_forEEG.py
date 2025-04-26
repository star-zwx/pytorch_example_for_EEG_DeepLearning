import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

file_name = r"C:\Users\Administrator\Desktop\pytorch_example\dataset\BCI_2a\S1\seession1\c1_r14.npz"
# 生成一个示例脑电信号（这里用随机数据作为示例）
data = np.load(file_name, allow_pickle=True)

eeg_data = data['data'].T[0]  # 640个采样点
sampling_rate = 256  # 假设采样率为256Hz

# 定义频段范围
bands = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 50)
}

# 对信号进行傅里叶变换
fft_result = np.fft.fft(eeg_data)
freqs = np.fft.fftfreq(len(eeg_data), d=1/sampling_rate)

# 绘制原始信号和分解后的信号
plt.figure(figsize=(12, 10))

# 绘制原始信号
plt.subplot(6, 1, 1)
plt.plot(eeg_data, label='Original Signal')
plt.title('Original EEG Signal')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.legend()

# 绘制每个频段的信号
for i, (band, (lowcut, highcut)) in enumerate(bands.items(), start=2):
    # 初始化频段信号
    band_signal = np.zeros_like(fft_result)

    # 提取特定频段的频率成分
    for j, freq in enumerate(freqs):
        if lowcut <= abs(freq) <= highcut:
            band_signal[j] = fft_result[j]

    # 逆傅里叶变换，得到时间域信号
    filtered_signal = np.fft.ifft(band_signal).real

    # 绘制频段信号
    plt.subplot(6, 1, i)
    plt.plot(filtered_signal, label=f'{band.capitalize()} Band ({lowcut}-{highcut} Hz)')
    plt.title(f'{band.capitalize()} Band')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()

plt.tight_layout()
plt.show()
