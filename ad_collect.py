import numpy as np
import librosa
from scipy import signal
import pickle
import os

fix_rate = 16000
win_length_seconds = 0.5
frequency_bits = 10
num_peaks = 15

# 提取局部最大特征点
def collect_map(y, fs, win_length_seconds=0.5, num_peaks=15):
    win_length = int(win_length_seconds * fs)
    hop_length = int(win_length // 2)
    S = librosa.stft(y, n_fft=win_length, win_length=win_length, hop_length=hop_length)
    S = np.abs(S)  # 获取频谱图
    D, T = np.shape(S)

    constellation_map = []
    for i in range(T):
        spectrum = S[:, i]
        peaks_index, props = signal.find_peaks(spectrum, prominence=0, distance=200)
        # 根据显著性进行排序
        n_peaks = min(num_peaks, len(peaks_index))
        largest_peaks_index = np.argpartition(props['prominences'], -n_peaks)[-n_peaks:]
        for peak_index in peaks_index[largest_peaks_index]:
            frequency = fs / win_length * peak_index
            # 保存局部最大值点的时-频信息
            constellation_map.append([i, frequency])
    return constellation_map

# 进行Hash编码
def create_hash(constellation_map, fs, frequency_bits=10, song_id=None):
    upper_frequency = fs / 2
    hashes = {}
    for idx, (time, freq) in enumerate(constellation_map):
        for other_time, other_freq in constellation_map[idx: idx + 100]:  # 从邻近的100个点中找点对
            diff = int(other_time - time)
            if diff <= 1 or diff > 10:  # 在一定时间范围内找点对
                continue
            freq_binned = int(freq / upper_frequency * (2 ** frequency_bits))
            other_freq_binned = int(other_freq / upper_frequency * (2 ** frequency_bits))
            hash = int(freq_binned) | (int(other_freq) << 10) | (int(diff) << 20)
            hashes[hash] = (time, song_id)
    return hashes

def create_ad_database(ad_files):
    ad_database = {}
    dic_idx2ad = {}
    for idx, ad_file in enumerate(ad_files):
        y, fs = librosa.load(ad_file, sr=fix_rate)
        constellation_map = collect_map(y, fs)
        hashes = create_hash(constellation_map, fs, frequency_bits, idx)
        for hash, (time, song_id) in hashes.items():
            if hash not in ad_database:
                ad_database[hash] = []
            ad_database[hash].append((time, song_id))
        dic_idx2ad[idx] = ad_file
    with open('ad_database.pickle', 'wb') as f:
        pickle.dump(ad_database, f)
    with open('ad_index.pickle', 'wb') as f:
        pickle.dump(dic_idx2ad, f)

ad_files = ['梦之蓝.mp3','旺旺碎冰冰.mp3','溜溜梅.mp3']
create_ad_database(ad_files)