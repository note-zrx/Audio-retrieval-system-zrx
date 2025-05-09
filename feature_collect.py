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

# 构造歌曲名与歌曲id之间的映射字典
def song_collect(base_path):
    index = 0
    dic_idx2song = {}
    for roots, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(('.mp3', '.wav')):
                file_song = os.path.join(roots, file)
                dic_idx2song[index] = file_song
                index += 1
    return dic_idx2song

# 获取数据库中所有音乐
path_music = 'data'
current_path = os.getcwd()
path_songs = os.path.join(current_path, path_music)
dic_idx2song = song_collect(path_songs)

# 对每条音乐进行特征提取
database = {}
for song_id in dic_idx2song.keys():
    file = dic_idx2song[song_id]
    print("collect info of file", file)
    # 读取音乐
    y, fs = librosa.load(file, sr=fix_rate)
    # 提取特征对
    constellation_map = collect_map(y, fs, win_length_seconds=win_length_seconds, num_peaks=num_peaks)
    # 获取hash值
    hashes = create_hash(constellation_map, fs, frequency_bits=frequency_bits, song_id=song_id)
    # 把hash信息填充入数据库
    for hash, time_index_pair in hashes.items():
        if hash not in database:
            database[hash] = []
        database[hash].append(time_index_pair)

# 对数据进行保存
with open('database.pickle', 'wb') as db:
    pickle.dump(database, db, pickle.HIGHEST_PROTOCOL)
with open('song_index.pickle', 'wb') as songs:
    pickle.dump(dic_idx2song, songs, pickle.HIGHEST_PROTOCOL)