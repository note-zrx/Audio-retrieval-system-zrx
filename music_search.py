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


# 加载数据库
database = pickle.load(open('database.pickle', 'rb'))
dic_idx2song = pickle.load(open('song_index.pickle', 'rb'))

# 检索过程
def getscores(y, fs, database):
    # 对检索语音提取hash
    constellation_map = collect_map(y, fs)
    hashes = create_hash(constellation_map, fs, frequency_bits=10, song_id=None)
    # 获取与数据库中每首歌的hash匹配
    matches_per_song = {}
    for hash, (sample_time, _) in hashes.items():
        if hash in database:
            maching_occurences = database[hash]
            for source_time, song_index in maching_occurences:
                if song_index not in matches_per_song:
                    matches_per_song[song_index] = []
                matches_per_song[song_index].append((hash, sample_time, source_time))
    scores = {}
    # 对于匹配的hash，计算测试样本时间和数据库中样本时间的偏差
    for song_index, matches in matches_per_song.items():
        #         scores[song_index] = len(matches)
        song_scores_by_offset = {}
        # 对相同的时间偏差进行累计
        for hash, sample_time, source_time in matches:
            delta = source_time - sample_time
            if delta not in song_scores_by_offset:
                song_scores_by_offset[delta] = 0
            song_scores_by_offset[delta] += 1

        # 计算每条歌曲的最大累计偏差
        max = (0, 0)
        for offset, score in song_scores_by_offset.items():
            if score > max[1]:
                max = (offset, score)
        scores[song_index] = max
    scores = sorted(scores.items(), key=lambda x: x[1][1], reverse=True)
    return scores

def get_feature_points_count(y, fs):
    """
    获取输入音频的特征点数量。
    参数:
    y (np.ndarray): 音频信号。
    fs (int): 采样率。
    返回:
    int: 特征点的数量。
    """
    constellation_map = collect_map(y, fs)
    return len(constellation_map)

path = 'test_music/test6.WAV'
y, fs = librosa.load(path, sr=fix_rate)

# 检查音频长度
duration = len(y) / fs
if duration > 15:
    # 截取前15秒的音频
    y = y[:int(15 * fs)]

# 检索打分
scores = getscores(y, fs, database)
C = get_feature_points_count(y, fs)

# 打印结果
if len(scores) > 0 and scores[0][1][1]//C> 1:
    print("检索结果为:", os.path.split(dic_idx2song[scores[0][0]])[-1])
else:
    print("没有搜索到该音乐")