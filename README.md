# Audio-retrieval-system-zrx
基于 Shazam 算法的示例音频检索系统，支持在本地音乐库中快速、准确地匹配并定位输入音频片段。

## 功能特性
- 支持对 WAV/MP3 音频文件进行指纹提取  
- 构建本地声纹数据库，实现 O(1) 哈希检索  
- 动态阈值与相对静态阈值结合，兼顾短片段与长片段匹配精度  
- 简易命令行接口，支持批量测试与结果导出

## 技术选型
- **语言**：Python 3.x  
- **音频处理**：`librosa`（STFT）、`scipy.signal`（峰值检测）
- **测试与可视化**：Matplotlib、Pandas

# 安装与使用
## 1.克隆仓库并进入项目目录：
git clone https://github.com/note-zrx/Audio-retrieval-system-zrx.git  
cd Audio-retrieval-system-zrx
## 2.构建声纹数据库：
python ad_collect.py        # 收集并预处理音乐库音频  
python feature_collect.py   # 提取声纹特征并生成哈希
## 3.检索音频示例：
python audio_search.py --input path/to/query.wav
## 4.批量测试与评估
python test.py 

## 测试与评估
- 准确率（Accuracy）：在数据库中片段的正确匹配比例
- 召回率（Recall）：数据库外片段能否被误判为库内片段
- 响应时间（Response Time）：从输入到输出的延迟
- 阈值分析：固定阈值、动态阈值及最终比值阈值的对比实验
测试结果表明，在特征点对数 65–70、片段时长 2–7 秒、噪声低于 25 dB 时，可在保证 ≥ 98% 准确率的同时，实现 ≤ 2 秒的实时检索。
