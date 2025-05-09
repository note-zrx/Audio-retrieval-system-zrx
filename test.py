import numpy as np
from scipy.io import wavfile

#读取
sample_rate,audio_data=wavfile.read('test_music/短时长/david.WAV')
#生成白噪声
white_noise=np.random.randn(len(audio_data),2)
#叠加
noisy_audio=audio_data+white_noise
#保存
wavfile.write('noisy_audio.wav',sample_rate,np.int16(noisy_audio))
