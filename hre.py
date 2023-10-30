import numpy as np
from scipy.signal import butter, lfilter, filtfilt
from scipy.fft import fft
from sklearn.neighbors import KNeighborsClassifier
import scipy.io as sio

# EEG Data Acquisition
# acquire raw EEG signal using Nervus EEG, USA with 64 channels at a sampling frequency of 256 Hz
data = sio.loadmat('eeg_signal.mat')
eeg_data = data['eeg_signal']

# Preprocessing
lowcut = 0.05
highcut = 60
nyquist = 0.5 * 256
low = lowcut / nyquist
high = highcut / nyquist
b, a = butter(4, [low, high], btype='band')
eeg_data = lfilter(b, a, eeg_data)
eeg_data = filtfilt(b, a, eeg_data)
# apply Surface Laplacian (SL) filter
eeg_data = np.gradient(np.gradient(eeg_data))


# Feature Extraction
frame_duration = 5 # seconds
frame_overlap = 0.5 # 50% overlap
frame_size = frame_duration * 256
frame_step = int(frame_size * (1 - frame_overlap))
frames = np.array([eeg_data[i:i+frame_size] for i in range(0, len(eeg_data)-frame_size, frame_step)])
frequency_bands = {'alpha': [8, 12], 'beta': [12, 30], 'gamma': [30, 70]}
features = []
for frame in frames:
    fft_frame = np.abs(fft(frame))
    for band, (low, high) in frequency_bands.items():
        band_fft = fft_frame[(low*frame_duration):(high*frame_duration)]
        band_mean = np.mean(band_fft)
        band_std = np.std(band_fft)
        features.append([band_mean, band_std])

# Emotion Classification
emotions = ['disgust', 'happy', 'surprise', 'fear', 'neutral']
knn = KNeighborsClassifier()
for k in range(2, 11):
    knn.n_neighbors = k
    knn.fit(features, emotions)
    accuracy = knn.score(features, emotions)
    print(f'k = {k}, accuracy = {accuracy}')