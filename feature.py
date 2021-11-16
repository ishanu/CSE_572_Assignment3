import pandas as pd
import numpy as np
import pickle_compat

pickle_compat.patch()
from scipy.fftpack import fft

def fourierTransform(dataList):
    ff = fft(dataList)
    size = len(dataList)
    t = 2 / 300
    amp = []
    freq = np.linspace(0, size * t, size)
    for i in ff:
        amp.append(np.abs(i))
    sortedAmplitude = sorted(amp)
    maxAmplitude = sortedAmplitude[(-2)]
    maxFrequency = freq.tolist()[amp.index(maxAmplitude)]
    return [maxAmplitude, maxFrequency]


def getFeatures(glucoseData):
    features = pd.DataFrame()
    for i in range(0, len(glucoseData)):

        dataList = glucoseData.iloc[i, :].tolist()

        feature1 = min(dataList)

        feature2 = max(dataList)

        mean = 0
        data = dataList[:13]
        for p in range(0, len(data) - 1):
            mean += np.abs(data[(p + 1)] - data[p])
        feature3 = mean / len(data)

        mean = 0
        data = dataList[:13]
        for p in range(0, len(data) - 1):
            mean += np.abs(data[(p + 1)] - data[p])
        feature4 = mean / len(data)

        rms = 0
        for p in range(0, len(dataList) - 1):
            rms += np.square(dataList[p])
        feature5 = np.sqrt(rms / len(dataList))

        rms = 0
        for p in range(0, len(dataList) - 1):
            rms += np.square(dataList[p])
        feature6 = np.sqrt(rms / len(dataList))

        feature7 = fourierTransform(dataList[:13])[0]
        feature8 = fourierTransform(dataList[:13])[1]
        feature9 = fourierTransform(dataList[13:])[0]
        feature10 = fourierTransform(dataList[13:])[1]

        features = features.append({
            'Minimum Value': feature1,
            'Maximum Value': feature2,
            'Mean of Absolute Values1': feature3,
            'Mean of Absolute Values2': feature4,
            'Root Mean Square': feature5,
            'Entropy': feature6,
            'Max FFT Amplitude1': feature7,
            'Max FFT Frequency1': feature8,
            'Max FFT Amplitude2': feature9,
            'Max FFT Frequency2': feature10},
            ignore_index=True)
    return features
