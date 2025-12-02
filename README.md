# wavelet_ecg (project in development)

Wavelet transform based algorithm for ECG analysis and extraction of real time HRV metrics. 

Non-clinical grade ECG data, like from a Polar H10 belt, tend to be much more prone to noise. The noise in these ECG recordings can exhibit very high variance and be highly non-stationary. The Wavelet-Transform offers a powerful framework to decompose a signal into different temporal and spectral scales. This allows to selectively look for the characteristic ECG traces in the temporal and spectral bands where it exhibits the best Signal-to-Noise Ratio. 

At the same time the Discrete Wavelet Transform (DWT) is computationally very efficient, allowing for the continous real-time analysis even on low-end edge devices. 

This together with the continous low-power nature of ECG signals (e.g. compared to PPG signal) and its temporal resolution allows to extract and compute HRV metrics in real team. HRV estimates from PPG signals usually have to be aggregated over many measurements and long time scales to get reliable results. Real time ECG enables HRV metrics on sub minute scale.
