import torch

class FeatureTransform():
    def __init__(self, n_fft, hop_len):
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.window = torch.hann_window(n_fft)

    def __call__(self, x):
        spectrum = torch.stft(x, self.n_fft, hop_length=self.hop_len, return_complex=True, window=self.window)
        #spectrum = spectrum[:, :self.n_fft // 2] # truncate to n_fft // 2
        mag, phase = torch.abs(spectrum), torch.angle(spectrum)
        return mag, phase

class InverseFeatureTransform():
    def __init__(self, n_fft, hop_len):
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.window = torch.hann_window(n_fft)

    def __call__(self, mag, phase):
        spectrum = torch.polar(mag, phase)
        #spectrum = torch.nn.functional.pad(spectrum, (0, 0, 0, 1)) # pad back to n_fft // 2 + 1
        wav = torch.istft(spectrum, self.n_fft, hop_length=self.hop_len, window=self.window)
        return wav
