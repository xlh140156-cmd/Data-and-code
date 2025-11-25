import numpy as np
from scipy.ndimage import gaussian_filter1d

class ATSD:
    """
    Adaptive Time-Scale Decomposition (ATSD)
    Simplified version for demonstration and reproducibility.
    """

    def __init__(self, data, window=12):
        self.data = data.astype(float)
        self.N = len(data)
        self.window = window
        self.components = {}

    def _local_variation(self, seq):
        """Compute normalized local variation."""
        half = self.window // 2
        local_std = np.zeros(self.N)
        global_std = np.std(seq) + 1e-8

        for i in range(self.N):
            s = max(0, i - half)
            e = min(self.N, i + half)
            local_std[i] = np.std(seq[s:e])
        return local_std / global_std

    def _adaptive_filter(self, seq, flen=5, lr=0.01):
        """Simplified adaptive filter."""
        pad = np.pad(seq, (flen, 0), "edge")
        w = np.ones(flen) / flen
        low = np.zeros(self.N)

        var = self._local_variation(seq)

        for i in range(self.N):
            x = pad[i:i+flen][::-1]
            y = np.dot(w, x)
            target = np.mean(x)
            error = target - y
            w = w + lr * (1 + var[i]) * error * x
            w = w / (np.sum(w) + 1e-8)
            low[i] = y

        high = seq - low
        return high, low

    def _residual_adjust(self, original, recon):
        r = original - recon
        return gaussian_filter1d(r, sigma=1) * 0.5

    def decompose(self):
        """Four-level decomposition."""
        A1, R1 = self._adaptive_filter(self.data, flen=3)
        D1, R2 = self._adaptive_filter(R1, flen=12)
        D2, D3 = self._adaptive_filter(R2, flen=24)

        recon = A1 + D1 + D2 + D3
        corr = self._residual_adjust(self.data, recon)
        D3 = D3 + corr

        self.components = {
            "A1": A1,
            "D1": D1,
            "D2": D2,
            "D3": D3
        }
        return self.components


if __name__ == "__main__":
    # Example demo with synthetic data
    t = np.arange(400)
    series = (
        0.01 * t +
        0.8 * np.sin(2*np.pi*t/50) +
        2.0 * np.sin(2*np.pi*t/12) +
        np.random.normal(0, 0.3, size=400)
    )

    decomp = ATSD(series)
    comps = decomp.decompose()

    np.savez("atsd_components.npz", **comps)
    print("Saved: atsd_components.npz")
