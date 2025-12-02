"""
Wavelet decomposition of a 130 Hz ECG signal
===========================================

Implements the undecimated (à‑trous / SWT) quadratic–spline filter‑bank
described in Martínez et al. (IEEE TBME 51(4) 2004) for the first five
dyadic scales, adapted to an arbitrary sampling rate (here: 130 Hz).

Author: <you>
"""

import asyncio
import numpy as np
import pywt
#from scipy.stats import kurtosis


class ECGAnalyzer:
    """
    Real-time ECG analyzer with undecimated wavelet transform, multi-scale QRS detection,
    and sliding-window heart rate variability (HRV) metrics.

    Metrics (SDNN, RMSSD, pNN50) are computed beat-to-beat over trailing windows
    of 15 s, 60 s, and 120 s.
    """

    def __init__(
        self,
        fs: float = 130.0,
        levels: int = 5,
        wavelet_name: str = "bior2.2",
        *,
        back_stop: int = 110,
        front_stop: int = 150,
        window: int = 40,
        initial_wait_seconds: float = 10.0,
        min_new_seconds: float = 2.5,
    ) -> None:
        # immutable configuration
        self.fs = float(fs)
        self.levels = int(levels)
        self.wave = pywt.Wavelet(wavelet_name)
        self.back_stop = int(back_stop)
        self.front_stop = int(front_stop)
        self.window = int(window)

        # processing cadence
        self.segment_unit: int = 2 ** self.levels
        self.min_start_samples: int = int(initial_wait_seconds * self.fs)
        self.min_new_samples: int = int(min_new_seconds * self.fs)

        # signal buffers
        self.ecg_data: np.ndarray = np.empty(0, dtype=float)
        self.scale3_data: np.ndarray = np.empty(0, dtype=float)
        self.scale2_data: np.ndarray = np.empty(0, dtype=float)
        self.scale1_data: np.ndarray = np.empty(0, dtype=float)

        # bookkeeping
        self.analyzed_until: int = 0

        # running estimates
        self.hr_estimate: float | None = None
        self.rms_estimate: float | None = None
        self.kurt_estimate: float | None = None

        # detection outputs
        self.peaks, self.peaks2, self.peaks3 = [], [], []
        self.amplitudes, self.amplitudes2, self.amplitudes3 = [], [], []
        # rr intervals from scale-3 peaks
        self.intervals: np.ndarray = np.array([])

        # HRV metrics storage
        self.hrv_windows = [15.0, 60.0, 120.0]
        # per-window lists of metric values
        self.hrv_metrics = {w: {'sdnn': [], 'rmssd': [], 'pnn50': []} for w in self.hrv_windows}
        # per-window list of times (seconds) at each beat
        self.hrv_times = {w: [] for w in self.hrv_windows}

    def add_data(self, data: "np.ndarray | list[float]") -> None:
        """Append new ECG samples and trigger analysis when criteria are met."""
        self.ecg_data = np.concatenate([self.ecg_data, np.asarray(data, dtype=float)])
        self._maybe_process_data()

    def _maybe_process_data(self) -> None:
        total = len(self.ecg_data)
        unproc = total - self.analyzed_until
        # first run: wait for initial buffer
        if self.analyzed_until == 0:
            if total < self.min_start_samples:
                return
        else:
            if unproc < self.min_new_samples:
                return
        # largest multiple of segment_unit
        slice_len = (unproc // self.segment_unit) * self.segment_unit
        if slice_len == 0:
            return
        start = self.analyzed_until
        extra_back = 4 * self.segment_unit if start else 0
        end = start + slice_len
        segment = self.ecg_data[start - extra_back : end]
        self._analyze_batch(segment, offset=start, extra_back=extra_back)
        self.analyzed_until = end

    def _analyze_batch(self, batch, offset, extra_back):
        # wavelet decomposition and update scale data
        coeffs = pywt.swt(batch, self.wave, level=self.levels, trim_approx=False)
        details = [cd for _, cd in coeffs] + [batch]
        self.scale3_data = np.concatenate([self.scale3_data[:-extra_back], details[-4]])
        self.scale2_data = np.concatenate([self.scale2_data[:-extra_back], details[-3]])
        self.scale1_data = np.concatenate([self.scale1_data[:-extra_back], details[-2]])

        # detect new peaks on scale-3
        prev_peak_count = len(self.peaks)
        i = self.peaks[-1] if self.peaks else (offset - 200 if offset else 0)
        while i < len(self.scale3_data) - self.window:
            i = self._peak_finder(i, 3)
        # refine on finer scales
        for p in self.peaks[prev_peak_count:]:
            self._peak_finder(max(int(p - 20), 0), 2)
            self._peak_finder(max(int(p - 20), 0), 1)

        # update RR intervals (scale-3)
        if len(self.peaks) > 1:
            old_rr_len = len(self.intervals)
            self.intervals = np.diff(self.peaks) / self.fs
            new_rr_count = len(self.intervals) - old_rr_len
            if new_rr_count > 0:
                # compute new-beat HRV for each new RR
                peak_times = np.array(self.peaks) / self.fs
                rr_times = peak_times[1:]
                new_indices = range(old_rr_len, len(self.intervals))
                self._update_hrv(new_indices, rr_times)

    def _peak_finder(self, i, scale, thresh_a=1):
        signal = getattr(self, f"scale{scale}_data")
        suffix = {1: '3', 2: '2', 3: ''}[scale]
        peaks = getattr(self, f"peaks{suffix}")
        amps = getattr(self, f"amplitudes{suffix}")
        if len(peaks) > 4:
            samp_dis = int(np.mean(np.diff(peaks)[-4:]))
        else:
            samp_dis = int((self.fs * 60.0) / 80.0)
        if peaks:
            last = peaks[-1]
            ratio = ((i - last) - samp_dis) / samp_dis * 2.0
            time_weight = (-(ratio / np.sqrt(1 + ratio**2)) + 1)**3 + 0.5
        else:
            last, time_weight = 0, 1.6
        rest = len(signal) - i
        forward = min(rest, self.front_stop)
        backward = max(self.back_stop + (self.front_stop - rest), self.back_stop)
        seg = signal[i-backward:i+forward] if i>130 else signal[:130]
        rms_val = seg.std()
        #kurt_val = kurtosis(seg)
        self.rms_estimate = rms_val
        #self.kurt_estimate = kurt_val
        thresh = (thresh_a * rms_val) * time_weight
        snippet = signal[i:i+self.window]
        diff = np.diff(np.hstack([0, snippet]))
        direction = np.sign(diff)
        inflections = np.diff(np.hstack([direction[0], direction]))
        extrema = np.argwhere(inflections)[:,0] - 1
        amplitudes = np.diff(np.vstack([snippet[extrema[:-1]], snippet[extrema[1:]]]), axis=0)
        crosses = np.argwhere(amplitudes > thresh)
        cross_inds = crosses[:,1] if crosses.size else np.array([])
        if cross_inds.size:
            high = amplitudes[0][cross_inds]
            ind = np.argmax(high)
            peak_loc = extrema[cross_inds[ind]+1] + i
            max_amp = amplitudes.max()
            if (peak_loc-last)>= (samp_dis/2.5) or not peaks:
                peaks.append(peak_loc)
                amps.append(max_amp)
            else:
                peaks[-1], amps[-1] = peak_loc, max_amp
            return peak_loc
        elif peaks and amplitudes.size:
            max_amp = amplitudes.max()
            if (i-last)<= (samp_dis/2.5) and max_amp>amps[-1]:
                peaks[-1], amps[-1] = last, max_amp
        return i+5

    def _update_hrv(self, new_indices: range, rr_times: np.ndarray) -> None:
        """
        Compute HRV metrics (SDNN, RMSSD, pNN50) for each new RR index
        over trailing windows, and store values and times.
        """
        for idx in new_indices:
            t = rr_times[idx]
            for w in self.hrv_windows:
                # select RR indices within window ending at idx
                window_start = t - w
                sel = np.where(rr_times[:idx+1] >= window_start)[0]
                if window_start < 0:
                    sdnn = rmssd = pnn50 = np.nan
                elif sel.size < 2:
                    sdnn = rmssd = pnn50 = np.nan
                else:
                    seg = self.intervals[sel]
                    sdnn = float(np.std(seg, ddof=1))
                    diffs = np.diff(seg)
                    rmssd = float(np.sqrt(np.mean(diffs**2)))
                    pnn50 = float(np.sum(np.abs(diffs)>0.05)/diffs.size*100.0)
                self.hrv_metrics[w]['sdnn'].append(sdnn)
                self.hrv_metrics[w]['rmssd'].append(rmssd)
                self.hrv_metrics[w]['pnn50'].append(pnn50)
                self.hrv_times[w].append(t)

    def _retrieve_metric(self, metric: str, window: float) -> float | None:
        if window not in self.hrv_windows:
            raise ValueError(f"Unsupported window {window}")
        times = self.hrv_times[window]
        vals = self.hrv_metrics[window][metric]
        if not times or (times[-1] - times[0] < window):
            return None
        return vals[-1]

    def get_sdnn(self, window: float) -> float | None:
        """Return latest SDNN over trailing `window` seconds (15, 60, or 120)."""
        return self._retrieve_metric('sdnn', window)

    def get_rmssd(self, window: float) -> float | None:
        """Return latest RMSSD over trailing `window` seconds (15, 60, or 120)."""
        return self._retrieve_metric('rmssd', window)

    def get_pnn50(self, window: float) -> float | None:
        """Return latest pNN50 over trailing `window` seconds (15, 60, or 120)."""
        return self._retrieve_metric('pnn50', window)

    def band(self, k: int) -> tuple[float, float]:
        """Return the frequency band (Hz) corresponding to detail level *k*."""
        return (self.fs / 2 ** (k + 2), self.fs / 2 ** (k + 1))


class ECGAnalyzerAsync:
    """
    Async real-time ECG analyzer:
      - add_data: async, enqueues samples
      - start/stop: manage background consumer
      - get_sdnn, get_rmssd, get_pnn50: async retrieval

    HRV metrics (SDNN, RMSSD, pNN50) computed beat-to-beat over 15, 60, 120 s windows.

    Note: do *not* call `asyncio.create_task` in __init__; use start()/stop().
    """

    def __init__(
        self,
        fs: float = 130.0,
        levels: int = 5,
        wavelet_name: str = "bior2.2",
        *,
        back_stop: int = 110,
        front_stop: int = 150,
        window: int = 40,
        initial_wait_seconds: float = 10.0,
        min_new_seconds: float = 2.5,
    ) -> None:
        # config
        self.fs = float(fs)
        self.levels = int(levels)
        self.wave = pywt.Wavelet(wavelet_name)
        self.back_stop = int(back_stop)
        self.front_stop = int(front_stop)
        self.window = int(window)

        # cadence
        self.segment_unit = 2 ** self.levels
        self.min_start_samples = int(initial_wait_seconds * self.fs)
        self.min_new_samples = int(min_new_seconds * self.fs)

        # buffers & state
        self.ecg_data: np.ndarray = np.empty(0, dtype=float)
        self.scale3_data: np.ndarray = np.empty(0, dtype=float)
        self.scale2_data: np.ndarray = np.empty(0, dtype=float)
        self.scale1_data: np.ndarray = np.empty(0, dtype=float)
        self.analyzed_until = 0
        self.hr_estimate: float | None = None
        self.rms_estimate: float | None = None
        self.kurt_estimate: float | None = None

        # QRS outputs
        self.peaks, self.peaks2, self.peaks3 = [], [], []
        self.amplitudes, self.amplitudes2, self.amplitudes3 = [], [], []
        self.intervals = np.array([])

        # HRV storage
        self.hrv_windows = [15.0, 60.0, 120.0]
        self.hrv_metrics = {w: {'sdnn': [], 'rmssd': [], 'pnn50': []} for w in self.hrv_windows}
        self.hrv_times = {w: [] for w in self.hrv_windows}

        # internal
        self._data_queue: asyncio.Queue[np.ndarray] | None = None
        self._consumer_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Create internal queue and launch background data consumer."""
        if self._consumer_task is not None:
            return  # already started
        self._data_queue = asyncio.Queue()
        self._consumer_task = asyncio.create_task(self._data_consumer())

    async def stop(self) -> None:
        """Cancel background consumer and clear queue."""
        if self._consumer_task:
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass
            self._consumer_task = None
        if self._data_queue:
            self._data_queue = None

    async def add_data(self, data: np.ndarray | list[float]) -> None:
        """Async: enqueue new ECG samples for processing."""
        if self._data_queue is None:
            raise RuntimeError("ECGAnalyzer not started; call await start() first.")
        arr = np.asarray(data, dtype=float)
        await self._data_queue.put(arr)

    async def _data_consumer(self) -> None:
        """Background task: consume queue and trigger analysis."""
        assert self._data_queue is not None
        while True:
            arr = await self._data_queue.get()
            self.ecg_data = np.concatenate([self.ecg_data, arr])
            await self._maybe_process_data()
            self._data_queue.task_done()

    async def _maybe_process_data(self) -> None:

        total = len(self.ecg_data)
        unproc = total - self.analyzed_until
        print(unproc, total, self.analyzed_until)
        if self.analyzed_until == 0:
            if total < self.min_start_samples:
                return
        else:
            if unproc < self.min_new_samples:
                return
        seg_len = (unproc // self.segment_unit) * self.segment_unit
        if seg_len <= 0:
            return
        start = self.analyzed_until
        extra = 4 * self.segment_unit if start else 0
        end = start + seg_len
        segment = self.ecg_data[start - extra : end]
        await self._analyze_batch(segment, start, extra)
        self.analyzed_until = end

    async def _analyze_batch(self, batch: np.ndarray, offset: int, extra_back: int) -> None:

        print('analyze batch')
        coeffs = pywt.swt(batch, self.wave, level=self.levels, trim_approx=False)
        details = [cd for _, cd in coeffs] + [batch]
        self.scale3_data = np.concatenate([self.scale3_data[:-extra_back], details[-4]])
        self.scale2_data = np.concatenate([self.scale2_data[:-extra_back], details[-3]])
        self.scale1_data = np.concatenate([self.scale1_data[:-extra_back], details[-2]])
        prev = len(self.peaks)
        i = self.peaks[-1] if self.peaks else (offset - 200 if offset else 0)
        while i < len(self.scale3_data) - self.window:
            i = self._peak_finder(i, 3)
        for p in self.peaks[prev:]:
            self._peak_finder(max(int(p - 20), 0), 2)
            self._peak_finder(max(int(p - 20), 0), 1)
        if len(self.peaks) > 1:
            print('yes')
            old = len(self.intervals)
            self.intervals = np.diff(self.peaks) / self.fs
            new_idx = range(old, len(self.intervals))
            if new_idx:
                times = np.array(self.peaks) / self.fs
                # offload CPU-bound HRV to thread pool
                await asyncio.to_thread(self._update_hrv, new_idx, times[1:])

    def _peak_finder(self, i, scale, thresh_a=1):

        signal = getattr(self, f"scale{scale}_data")
        suffix = {1: '3', 2: '2', 3: ''}[scale]
        peaks = getattr(self, f"peaks{suffix}")
        amps = getattr(self, f"amplitudes{suffix}")
        if len(peaks) > 4:
            samp_dis = int(np.mean(np.diff(peaks)[-4:]))
        else:
            samp_dis = int((self.fs * 60.0) / 80.0)
        if peaks:
            last = peaks[-1]
            ratio = ((i - last) - samp_dis) / samp_dis * 2.0
            time_weight = (-(ratio / np.sqrt(1 + ratio**2)) + 1)**3 + 0.5
        else:
            last, time_weight = 0, 1.6
        rest = len(signal) - i
        forward = min(rest, self.front_stop)
        backward = max(self.back_stop + (self.front_stop - rest), self.back_stop)
        seg = signal[i-backward:i+forward] if i>130 else signal[:130]
        rms_val = seg.std()
        #kurt_val = kurtosis(seg)
        self.rms_estimate = rms_val
        #self.kurt_estimate = kurt_val
        thresh = (thresh_a * rms_val) * time_weight
        snippet = signal[i:i+self.window]
        diff = np.diff(np.hstack([0, snippet]))
        direction = np.sign(diff)
        inflections = np.diff(np.hstack([direction[0], direction]))
        extrema = np.argwhere(inflections)[:,0] - 1
        amplitudes = np.diff(np.vstack([snippet[extrema[:-1]], snippet[extrema[1:]]]), axis=0)
        crosses = np.argwhere(amplitudes > thresh)
        cross_inds = crosses[:,1] if crosses.size else np.array([])
        if cross_inds.size:
            high = amplitudes[0][cross_inds]
            ind = np.argmax(high)
            peak_loc = extrema[cross_inds[ind]+1] + i
            max_amp = amplitudes.max()
            if (peak_loc-last)>= (samp_dis/2.5) or not peaks:
                peaks.append(peak_loc)
                amps.append(max_amp)
            else:
                peaks[-1], amps[-1] = peak_loc, max_amp
            return peak_loc
        elif peaks and amplitudes.size:
            max_amp = amplitudes.max()
            if (i-last)<= (samp_dis/2.5) and max_amp>amps[-1]:
                peaks[-1], amps[-1] = last, max_amp
        return i+5

    def _update_hrv(self, new_indices: range, rr_times: np.ndarray) -> None:
        """
        Compute HRV metrics (SDNN, RMSSD, pNN50) for each new RR index
        over trailing windows, and store values and times.
        """
        for idx in new_indices:
            t = rr_times[idx]
            for w in self.hrv_windows:
                # select RR indices within window ending at idx
                window_start = t - w
                sel = np.where(rr_times[:idx+1] >= window_start)[0]
                if window_start < 0:
                    sdnn = rmssd = pnn50 = np.nan
                elif sel.size < 2:
                    sdnn = rmssd = pnn50 = np.nan
                else:
                    seg = self.intervals[sel]
                    sdnn = float(np.std(seg, ddof=1))
                    diffs = np.diff(seg)
                    rmssd = float(np.sqrt(np.mean(diffs**2)))
                    pnn50 = float(np.sum(np.abs(diffs)>0.05)/diffs.size*100.0)
                self.hrv_metrics[w]['sdnn'].append(sdnn)
                self.hrv_metrics[w]['rmssd'].append(rmssd)
                self.hrv_metrics[w]['pnn50'].append(pnn50)
                self.hrv_times[w].append(t)

    async def get_sdnn(self, window: float) -> float | None:
        if window not in self.hrv_windows:
            raise ValueError(f"Unsupported window {window}")
        times = self.hrv_times[window]
        vals = self.hrv_metrics[window]['sdnn']
        if not times or (times[-1] - times[0] < window):
            return None
        return vals[-1]

    async def get_rmssd(self, window: float) -> float | None:
        if window not in self.hrv_windows:
            raise ValueError(f"Unsupported window {window}")
        times = self.hrv_times[window]
        vals = self.hrv_metrics[window]['rmssd']
        if not times or (times[-1] - times[0] < window):
            return None
        return vals[-1]

    async def get_pnn50(self, window: float) -> float | None:
        if window not in self.hrv_windows:
            raise ValueError(f"Unsupported window {window}")
        times = self.hrv_times[window]
        vals = self.hrv_metrics[window]['pnn50']
        if not times or (times[-1] - times[0] < window):
            return None
        return vals[-1]

    def band(self, k: int) -> tuple[float, float]:
        return (self.fs / 2**(k+2), self.fs / 2**(k+1))


class ECGTestHarness:
    """
    Simulates real-time feeding of pre-recorded ECG data into an async ECGAnalyzer,
    and periodically retrieves HRV metrics to validate concurrency.

    Manages its own event loop via `asyncio.run()` in example usage.
    """

    def __init__(self, analyzer: ECGAnalyzer, ecg_signal: np.ndarray, chunk_seconds: float = 0.5) -> None:
        self.analyzer = analyzer
        self.signal = ecg_signal
        self.fs = analyzer.fs
        self.chunk_size = int(chunk_seconds * self.fs)

    async def run(self) -> None:
        # start analyzer consumer
        await self.analyzer.start()

        total = len(self.signal)
        idx = 0
        async def retrieve_loop():
            while True:
                await asyncio.sleep(1.0)
                for w in self.analyzer.hrv_windows:
                    sd = await self.analyzer.get_sdnn(w)
                    rm = await self.analyzer.get_rmssd(w)
                    pn = await self.analyzer.get_pnn50(w)
                    print(f"[{w:.0f}s] SDNN={sd}, RMSSD={rm}, pNN50={pn}")

        retrieve_task = asyncio.create_task(retrieve_loop())

        while idx < total:
            end = min(idx + self.chunk_size, total)
            chunk = self.signal[idx:end]
            await self.analyzer.add_data(chunk)
            await asyncio.sleep(0.25)
            idx = end

        await self.analyzer._data_queue.join()
        retrieve_task.cancel()
        await self.analyzer.stop()

# Example:
# if __name__ == '__main__':
#     import numpy as np
#     ecg = np.load('recorded_ecg.npy')
#     analyzer = ECGAnalyzer()
#     harness = ECGTestHarness(analyzer, ecg, chunk_seconds=0.5)
#     asyncio.run(harness.run())