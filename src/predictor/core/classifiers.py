from abc import ABC, abstractmethod
import joblib
import numpy as np
import time
import threading
from pylsl import resolve_streams, StreamInlet, local_clock, proc_clocksync

class BaseClassifier(ABC):
    def __init__(self):
        self._min_window = 1.0 # Default
        self._max_window = 5.0 # Default
    
    @property
    def min_window(self) -> float:
        """Minimum required window duration in seconds."""
        return self._min_window
        
    @property
    def max_window(self) -> float:
        """Maximum supported window duration in seconds."""
        return self._max_window
        
    @abstractmethod
    def predict_proba(self, data: np.ndarray, fs: float) -> np.ndarray:
        """
        Predict probabilities for the class set:
        [Relax, Left, Right, Both, Feet]
        
        Args:
            data: EEG data (n_channels, n_samples)
            fs: Sampling rate of the data
            
        Returns:
            np.ndarray: Probabilities array of shape (5,)
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Display name of the classifier."""
        pass

class MockClassifier(BaseClassifier):
    def __init__(self):
        super().__init__()
        self._min_window = 0.5
        self._max_window = 10.0
        
    @property
    def name(self):
        return "MockGeneric"
        
    def predict_proba(self, data: np.ndarray, fs: float) -> np.ndarray:
        # Return random probabilities normalized to sum 1
        probs = np.random.dirichlet(np.ones(5), size=1)[0]
        return probs

class CSPSVMClassifier(BaseClassifier):
    def __init__(self, model_path: str):
        super().__init__()
        self.model_path = model_path
        self._min_window = 2.0
        self._max_window = 5.0
        
        try:
            self.model = joblib.load(self.model_path)
            print(f"Loaded model: {model_path}")
        except Exception as e:
            print(f"Failed to load model {model_path}: {e}")
            raise

    @property
    def name(self):
        return "CSP+SVM"

    def predict_proba(self, data: np.ndarray, fs: float) -> np.ndarray:
        # Check magnitude of data and if it's too big, scale it down
        if np.max(np.abs(data)) > 1e-6:
            data = data * 1e-6
        
        # Prepare for prediction (1, ch, time)
        X = data[np.newaxis, :, :]
        
        try:
            probs = self.model.predict_proba(X)[0] 
            classes = self.model.classes_ # e.g. [1, 2] or [2, 3, 4, 5], where 1=Relax, 2=Left, etc.
            
            # Map to standard vector of size 5 even if the model has different number of classes
            # classes-1 because the classes are 1-based, and we want 0-based indexing
            full_probs = np.zeros(5)
            full_probs[classes-1] = probs
            return full_probs
        except Exception as e:
            print(f"Prediction error: {e}")
            return np.zeros(5)

class GroundTruthClassifier(BaseClassifier):
    def __init__(self, target_stream_name: str = "test-player"):
        super().__init__()
        self._name = "Ground Truth"
        self._target = target_stream_name
        
        # Determine annotation stream name
        # User says: original name + "-annotations"
        self.annot_stream_name = f"{self._target}-annotations"
        
        self.latest_label_idx = 0 # Default Relax
        self.running = True
        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()
        
    @property
    def name(self):
        return self._name

    def _listen_loop(self):
        target = None
        retries = 0
        max_retries = 3
        while target is None and self.running and retries < max_retries:
            print(f"GroundTruth: Looking for {self.annot_stream_name} (attempt {retries + 1}/{max_retries})...")
            streams = resolve_streams(wait_time=5.0)
            for s in streams:
                if s.name() == self.annot_stream_name:
                    target = s
                    break
            
            if not target:
                retries += 1
                if retries < max_retries:
                    print(f"GroundTruth: Could not find stream {self.annot_stream_name}, retrying...")
                else:
                    print(f"GroundTruth: Failed to find stream {self.annot_stream_name} after {max_retries} attempts. Exiting listener.")
                    return
        
        if not self.running: # If loop exited because self.running became False
            print("GroundTruth: Listener stopped before finding stream.")
            return
            
        print(f"GroundTruth: Connected to {target.name()} ({target.type()})")
        inlet = StreamInlet(target, processing_flags=proc_clocksync)
        
        while self.running:
            try:
                sample, ts = inlet.pull_sample(timeout=1.0)
                if sample:
                    # Sync check, see notes: https://mne.tools/mne-lsl/stable/generated/api/mne_lsl.player.PlayerLSL.html
                    now = local_clock()
                    delay = ts - now
                    if delay > 0:
                        time.sleep(delay)
                        
                    # We expect exactly 5 channels corresponding to '1'..'5' which map 1:1 to Relax(0)..Feet(4)
                    # sample is a list of floats, e.g. [0, 1, 0, 0, 0]
                    # We accept 1 or -1 as active
                    arr = np.abs(np.array(sample))
                    
                    # Find max. If all zero, remain the same
                    if np.max(arr) > 0.1:
                        self.latest_label_idx = np.argmax(arr)
                    
            except Exception as e:
                print(f"GroundTruth Error: {e}")
                time.sleep(1.0)

    def predict_proba(self, data: np.ndarray, fs: float) -> np.ndarray:
        # Ignore EEG data, return ground truth
        probs = np.zeros(5)
        if 0 <= self.latest_label_idx < 5:
            probs[self.latest_label_idx] = 1.0
        else:
            print(f"GroundTruth: Invalid label index: {self.latest_label_idx}")
        return probs
