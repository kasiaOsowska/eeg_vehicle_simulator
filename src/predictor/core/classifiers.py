from abc import ABC, abstractmethod
import joblib
import numpy as np
import time
import threading
from pylsl import resolve_streams, StreamInlet, local_clock, proc_clocksync
import src.predictor.core.tools as tools

from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
import mne
from mne.decoding import CSP
from pathlib import Path

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder, StandardScaler

DEFAULT_TRAINING_FILES = [
    'mati_imagery_1_run1_20251207_183304_raw.fif',
    'mati_imagery_2_run1_20251207_190808_raw.fif',
    #'mati_imagery_3_real_classifier_run1_20251207_204045_raw.fif', # Corrupted
    #'mati_imagery_4_real_classifier_run1_20251207_210156_raw.fif', # Corrupted
    'mati_imagery2_run1_20251211_211514_raw.fif',
    'mati_imagery2_run2_20251211_205847_raw.fif',
    'mati_imagery3_run1_20251217_204245_raw.fif',
    #'mati_imagery3_run2_20251217_212624_raw.fif'
]

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
            print(f"Warning: Failed to load model {model_path} (it may not exist yet): {e}")
            self.model = None

    @property
    def name(self):
        return "CSP+SVM"

    def train(self):
        current_file = Path(__file__).resolve()
        project_root = current_file.parents[4]
        data_dir = project_root / "eeg_collector" / "data"
        models_dir = current_file.parents[1] / "models"
        
        training_data = DEFAULT_TRAINING_FILES
        training_data = [data_dir / data_path for data_path in training_data]
        target_events = ["relax", "left_hand", "right_hand", "both_hands", "both_feets"]

        epoch_segment = 2.0
        epoch_step = 1.0
        epochs = tools.split_annotated_into_segments(training_data, epoch_segment, epoch_step)
        epochs = epochs[target_events]

        csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
        svm = OneVsRestClassifier(SVC(kernel='rbf', probability=True))
        clf = make_pipeline(csp, svm)
        
        X = epochs.get_data(copy=True)
        y = epochs.events[:, -1]

        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        cv_scores = cross_val_score(clf, X, y, cv=cv)
        print(f"CV Accuracy: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")

        clf.fit(X, y)
        joblib.dump(clf, models_dir / "csp_svm.joblib")

    def predict_proba(self, data: np.ndarray, fs: float) -> np.ndarray:
        # Set proper magnitude of data
        if np.max(np.abs(data)) > 1e-3:
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
    def __init__(self):
        super().__init__()
        self._name = "Ground Truth"
        self.latest_label_idx = 0 # Default Relax
        self.running = False
        
    @property
    def name(self):
        return self._name

    def start(self, target_stream_name: str = "test-player"):
        if (self.running):
            print("GroundTruth: Already running, restarting...")
            self.stop()
        self.annot_stream_name = f"{target_stream_name}-annotations"
        self.running = True
        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()    
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

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

class TGSPClassifier(BaseClassifier):
    def __init__(self, model_path: str):
        super().__init__()
        self.model_path = model_path
        self._min_window = 2.0
        self._max_window = 5.0
        
        try:
            self.model = joblib.load(self.model_path)
            print(f"Loaded model: {model_path}")
        except Exception as e:
            print(f"Warning: Failed to load model {model_path} (it may not exist yet): {e}")
            self.model = None

    @property
    def name(self):
        return "TGSP"

    def train(self):
        current_file = Path(__file__).resolve()
        project_root = current_file.parents[4]
        data_dir = project_root / "eeg_collector" / "data"
        models_dir = current_file.parents[1] / "models"
        
        training_data = DEFAULT_TRAINING_FILES
        training_data = [data_dir / data_path for data_path in training_data]
        target_events = ["relax", "left_hand", "right_hand", "both_hands", "both_feets"]

        epoch_segment = 2.0
        epoch_step = 1.0
        epochs = tools.split_annotated_into_segments(training_data, epoch_segment, epoch_step)
        epochs = epochs[target_events]
        
        clf = make_pipeline(Covariances("oas"), TangentSpace(metric="riemann"), SVC(kernel="linear", probability=True))

        X = epochs.get_data(copy=True)
        y = epochs.events[:, -1]

        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        cv_scores = cross_val_score(clf, X, y, cv=cv)
        print(f"CV Accuracy: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")

        clf.fit(X, y)
        joblib.dump(clf, models_dir / "tgsp.joblib")

    def predict_proba(self, data: np.ndarray, fs: float) -> np.ndarray:        
        # Set proper magnitude of data
        if np.max(np.abs(data)) > 1e-3:
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

class CNNClassifier(BaseClassifier):
    def __init__(self, model_path: str):
        super().__init__()
        self.model_path = model_path
        self._min_window = 2.0 
        self._max_window = 5.0
        
        # Determine paths for aux files
        self.aux_path = str(Path(model_path).with_suffix('.aux.joblib'))
        
        try:
            self.model = keras.models.load_model(self.model_path)
            aux = joblib.load(self.aux_path)
            self.norm_params = aux['norm_params']
            self.le = aux['le']
            print(f"Loaded CNN model: {model_path}")
        except Exception as e:
            print(f"Warning: Failed to load CNN model/aux {model_path}: {e}")
            self.model = None
            self.norm_params = None
            self.le = None

    @property
    def name(self):
        return "CNN"

    def _compute_freq(self, epochs):
        """Extract PSD features"""
        psd = epochs.compute_psd(fmin=8., fmax=38., verbose=False)
        X = psd.get_data()  # (n_epochs, n_channels, n_freqs)
        return X

    def train(self):
        current_file = Path(__file__).resolve()
        project_root = current_file.parents[4]
        data_dir = project_root / "eeg_collector" / "data"
        models_dir = current_file.parents[1] / "models"
        
        # Split files BEFORE segmentation to prevent data leakage
        from sklearn.model_selection import train_test_split as split_files
        train_files, val_files = split_files(
            DEFAULT_TRAINING_FILES, 
            test_size=0.2, 
            random_state=42
        )
        
        print(f"Train files ({len(train_files)}): {train_files}")
        print(f"Val files ({len(val_files)}): {val_files}")
        
        target_events = ["relax", "left_hand", "right_hand", "both_hands", "both_feets"]
        epoch_segment = 2.0
        epoch_step = 1.0
        
        # Process training files
        print("Loading training epochs...")
        train_paths = [data_dir / f for f in train_files]
        epochs_train = tools.split_annotated_into_segments(train_paths, epoch_segment, epoch_step)
        epochs_train = epochs_train[target_events]
        y_train = epochs_train.events[:, -1]
        
        # Process validation files
        print("Loading validation epochs...")
        val_paths = [data_dir / f for f in val_files]
        epochs_val = tools.split_annotated_into_segments(val_paths, epoch_segment, epoch_step)
        epochs_val = epochs_val[target_events]
        y_val = epochs_val.events[:, -1]
        
        # Freq extraction
        print("Computing PSD for training data...")
        X_train = self._compute_freq(epochs_train)
        print("Computing PSD for validation data...")
        X_val = self._compute_freq(epochs_val)
        
        # Encoding
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        y_val_encoded = le.transform(y_val)
        n_classes = len(np.unique(y_train_encoded))

        # Transpose: (N, Ch, F) -> (N, F, Ch)
        X_train = np.transpose(X_train, (0, 2, 1))
        X_val = np.transpose(X_val, (0, 2, 1))
        
        # Normalization with mean/std (like user's code)
        mean = X_train.mean(axis=0, keepdims=True)  # (1, F, Ch)
        std = X_train.std(axis=0, keepdims=True) + 1e-8  # (1, F, Ch)
        
        X_train = (X_train - mean) / std
        X_val = (X_val - mean) / std
        
        # Store normalization params
        norm_params = {'mean': mean, 'std': std}

        # Build Model (original architecture from user's code)
        N_train, F, C = X_train.shape
        inputs = keras.Input(shape=(F, C))

        x = layers.Conv1D(16, kernel_size=2, padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(0.2)(x)

        x = layers.Conv1D(32, kernel_size=2, padding='same', activation='relu')(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(0.2)(x)

        x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(0.2)(x)

        x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(0.2)(x)

        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.1)(x)

        outputs = layers.Dense(n_classes, activation='softmax')(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        model.compile(
            optimizer=keras.optimizers.Adam(5e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        callbacks = [
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6, verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=50, restore_best_weights=True
            )
        ]

        print("Training CNN...")
        history = model.fit(
            X_train, y_train_encoded,
            validation_data=(X_val, y_val_encoded),
            epochs=200,
            batch_size=64,
            shuffle=True,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED - Best Model Statistics")
        print("="*60)
        
        # Get best epoch (where EarlyStopping stopped or last epoch)
        best_epoch = len(history.history['loss']) - 1
        if 'EarlyStopping' in str(callbacks):
            # EarlyStopping restores best weights, so we need to find the best epoch
            best_val_loss_epoch = np.argmin(history.history['val_loss'])
            best_epoch = best_val_loss_epoch
        
        print(f"Best Epoch: {best_epoch + 1}/{len(history.history['loss'])}")
        print(f"Training Loss:     {history.history['loss'][best_epoch]:.4f}")
        print(f"Training Accuracy: {history.history['accuracy'][best_epoch]:.4f}")
        print(f"Validation Loss:     {history.history['val_loss'][best_epoch]:.4f}")
        print(f"Validation Accuracy: {history.history['val_accuracy'][best_epoch]:.4f}")
        
        # Evaluate on both sets with restored best model
        print("\nFinal Evaluation (with best weights):")
        train_loss, train_acc = model.evaluate(X_train, y_train_encoded, verbose=0)
        val_loss, val_acc = model.evaluate(X_val, y_val_encoded, verbose=0)
        
        print(f"Final Training Accuracy:   {train_acc:.4f}")
        print(f"Final Validation Accuracy: {val_acc:.4f}")
        print("="*60 + "\n")
        
        # Save
        print(f"Saving model to {self.model_path}")
        model.save(self.model_path)
        
        print(f"Saving aux data to {self.aux_path}")
        joblib.dump({'norm_params': norm_params, 'le': le}, self.aux_path)
        
        self.model = model
        self.norm_params = norm_params
        self.le = le

    def predict_proba(self, data: np.ndarray, fs: float) -> np.ndarray:
        if self.model is None or self.norm_params is None:
            print("CNN: Model or norm_params not loaded!")
            return np.zeros(5)

        # Set proper magnitude of data
        if np.max(np.abs(data)) > 1e-3:
            data = data * 1e-6
        
        # data is (n_channels, n_samples)
        # Need to create MNE Epochs or just compute PSD directly utilizing MNE functions if possible, 
        # or manually. But compute_psd expects Epochs or Raw.
        # Since we are essentially predicting a single window, we can wrap it in an EpochsArray.
        
        try:
            # Create dummy info
            info = mne.create_info(ch_names=[str(i) for i in range(data.shape[0])], sfreq=fs, ch_types='eeg')
            # EpochsArray expects (n_epochs, n_channels, n_times)
            data_3d = data[np.newaxis, :, :]
            epochs = mne.EpochsArray(data_3d, info, verbose=False)
            
            # Use the same feature extraction as training (multi-band + wavelet)
            X = self._compute_freq(epochs)  # (1, Ch, Features)
            
            # Transpose: (N, Ch, F) -> (N, F, Ch)
            X = np.transpose(X, (0, 2, 1))  # (1, F, Ch)
            
            # Normalize using stored mean/std
            mean = self.norm_params['mean']
            std = self.norm_params['std']
            X = (X - mean) / std
            
            # Predict
            pred_encoded = self.model.predict(X, verbose=0)[0]  # (n_classes,) probabilities for encoded classes
            
            # Map back to original class labels using LabelEncoder
            # pred_encoded is probabilities for classes [0, 1, 2, 3, 4] (encoded)
            # We need to map to original labels [1, 2, 3, 4, 5]
            
            # Create full probability array for all 5 classes
            probs = np.zeros(5)
            
            # LabelEncoder.classes_ gives us the original labels in order
            # e.g., if classes_ = [1, 2, 3, 4, 5], then:
            # encoded 0 -> original 1 (Relax)
            # encoded 1 -> original 2 (Left)
            # etc.
            for encoded_idx, original_label in enumerate(self.le.classes_):
                # original_label is 1-5, we need 0-4 for array indexing
                array_idx = original_label - 1
                probs[array_idx] = pred_encoded[encoded_idx]
            
            print(f"CNN prediction: {probs}")  # Debug output
            
            return probs
            
        except Exception as e:
            print(f"CNN Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros(5)
