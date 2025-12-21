from pathlib import Path
import sys
import numpy as np
import mne
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from mne.decoding import CSP
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from src.predictor.core.classifiers import DEFAULT_TRAINING_FILES
import src.predictor.core.tools as tools

def evaluate_fold(train_files, test_file, data_dir):
    train_paths = [data_dir / f for f in train_files]
    test_path = data_dir / test_file
    
    target_events = ["relax", "left_hand", "right_hand", "both_hands", "both_feets"]
    epoch_segment = 2.0
    epoch_step = 1.0

    # Load Train Data
    # print(f"  Loading training data: {len(train_files)} files")
    epochs_train = tools.split_annotated_into_segments(train_paths, epoch_segment, epoch_step)
    epochs_train = epochs_train[target_events]
    X_train = epochs_train.get_data(copy=True)
    y_train = epochs_train.events[:, -1]

    # Load Test Data
    # print(f"  Loading test data: {test_file}")
    epochs_test = tools.split_annotated_into_segments([test_path], epoch_segment, epoch_step)
    epochs_test = epochs_test[target_events]
    X_test = epochs_test.get_data(copy=True)
    y_test = epochs_test.events[:, -1]

    # Model 1: CSP + SVM
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
    svm = OneVsRestClassifier(SVC(kernel='rbf', probability=True))
    clf_csp = make_pipeline(csp, svm)
    clf_csp.fit(X_train, y_train)
    acc_csp = clf_csp.score(X_test, y_test)

    # Model 2: TGSP
    tgsp = make_pipeline(Covariances("oas"), TangentSpace(metric="riemann"), SVC(kernel="linear", probability=True))
    tgsp.fit(X_train, y_train)
    acc_tgsp = tgsp.score(X_test, y_test)

    return acc_csp, acc_tgsp

def run_cross_validation():
    current_file = Path(__file__).resolve()
    # Path navigation: src/predictor/tests/cross_validate.py -> src/predictor -> src -> root
    # parents[2] should be 'eeg_vehicle_simulator' if file is in tests
    # Actually, project_root was calculated as parents[3] above because of src/predictor/tests/file
    # Let's align with that. eeg_vehicle_simulator root.
    
    data_dir = project_root.parent / "eeg_collector" / "data" 
    # Adjust path if needed. Based on classifiers.py: project_root = current_file.parents[4] (from core/classifiers.py)
    # Here: src/predictor/tests/cross_validate.py
    # parents[0] = tests
    # parents[1] = predictor
    # parents[2] = src
    # parents[3] = eeg_vehicle_simulator
    
    # In classifiers.py:
    # current_file = Path(__file__).resolve() (core/classifiers.py)
    # project_root = current_file.parents[4] (core -> predictor -> src -> eeg_vehicle_simulator -> mati ?) 
    # Wait, classifiers.py says: project_root = current_file.parents[4]
    # Let's check path: c:\Users\kasia\projekt_badawczy\mati\eeg_vehicle_simulator\src\predictor\core\classifiers.py
    # p[0]=core, p[1]=predictor, p[2]=src, p[3]=eeg_vehicle_simulator, p[4]=mati
    # So data_dir = mati/eeg_collector/data
    
    # In this new file: c:\Users\kasia\projekt_badawczy\mati\eeg_vehicle_simulator\src\predictor\tests\cross_validate.py
    # p[0]=tests, p[1]=predictor, p[2]=src, p[3]=eeg_vehicle_simulator, p[4]=mati
    
    project_root_mati = Path(__file__).resolve().parents[4]
    data_dir = project_root_mati / "eeg_collector" / "data"

    print(f"Data Dir: {data_dir}")
    print(f"Files to cross-validate: {len(DEFAULT_TRAINING_FILES)}")

    results_csp = []
    results_tgsp = []

    for i, test_file in enumerate(DEFAULT_TRAINING_FILES):
        train_files = [f for f in DEFAULT_TRAINING_FILES if f != test_file]
        print(f"\n[{i+1}/{len(DEFAULT_TRAINING_FILES)}] Testing dictionary: {test_file}")
        
        try:
            acc_csp, acc_tgsp = evaluate_fold(train_files, test_file, data_dir)
            results_csp.append(acc_csp)
            results_tgsp.append(acc_tgsp)
            print(f"  -> CSP+SVM Accuracy: {acc_csp:.4f}")
            print(f"  -> TGSP Accuracy:    {acc_tgsp:.4f}")
        except Exception as e:
            print(f"  -> FAILED: {e}")

    print("\n=== Cross-Validation Results (Leave-One-File-Out) ===")
    if results_csp:
        print(f"CSP+SVM Mean Accuracy: {np.mean(results_csp):.4f} +/- {np.std(results_csp):.4f}")
    if results_tgsp:
        print(f"TGSP Mean Accuracy:    {np.mean(results_tgsp):.4f} +/- {np.std(results_tgsp):.4f}")

if __name__ == "__main__":
    run_cross_validation()
