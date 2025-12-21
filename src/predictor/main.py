import sys
from PyQt6.QtWidgets import QApplication
from .ui.main_window import PredictorWindow
import os
import glob
from .core.classifiers import CSPSVMClassifier, TGSPClassifier, CNNClassifier

def main():
    app = QApplication(sys.argv)
    window = PredictorWindow()
    
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    possible_paths = (
        glob.glob(os.path.join(models_dir, "*.pkl")) + 
        glob.glob(os.path.join(models_dir, "*.joblib")) +
        glob.glob(os.path.join(models_dir, "*.keras"))
    )
    
    # Filter out auxiliary files (e.g., cnn.aux.joblib)
    possible_paths = [p for p in possible_paths if ".aux." not in p.lower()]
    
    # Store available classifiers
    available_classifiers = {}
    
    seen_names = set()
    for p in possible_paths:
        try:
            if p in seen_names: continue
            # Just a heuristic to guess it's a model we can load
            if "csp" in p.lower() or "svm" in p.lower():
                seen_names.add(p)
                clf = CSPSVMClassifier(p)
                available_classifiers['CSP+SVM'] = clf
            elif "tgsp" in p.lower():
                seen_names.add(p)
                clf = TGSPClassifier(p)
                available_classifiers['TGSP'] = clf
            elif "cnn" in p.lower():
                seen_names.add(p)
                clf = CNNClassifier(p)
                available_classifiers['CNN'] = clf
        except Exception as e:
            print(f"Error loading classifier from {p}: {e}")
            pass
    
    # Pass available classifiers to window
    window.set_available_classifiers(available_classifiers)
            
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
