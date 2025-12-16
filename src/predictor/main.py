import sys
from PyQt6.QtWidgets import QApplication
from .ui.main_window import PredictorWindow
import os
import glob
from .core.classifiers import CSPSVMClassifier

def main():
    app = QApplication(sys.argv)
    window = PredictorWindow()
    
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    possible_paths = glob.glob(os.path.join(models_dir, "*.pkl")) + glob.glob(os.path.join(models_dir, "*.joblib"))
    
    seen_names = set()
    for p in possible_paths:
        try:
             # Just a heuristic to guess it's a model we can load
             if "csp" in p.lower() or "svm" in p.lower():
                 if p in seen_names: continue
                 seen_names.add(p)
                 
                 clf = CSPSVMClassifier(p)
                 window.add_classifier_ui(clf)
        except Exception:
            pass
            
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
