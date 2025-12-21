from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))
from src.predictor.core.classifiers import CSPSVMClassifier, TGSPClassifier, CNNClassifier

if __name__ == "__main__":
    # Ustal katalog z modelami (taki sam, jak w train())
    current_file = Path(__file__).resolve()
    models_dir = current_file.parent.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Ścieżki do plików, które będą zapisane
    csp_model_path = models_dir / "csp_svm.joblib"
    tgsp_model_path = models_dir / "tgsp.joblib"
    cnn_model_path = models_dir / "cnn.keras"

    """
    # 1) Trening CSP+SVM
    print("=== Trening CSP+SVM ===")
    csp_clf = CSPSVMClassifier(model_path=str(csp_model_path)) 
    csp_clf.train()
    print(f"Zapisano model CSP+SVM do: {csp_model_path}")

    # 2) Trening TGSP
    print("=== Trening TGSP ===")
    tgsp_clf = TGSPClassifier(model_path=str(tgsp_model_path))
    tgsp_clf.train()
    print(f"Zapisano model TGSP do: {tgsp_model_path}")
    """
    # 3) Trening CNN
    print("=== Trening CNN ===")
    cnn_clf = CNNClassifier(model_path=str(cnn_model_path))
    cnn_clf.train()
    print(f"Zapisano model CNN do: {cnn_model_path}")