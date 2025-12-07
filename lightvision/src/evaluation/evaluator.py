import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from src.evaluation.evaluator import evaluate_model
from src.models.model_builder import build_model  # assuming you have this
from src.utils.device import get_device  # helper that returns "cuda" or "cpu"

DATA_DIR = "data/processed"          # processed EuroSAT data
MODEL_DIR = "outputs/models"         # where teacher/student/etc. are saved
PLOTS_DIR = "outputs/plots"
REPORTS_DIR = "outputs/reports"

BATCH_SIZE = 32
NUM_CLASSES = 10  # EuroSAT RGB = 10 classes


def load_dataset():
    """Loads the EuroSAT test set"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    test_path = os.path.join(DATA_DIR, "test")

    test_dataset = datasets.ImageFolder(root=test_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    class_names = test_dataset.classes
    print(f"[✓] Loaded test dataset with {len(test_dataset)} samples.")

    return test_loader, class_names


def load_model(model_path):
    """Loads a model from disk with correct architecture."""
    if not os.path.exists(model_path):
        print(f"[!] Model not found: {model_path}")
        return None

    # Infer model type from filename
    if "teacher" in model_path:
        model_type = "teacher"
    elif "student" in model_path:
        model_type = "student"
    elif "distilled" in model_path:
        model_type = "student"
    else:
        model_type = "student"

    print(f"[✓] Loading {model_type} model → {model_path}")

    model = build_model(model_type=model_type, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.to(get_device())

    return model


def evaluate_all_models():
    device = get_device()
    print(f"\n[✓] Using device: {device}")

    # 1) Load dataset
    test_loader, class_names = load_dataset()

    # 2) List all models in outputs/models
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pth")]

    if len(model_files) == 0:
        print("[!] No models found in outputs/models/")
        return

    print("\n[✓] Found models:")
    for f in model_files:
        print("   →", f)

    # 3) Evaluate each model
    for model_file in model_files:
        model_path = os.path.join(MODEL_DIR, model_file)
        model = load_model(model_path)

        if model is None:
            continue

        model_name = os.path.splitext(model_file)[0]

        print(f"\n[✓] Evaluating {model_name}...")

        metrics, y_true, y_pred, plot_paths = evaluate_model(
            model=model,
            data_loader=test_loader,
            device=device,
            classes=class_names,
            history=None,              # could load from saved histories if needed
            plots_dir=PLOTS_DIR,
            reports_dir=REPORTS_DIR,
            model_name=model_name
        )

        print(f"[✓] Done evaluating {model_name}.")
        print(f"    Accuracy: {metrics['accuracy']:.4f}")
        print(f"    F1 Score: {metrics['f1_score']:.4f}")
        print(f"    Saved results to outputs/reports and outputs/plots.\n")


if __name__ == "__main__":
    evaluate_all_models()
