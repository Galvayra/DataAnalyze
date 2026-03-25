from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "parsing"
EMBEDDING_DIR = ROOT_DIR / "pickle" / "embedding"
DEFAULT_CKPT_ROOT = ROOT_DIR / "outputs" / "ckpt"


def ensure_torch_available() -> None:
    if torch is None or nn is None or DataLoader is None or TensorDataset is None:
        raise ImportError(
            "PyTorch is required for run_train.py. Install torch in the target environment first."
        )


def sanitize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_") or "default"


def get_model_basename(model_name: str) -> str:
    return Path(model_name).name or sanitize_name(model_name)


def get_output_prefix(model_name: str) -> str:
    return sanitize_name(get_model_basename(model_name))


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of records in {path}, got {type(data)!r}")
    return data


def load_labels(data_dir: Path, split_name: str) -> list[str]:
    records = load_records(data_dir / f"{split_name}.json")
    labels: list[str] = []
    for idx, row in enumerate(records):
        label = str(row.get("label", "") or "").strip()
        if label == "":
            raise ValueError(f"{split_name}.json row {idx} has an empty label.")
        labels.append(label)
    return labels


def get_embedding_cache_path(embedding_dir: Path, model_name: str, split_name: str) -> Path:
    model_key = sanitize_name(get_model_basename(model_name))
    return embedding_dir / f"{model_key}_{split_name}.pkl"


def load_embedding_array(embedding_dir: Path, model_name: str, split_name: str) -> np.ndarray:
    cache_path = get_embedding_cache_path(embedding_dir, model_name, split_name)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Embedding cache not found: {cache_path}. "
            f"Run `python indexing.py --model-name {model_name}` first."
        )

    with cache_path.open("rb") as f:
        payload = pickle.load(f)

    if not isinstance(payload, dict) or not isinstance(payload.get("embeddings"), np.ndarray):
        raise ValueError(f"Invalid embedding payload in {cache_path}")

    embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
    print(f"[Embedding] Load cache - {cache_path}")
    return embeddings


def validate_split_alignment(labels: list[str], embeddings: np.ndarray, split_name: str) -> None:
    if embeddings.ndim != 2:
        raise ValueError(f"{split_name} embeddings must be 2D, got shape={embeddings.shape}")
    if len(labels) != embeddings.shape[0]:
        raise ValueError(
            f"{split_name} split mismatch: labels={len(labels)} != embeddings={embeddings.shape[0]}"
        )


class EmbeddingMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | tuple[int, ...],
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        normalized_hidden_dims = [int(hidden_dim) for hidden_dim in hidden_dims]
        if any(hidden_dim <= 0 for hidden_dim in normalized_hidden_dims):
            raise ValueError(f"All hidden_dims must be positive, got {normalized_hidden_dims}")

        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in normalized_hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def make_dataloader(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(features.astype(np.float32)),
        torch.from_numpy(labels.astype(np.int64)),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_loader(
    model: EmbeddingMLP,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_predictions: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            predictions = torch.argmax(logits, dim=1)
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / max(total_samples, 1)
    return avg_loss, np.concatenate(all_labels), np.concatenate(all_predictions)


def train_one_epoch(
    model: EmbeddingMLP,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    epochs: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    progress = tqdm(data_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)

    for features, labels in progress:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        progress.set_postfix(loss=f"{loss.item():.4f}")

    progress.close()
    return total_loss / max(total_samples, 1)


def save_checkpoint(
    ckpt_path: Path,
    model: EmbeddingMLP,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    model_name: str,
    label_encoder: LabelEncoder,
    input_dim: int,
    hidden_dims: tuple[int, ...],
    num_classes: int,
    dropout: float,
    metrics: dict[str, float],
) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_name": model_name,
            "input_dim": input_dim,
            "hidden_dims": hidden_dims,
            "num_classes": num_classes,
            "dropout": dropout,
            "label_classes": label_encoder.classes_.tolist(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        ckpt_path,
    )


def parse_hidden_dims(value: str) -> tuple[int, ...]:
    raw_parts = [part.strip() for part in str(value).split(",")]
    parts = [part for part in raw_parts if part != ""]
    if not parts:
        raise argparse.ArgumentTypeError(
            "hidden_dims must be a comma-separated list like 256,128"
        )

    try:
        hidden_dims = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"hidden_dims must contain only integers, got {value!r}"
        ) from exc

    if any(hidden_dim <= 0 for hidden_dim in hidden_dims):
        raise argparse.ArgumentTypeError(
            f"hidden_dims must contain only positive integers, got {value!r}"
        )
    return hidden_dims


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a PyTorch MLP classifier from precomputed embedding pickle files."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing train.json and valid.json.",
    )
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        default=EMBEDDING_DIR,
        help="Directory containing precomputed embedding pickle files.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="models/KoE5",
        help="Model name/path used to resolve embedding cache filenames.",
    )
    parser.add_argument(
        "--ckpt-root",
        type=Path,
        default=DEFAULT_CKPT_ROOT,
        help="Root directory where checkpoints are stored.",
    )
    parser.add_argument(
        "--ckpt-name",
        type=str,
        default="",
        help="Checkpoint folder name under outputs/ckpt. Defaults to model basename.",
    )
    parser.add_argument(
        "--hidden-dims",
        type=parse_hidden_dims,
        default=(256, 256, 128, 128),
        help="Hidden layer sizes as a comma-separated list. Example: --hidden-dims 256,128",
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training.")
    parser.add_argument("--epochs", "--max-iter", dest="epochs", type=int, default=200, help="Number of training epochs.")
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience based on validation macro F1.",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Training device: auto, cpu, cuda, cuda:0, ...",
    )
    return parser.parse_args()


def main() -> None:
    ensure_torch_available()
    args = parse_args()

    output_prefix = get_output_prefix(args.model_name)
    ckpt_name = sanitize_name(args.ckpt_name) if args.ckpt_name else output_prefix
    ckpt_dir = args.ckpt_root / ckpt_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)

    train_labels = load_labels(args.data_dir, "train")
    valid_labels = load_labels(args.data_dir, "valid")
    train_embeddings = load_embedding_array(args.embedding_dir, args.model_name, "train")
    valid_embeddings = load_embedding_array(args.embedding_dir, args.model_name, "valid")

    validate_split_alignment(train_labels, train_embeddings, "train")
    validate_split_alignment(valid_labels, valid_embeddings, "valid")

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_labels)
    y_valid = label_encoder.transform(valid_labels)

    train_loader = make_dataloader(train_embeddings, y_train, args.batch_size, shuffle=True)
    valid_loader = make_dataloader(valid_embeddings, y_valid, args.batch_size, shuffle=False)

    input_dim = int(train_embeddings.shape[1])
    num_classes = int(len(label_encoder.classes_))
    hidden_dims = tuple(args.hidden_dims)
    if not hidden_dims:
        raise ValueError("hidden_dims must contain at least one layer size.")
    if any(hidden_dim <= 0 for hidden_dim in hidden_dims):
        raise ValueError(f"All hidden_dims must be positive, got {hidden_dims}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("Training config -")
    print(f"  model_name: {args.model_name}")
    print(f"  embedding_dim: {input_dim}")
    print(f"  num_classes: {num_classes}")
    print(f"  hidden_layer_sizes: {hidden_dims}")
    print(f"  dropout: {args.dropout}")
    print(f"  learning_rate: {args.learning_rate}")
    print(f"  weight_decay: {args.weight_decay}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  epochs: {args.epochs}")
    print(f"  patience: {args.patience}")
    print(f"  device: {device}")
    print(f"  checkpoint_dir: {ckpt_dir}")

    model = EmbeddingMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        dropout=args.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_valid_macro_f1 = -1.0
    best_valid_metrics: dict[str, float] = {}
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            epochs=args.epochs,
        )

        valid_loss, valid_true, valid_pred = evaluate_loader(
            model=model,
            data_loader=valid_loader,
            criterion=criterion,
            device=device,
        )
        valid_macro_f1 = float(f1_score(valid_true, valid_pred, average="macro"))
        valid_accuracy = float(accuracy_score(valid_true, valid_pred))
        valid_weighted_f1 = float(f1_score(valid_true, valid_pred, average="weighted"))
        current_metrics = {
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "valid_accuracy": valid_accuracy,
            "valid_macro_f1": valid_macro_f1,
            "valid_weighted_f1": valid_weighted_f1,
        }

        save_checkpoint(
            ckpt_path=ckpt_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            model_name=args.model_name,
            label_encoder=label_encoder,
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            dropout=args.dropout,
            metrics=current_metrics,
        )

        if valid_macro_f1 > best_valid_macro_f1:
            best_valid_macro_f1 = valid_macro_f1
            best_valid_metrics = current_metrics
            patience_counter = 0
            save_checkpoint(
                ckpt_path=ckpt_dir / "best_valid_macro_f1.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                model_name=args.model_name,
                label_encoder=label_encoder,
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                num_classes=num_classes,
                dropout=args.dropout,
                metrics=current_metrics,
            )
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch}/{args.epochs} - "
            f"train_loss={train_loss:.4f}, "
            f"valid_loss={valid_loss:.4f}, "
            f"valid_accuracy={valid_accuracy:.4f}, "
            f"valid_macro_f1={valid_macro_f1:.4f}, "
            f"patience={patience_counter}/{args.patience}"
        )

        if patience_counter >= args.patience:
            print(
                f"Early stopping triggered at epoch {epoch} "
                f"(no validation macro F1 improvement for {args.patience} epochs)."
            )
            break

    training_summary = {
        "model_name": args.model_name,
        "embedding_dim": input_dim,
        "num_classes": num_classes,
        "hidden_layer_sizes": list(hidden_dims),
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "patience": args.patience,
        "best_valid_during_training": best_valid_metrics,
    }
    with (ckpt_dir / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(training_summary, f, ensure_ascii=False, indent=2)

    print(f"Saved checkpoints to: {ckpt_dir}")


if __name__ == "__main__":
    main()
