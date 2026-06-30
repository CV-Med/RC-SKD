import os
import torch
import numpy as np
from scipy import stats
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedKFold


def mean_confidence_interval(data, confidence=0.95):
    n = len(data)
    if n < 2:
        return float(data[0]) if n == 1 else 0.0, 0.0
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return float(np.mean(data)), float(h)


def get_kfold_loaders(dataset, n_splits=5, batch_size=16, shuffle=True, num_workers=2, seed=42, patient_wise=False):
    if patient_wise:
        unique_patients = list(set(dataset.patient_ids))
        np.random.seed(seed)
        np.random.shuffle(unique_patients)
        split_size = len(unique_patients) // n_splits
        folds = []
        for fold in range(n_splits):
            val_patients = set(unique_patients[fold * split_size:(fold + 1) * split_size])
            train_idx = [i for i, pid in enumerate(dataset.patient_ids) if pid not in val_patients]
            val_idx = [i for i, pid in enumerate(dataset.patient_ids) if pid in val_patients]
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            folds.append((train_loader, val_loader))
        return folds
    else:
        labels = np.array([sample[1] for sample in dataset.samples])
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = []
        for train_idx, val_idx in skf.split(np.zeros(len(labels)), labels):
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            folds.append((train_loader, val_loader))
        return folds


def cross_validate(train_fn, model_init_fn, dataset_name, n_splits=5, patient_wise=False, **train_kwargs):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    ])

    if dataset_name == 'CIFAR100':
        full_dataset = datasets.CIFAR100(root='./data', train=True, transform=transform, download=True)
        batch_size = 32
    else:
        data_root = os.path.abspath(os.path.join(os.getcwd(), "./"))
        image_path = os.path.join(data_root, "data", dataset_name)
        full_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"), transform=transform)
        batch_size = 16

    if patient_wise:
        if not hasattr(full_dataset, 'patient_ids'):
            print("[Warning] No patient_ids attribute; falling back to stratified split.")
            patient_wise = False

    folds = get_kfold_loaders(full_dataset, n_splits=n_splits, batch_size=batch_size, seed=42, patient_wise=patient_wise)
    fold_accs = []

    for fold, (train_loader, val_loader) in enumerate(folds):
        print(f"\n{'='*60}\nFold [{fold+1}/{n_splits}]\n{'='*60}")
        model = model_init_fn()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        acc = train_fn(model, train_loader, val_loader, device, **train_kwargs)
        fold_accs.append(acc)

    fold_accs = np.array(fold_accs)
    mean_acc, ci = mean_confidence_interval(fold_accs)

    print(f"\n{'='*60}")
    print(f"Cross-validation ({n_splits}-fold{' patient-wise' if patient_wise else ''}) results:")
    print(f"  Accuracies: {fold_accs}")
    print(f"  Mean ± 95% CI: {mean_acc:.4f} ± {ci:.4f}")
    print(f"  Std: {fold_accs.std():.4f}")
    return fold_accs, mean_acc, ci
