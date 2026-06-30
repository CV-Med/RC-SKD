import torch
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


try:
    from fvcore.nn import FlopCountAnalysis
    _FVCODE_AVAILABLE = True
except ImportError:
    _FVCODE_AVAILABLE = False


def compute_flops_reduction(student_model, teacher_model, input_shape=(1, 3, 224, 224)):
    if not _FVCODE_AVAILABLE:
        return 0.0
    device = next(student_model.parameters()).device
    dummy = torch.randn(input_shape).to(device)
    flops_t = FlopCountAnalysis(teacher_model, dummy).total()
    flops_s = FlopCountAnalysis(student_model, dummy).total()
    fr = (1 - flops_s / flops_t) * 100.0
    return float(fr)


def compute_metrics(y_true, y_pred, y_prob, num_classes=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    if num_classes is None:
        num_classes = len(np.unique(y_true))

    if num_classes == 2:
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tp = np.sum((y_true == 1) & (y_pred == 1))
        acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        fpr = fp / (fp + tn + 1e-8)
        fnr = fn / (fn + tp + 1e-8)
        try:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        except ValueError:
            auc = 0.0
    else:
        correct = np.sum(y_true == y_pred)
        acc = correct / len(y_true)
        fpr = 0.0
        fnr = 0.0
        fpr_per_class = []
        fnr_per_class = []
        for c in range(num_classes):
            y_true_bin = (y_true == c).astype(int)
            y_pred_bin = (y_pred == c).astype(int)
            tn = np.sum((y_true_bin == 0) & (y_pred_bin == 0))
            fp = np.sum((y_true_bin == 0) & (y_pred_bin == 1))
            fn = np.sum((y_true_bin == 1) & (y_pred_bin == 0))
            tp = np.sum((y_true_bin == 1) & (y_pred_bin == 1))
            fpr_per_class.append(fp / (fp + tn + 1e-8))
            fnr_per_class.append(fn / (fn + tp + 1e-8))
        fpr = float(np.mean(fpr_per_class))
        fnr = float(np.mean(fnr_per_class))
        try:
            auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
        except ValueError:
            auc = 0.0

    return {
        'accuracy': float(acc),
        'auc': float(auc),
        'fpr': float(fpr),
        'fnr': float(fnr),
    }
