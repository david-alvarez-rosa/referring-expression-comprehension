def tversky_loss(inputs, targets, smooth=1, alpha=0.5, beta=0.5):
    """Tversky loss function implementation"""

    # Flatten label and prediction tensors.
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    # True positives, false positives and false negatives.
    TP = (inputs * targets).sum()
    FP = ((1-targets) * inputs).sum()
    FN = (targets * (1-inputs)).sum()

    tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)

    return 1 - tversky
