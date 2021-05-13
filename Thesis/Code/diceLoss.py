def dice_loss(inputs, targets, smooth=1):
    """Dice Loss function implementation.

    Inputs and targets must be presented. Smooth is auxiliary value."""

    intersection = (inputs * targets).sum()

    num = 2.*intersection + smooth
    den = inputs.sum() + targets.sum() + smooth
    dice = num/den

    return 1 - dice
