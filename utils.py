def count_parameters(model, human_readable=True):
    """
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    trainable_percent = (trainable / total * 100) if total > 0 else 0.0

    def to_human(x):
        if x < 1e3:
            return f"{x}"
        elif x < 1e6:
            return f"{x / 1e3:.2f}K"
        elif x < 1e9:
            return f"{x / 1e6:.2f}M"
        else:
            return f"{x / 1e9:.2f}B"

    if human_readable:
        total_out = to_human(total)
        trainable_out = to_human(trainable)
        frozen_out = to_human(frozen)
    else:
        total_out = total
        trainable_out = trainable
        frozen_out = frozen

    return {
        "total": total_out,
        "trainable": trainable_out,
        "frozen": frozen_out,
        "trainable_percent": round(trainable_percent, 2)
    }