# Checkpoints

The pretrained BioPM / 50MR checkpoint is included:

```
checkpoints/checkpoint.pt    (5.6 MB)
```

This file contains the pretrained weights for `encoder_acc` (the
movement-element transformer, trained with 50% masking rate).

It is a PyTorch state dict saved with `torch.save()` containing only
the `encoder_acc` weights (not the classifier head or gravity CNN).

The scripts in this package default to this path.  For example:

```bash
python scripts/extract_features.py \
    --checkpoint checkpoints/checkpoint.pt \
    ...
```
