# utils.py

import os
import torch
import config
from torchvision.utils import make_grid, save_image

def save_some_examples(
    gen,
    val_loader,
    epoch,
    folder,
    example_filenames: list[str] | None = None
):
    # Δημιουργούμε το φάκελο, αν δεν υπάρχει
    os.makedirs(folder, exist_ok=True)

    # Παίρνουμε τα x,y είτε από filenames είτε τυχαία
    if example_filenames is None:
        x, y = next(iter(val_loader))
    else:
        ds = val_loader.dataset
        xs, ys = [], []
        for fname in example_filenames:
            if fname not in ds.common_files:
                raise ValueError(f"Filename {fname} not found in dataset")
            idx = ds.common_files.index(fname)
            xi, yi = ds[idx]
            xs.append(xi)
            ys.append(yi)
        x = torch.stack(xs, dim=0)
        y = torch.stack(ys, dim=0)

    # Μεταφέρουμε συσκευές και βάζουμε τον generator σε eval
    x = x.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        # Παράγουμε νέες εικόνες και τις denormalize
        y_fake = gen(x)
        y_fake = (y_fake * 0.5) + 0.5
        x_denorm = (x * 0.5) + 0.5

        # Φτιάχνουμε λίστα με τα ζεύγη (real αριστερά, fake δεξιά)
        paired = [
            torch.cat([real, fake], dim=2)
            for real, fake in zip(x_denorm, y_fake)
        ]

        # Κάνουμε grid 4 στη σειρά → 2 σειρές των 4 ζευγών
        grid = make_grid(paired, nrow=4, normalize=False, padding=2)

        # Σώζουμε ένα αρχείο με όλα τα ζεύγη
        save_image(grid, os.path.join(folder, f"pairs_{epoch}.png"))

    gen.train()

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    """
    Save model + optimizer state to disk.
    """
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    """
    Load model + optimizer state from disk and reset learning rate.
    """
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
