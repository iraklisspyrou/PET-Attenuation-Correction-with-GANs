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
    # Ensure the output folder exists
    os.makedirs(folder, exist_ok=True)

    # Load data samples, either specific filenames or a random batch
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

    # Move inputs to device and set generator to evaluation mode
    x = x.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        # Generate corrected images and denormalize outputs
        y_fake = gen(x)
        y_fake = (y_fake * 0.5) + 0.5
        x_denorm = (x * 0.5) + 0.5

        # Pair real and fake images side by side
        paired = [torch.cat([real, fake], dim=2) for real, fake in zip(x_denorm, y_fake)]

        # Arrange into a grid: 4 pairs per row
        grid = make_grid(paired, nrow=4, normalize=False, padding=2)

        # Save the image grid to disk
        save_image(grid, os.path.join(folder, f"pairs_{epoch}.png"))

    gen.train()

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    """
    Save model and optimizer state to disk
    """
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    """
    Load model and optimizer state from disk and reset learning rate
    """
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    # Update optimizer learning rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
