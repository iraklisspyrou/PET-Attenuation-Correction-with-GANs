import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import config
from dataset import PETDataset
from generator import Generator
from discriminator import Discriminator
from utils import save_checkpoint, load_checkpoint, save_some_examples

# Metric imports (optional: ensure torchmetrics[image] is installed)
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


def gradient_penalty(disc, x, real, fake, device):
    alpha = torch.rand(real.size(0), 1, 1, 1, device=device)
    interpolated = real + alpha * (fake - real)
    interpolated.requires_grad_(True)
    d_interpolated = disc(x, interpolated)
    # mean over patch outputs
    d_interpolated = d_interpolated.view(d_interpolated.size(0), -1).mean(1)
    grad_outputs = torch.ones_like(d_interpolated, device=device)
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    norm = gradients.norm(2, dim=1)
    gp = config.GP_WEIGHT * ((norm - 1) ** 2).mean()
    return gp


def train_fn(disc, gen, opt_disc, opt_gen, loader, device, writer, epoch):
    disc_loss_sum, gen_loss_sum, batches = 0.0, 0.0, 0
    loop = tqdm(loader, leave=True)
    for step, (x, y) in enumerate(loop):
        x, y = x.to(device), y.to(device)
        # 1) update critic
        for _ in range(config.CRITIC_ITER):
            y_fake = gen(x).detach()
            D_real = disc(x, y).view(y.size(0), -1).mean(1)
            D_fake = disc(x, y_fake).view(y.size(0), -1).mean(1)
            gp = gradient_penalty(disc, x, y, y_fake, device)
            loss_D = D_fake.mean() - D_real.mean() + gp
            opt_disc.zero_grad()
            loss_D.backward()
            opt_disc.step()
        # 2) update generator
        y_fake = gen(x)
        D_fake = disc(x, y_fake).view(y.size(0), -1).mean(1)
        loss_G_adv = -D_fake.mean()
        loss_L1 = nn.L1Loss()(y_fake, y) * config.L1_LAMBDA
        loss_G = loss_G_adv + loss_L1
        opt_gen.zero_grad()
        loss_G.backward()
        opt_gen.step()

        disc_loss_sum += loss_D.item()
        gen_loss_sum += loss_G.item()
        batches += 1
        loop.set_postfix({"D_loss": loss_D.item(), "G_loss": loss_G.item()})

        # TensorBoard batch logging
        global_step = epoch * len(loader) + step
        writer.add_scalar('Loss/Train_D', loss_D.item(), global_step)
        writer.add_scalar('Loss/Train_G', loss_G.item(), global_step)

    avg_D = disc_loss_sum / batches if batches else 0.0
    avg_G = gen_loss_sum / batches if batches else 0.0
    return avg_D, avg_G


def val_fn(disc, gen, loader, device, writer, epoch,
           ssim_metric, psnr_metric, example_filenames=None):
    disc_loss_sum, gen_loss_sum, batches = 0.0, 0.0, 0
    # reset metrics
    ssim_metric.reset()
    psnr_metric.reset()

    # optionally save fixed examples
    save_some_examples(gen, loader, epoch, folder="evaluation", example_filenames=example_filenames)

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        # generate fake for gradient penalty and D eval
        y_fake_gp = gen(x)
        # discriminator real/fake scores
        D_real = disc(x, y).view(y.size(0), -1).mean(1)
        D_fake_val = disc(x, y_fake_gp).view(y.size(0), -1).mean(1)
        # gradient penalty (requires grad on y_fake_gp)
        gp = gradient_penalty(disc, x, y, y_fake_gp, device)
        loss_D = D_fake_val.mean() - D_real.mean() + gp

        # compute generator loss without gradient tracking
        with torch.no_grad():
            loss_G_adv = -disc(x, y_fake_gp).view(y.size(0), -1).mean(1).mean()
            loss_L1 = nn.L1Loss()(y_fake_gp, y) * config.L1_LAMBDA
            loss_G = loss_G_adv + loss_L1

            # update image metrics
            y_norm = (y + 1) / 2
            y_fake_norm = (y_fake_gp + 1) / 2
            ssim_metric.update(y_fake_norm, y_norm)
            psnr_metric.update(y_fake_norm, y_norm)

        disc_loss_sum += loss_D.item()
        gen_loss_sum += loss_G.item()
        batches += 1

    avg_D = disc_loss_sum / batches if batches else 0.0
    avg_G = gen_loss_sum / batches if batches else 0.0
    ssim_val = ssim_metric.compute().item()
    psnr_val = psnr_metric.compute().item()

    # TensorBoard epoch logging
    writer.add_scalar('Loss/Val_D', avg_D, epoch)
    writer.add_scalar('Loss/Val_G', avg_G, epoch)
    writer.add_scalar('Metrics/SSIM', ssim_val, epoch)
    writer.add_scalar('Metrics/PSNR', psnr_val, epoch)

    return avg_D, avg_G 


def main():
    device = config.DEVICE
    # setup writer
    writer = SummaryWriter(log_dir=os.path.join('runs','WGAN_GP'))

    # models & optimizers
    disc = Discriminator().to(device)
    gen  = Generator().to(device)
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.9))
    opt_gen  = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.9))

    # metrics
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE)

    # datasets and loaders
    train_dataset = PETDataset(
        nac_dir="/content/training_NAC",
        ac_dir ="/content/training_AC",
        transform=config.both_transform
    )
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                              shuffle=True, num_workers=config.NUM_WORKERS)

    val_dataset = PETDataset(
        nac_dir="/content/testing_NAC",
        ac_dir ="/content/testing_AC",
        transform=config.both_transform
    )
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE,
                            shuffle=False, num_workers=config.NUM_WORKERS)

    # fixed filenames to track
    example_filenames = [
        'img_3.png','img_102.png','img_117.png','img_123.png',
        'img_671.png','img_700.png','img_782.png','img_999.png'
    ]

    for epoch in range(config.NUM_EPOCHS):
        train_D, train_G = train_fn(
            disc, gen, opt_disc, opt_gen,
            train_loader, device, writer, epoch
        )
        val_D, val_G = val_fn(
            disc, gen, val_loader, device, writer, epoch,
            ssim_metric, psnr_metric,
            example_filenames=example_filenames
        )
        # log epoch aggregates
        writer.add_scalars('Loss/D', {'train':train_D,'val':val_D}, epoch)
        writer.add_scalars('Loss/G', {'train':train_G,'val':val_G}, epoch)

        # save checkpoints every 5 epochs
        if config.SAVE_MODEL and (epoch + 1) % 5 == 0:
          # build filenames that include the epoch number
          gen_path  = f"gen_epoch_{epoch+1}.pth.tar"
          disc_path = f"disc_epoch_{epoch+1}.pth.tar"
          save_checkpoint(gen, opt_gen,  gen_path)
          save_checkpoint(disc, opt_disc, disc_path)

    writer.close()

if __name__ == "__main__":
    main()
