import torch
import torch.nn as nn
from utils import save_checkpoint, load_checkpoint, save_some_examples
import torch.optim as optim
import config
from dataset import PETDataset
from generator import Generator
from discriminator import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def train_fn(disc, gen, opt_disc, opt_gen, l1, bce, g_scaler, d_scaler, loader, writer, epoch):
    loop=tqdm(loader, leave=True)

    for idx, (x,y) in enumerate(loop):
        x, y=x.to(config.DEVICE), y.to(config.DEVICE)
        
        
        #Train discriminator

        with torch.cuda.amp.autocast():
            y_fake=gen(x)
            D_real=disc(x,y)
            D_fake=disc(x, y_fake.detach())
            D_real_loss=bce(D_real, torch.ones_like(D_real))
            D_fake_loss=bce(D_fake, torch.zeros_like(D_fake))
            D_loss=(D_real_loss+D_fake_loss) / 2


        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()


        #Train generator
        with torch.cuda.amp.autocast():
            D_fake=disc(x, y_fake)
            G_fake_loss=bce(D_fake, torch.ones_like(D_fake))
            L1=l1(y_fake, y)*config.L1_LAMBDA
            G_loss=G_fake_loss+L1

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        loop.set_postfix({
            "D_loss": D_loss.item(),
            "G_loss": G_loss.item()
        })

        step = (epoch-1)*len(loader) + idx
        writer.add_scalar("Loss/Discriminator", D_loss.item(), step)
        writer.add_scalar("Loss/Generator"    , G_loss.item(), step)
    


def main():

    # Setup TensorBoard writer
    writer = SummaryWriter(log_dir="runs/pet_gan")


    disc =Discriminator(in_channels=3).to(config.DEVICE)
    gen= Generator(in_channels=3).to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen= optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    BCE=nn.BCEWithLogitsLoss()
    L1_LOSS=nn.L1Loss()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE)

    train_dataset = PETDataset(
        nac_dir="/content/training_NAC",
        ac_dir ="/content/training_AC",
        transform=config.both_transform  
    )
    train_loader=DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    g_scaler= torch.cuda.amp.GradScaler()
    d_scaler= torch.cuda.amp.GradScaler()

    val_dataset = PETDataset(
        nac_dir="/content/testing_NAC",
        ac_dir ="/content/testing_AC",
        transform=config.both_transform
    )
    val_loader=DataLoader(val_dataset, batch_size=1, shuffle=False)


    for epoch in range(config.NUM_EPOCHS):
        train_fn(disc, gen, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler, train_loader, writer, epoch)
        if config.SAVE_MODEL and epoch % 5==0:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)

        save_some_examples(gen, val_loader, epoch, folder="evaluation")


if __name__=="__main__":
    main()


