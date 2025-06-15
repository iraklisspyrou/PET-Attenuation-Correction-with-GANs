# config.py
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 2e-4
BATCH_SIZE     = 16
NUM_WORKERS    = 2
IMAGE_SIZE     = 256
CHANNELS_IMG   = 3
L1_LAMBDA      = 100
NUM_EPOCHS     = 500
LOAD_MODEL     = False
SAVE_MODEL     = True
CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN  = "gen.pth.tar"

# WGAN-GP settings
CRITIC_ITER    = 5    # number of critic updates per generator update
GP_WEIGHT      = 10.0 # gradient penalty weight

both_transform = A.Compose(
    [
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),   
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5]*3, std=[0.5]*3, max_pixel_value=255.0),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"}  
)


transform_only_input = A.Compose(
    [
        A.ColorJitter(p=0.1),                        
        A.Normalize(
            mean=[0.5, 0.5, 0.5],                    
            std=[0.5, 0.5, 0.5],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ]
)
