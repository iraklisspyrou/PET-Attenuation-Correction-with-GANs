import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class PETDataset(Dataset):
    def __init__(self, nac_dir, ac_dir, transform=None):
        
        self.nac_dir = nac_dir
        self.ac_dir  = ac_dir
        self.transform = transform

        # Βρες κοινά αρχεία (με ίδια ονόματα) στους δύο φακέλους
        nac_files = set(os.listdir(nac_dir))
        ac_files  = set(os.listdir(ac_dir))
        self.common_files = sorted(list(nac_files & ac_files))
        if not self.common_files:
            raise RuntimeError(f"No common files in {nac_dir} and {ac_dir}")

    def __len__(self):
        return len(self.common_files)

    def __getitem__(self, index):
        # Πάρε το όνομα του αρχείου
        filename = self.common_files[index]

        # Paths
        nac_path = os.path.join(self.nac_dir, filename)
        ac_path  = os.path.join(self.ac_dir,  filename)

        # Φόρτωσε με PIL
        nac_img = Image.open(nac_path).convert("RGB")  # "L" για grayscale, ή "RGB" αν είναι RGB
        ac_img  = Image.open(ac_path).convert("RGB")

        # Εφάρμοσε transforms αν υπάρχουν
        if self.transform is not None:
            nac_np = np.array(nac_img)
            ac_np  = np.array(ac_img)
            aug = self.transform(image=nac_np, image0=ac_np)
            nac_img = aug["image"]
            ac_img  = aug["image0"]

        return nac_img, ac_img


