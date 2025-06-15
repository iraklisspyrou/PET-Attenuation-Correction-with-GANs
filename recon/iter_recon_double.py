import os
import numpy as np
import uproot
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon

# Configuration
num_iters = 10  # Number of OSEM iterations
bins = 200      # Number of bins for sinogram
angles = np.linspace(0., 180., 180, endpoint=False)  # Projection angles in degrees

# Input directories containing ROOT files
root_dirs = {
    "NAC": r"C:\Users\INFOLYSiS\Desktop\MSc AI\DeepLearning\roots_testing\NAC_double",
    "AC":  r"C:\Users\INFOLYSiS\Desktop\MSc AI\DeepLearning\roots_testing\AC_double"
}

# Output directories for reconstructed images
out_base = r"C:\Users\INFOLYSiS\Desktop\MSc AI\DeepLearning\Combined"
out_dirs = {
    "NAC": os.path.join(out_base, "NAC_test"),
    "AC":  os.path.join(out_base, "AC_test")
}
for d in out_dirs.values():
    os.makedirs(d, exist_ok=True)


def osem_reconstruction(sinogram: np.ndarray, niter: int) -> np.ndarray:
    """
    Perform OSEM reconstruction on a sinogram array.
    """
    # Initialize image estimate
    img = np.ones((sinogram.shape[0], sinogram.shape[0]))
    for i in range(niter):
        # Forward projection
        proj = radon(img, theta=angles, circle=True)
        # Compute correction ratio and backproject
        ratio = sinogram / (proj + 1e-8)
        backproj = iradon(ratio, theta=angles, filter_name=None, circle=True)
        # Update image estimate
        img *= backproj
        print(f"  OSEM iter {i+1}/{niter}")
    return img

# Main processing loop
for label, root_dir in root_dirs.items():
    # List .root files in the current directory
    files = [f for f in os.listdir(root_dir) if f.lower().endswith(".root")]
    if not files:
        print(f"⚠️  No .root files found in {root_dir}")
        continue

    print(f"\n=== Reconstructions for {label} ===")
    for fname in files:
        path = os.path.join(root_dir, fname)
        print(f"Processing {fname}…")

        # 1) Load ROOT tree data
        with uproot.open(path) as f:
            tree = f["Coincidences"]
            x1 = tree["globalPosX1"].array(library="np")
            y1 = tree["globalPosY1"].array(library="np")
            x2 = tree["globalPosX2"].array(library="np")
            y2 = tree["globalPosY2"].array(library="np")

        # 2) Build sinogram from event positions
        theta = np.arctan2(y2 - y1, x2 - x1)
        d = x1 * np.sin(theta) - y1 * np.cos(theta)
        rmax = 400
        rbins = np.linspace(-rmax, rmax, bins)
        thetabins = np.linspace(-np.pi/2, np.pi/2, len(angles) + 1)
        sino, _, _ = np.histogram2d(d, theta, bins=[rbins, thetabins])

        # 3) Perform reconstruction
        reco = osem_reconstruction(sino, num_iters)

        # 4) Save reconstructed image as PNG
        base = os.path.splitext(fname)[0]
        out_png = os.path.join(out_dirs[label], f"{base}.png")
        plt.imsave(out_png, reco, cmap="gray", origin="lower")
        print(f"  → Saved {out_png}")