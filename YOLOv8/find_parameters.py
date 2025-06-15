import os
import re
import uproot
import numpy as np

# Directory containing .root files
root_dir = r"C:\Users\INFOLYSiS\Desktop\MSc AI\DeepLearning\roots\NAC_single"
output_txt = "parameters.txt"

# Numeric sort key based on first integer found in filename
def numeric_key(fname):
    match = re.search(r"(\d+)", fname)
    return int(match.group(1)) if match else -1

# Compute center (mean) and maximum radius of positions
def analyze_source(x, y):
    cx = x.mean()
    cy = y.mean()
    r_max = np.sqrt(((x - cx) ** 2 + (y - cy) ** 2)).max()
    return cx, cy, r_max

# List and sort .root files in the directory
all_files = [f for f in os.listdir(root_dir) if f.lower().endswith('.root')]
sorted_files = sorted(all_files, key=numeric_key)

# Open output text file for writing parameters
with open(output_txt, 'w') as fout:
    fout.write("filename cx1 cy1 r1\n")

    # Process each ROOT file
    for fname in sorted_files:
        path = os.path.join(root_dir, fname)
        print(f"Processing {fname}…")

        # Read only the source position arrays from the tree
        with uproot.open(path) as f:
            tree = f['Coincidences']
            x1 = tree['sourcePosX1'].array(library='np')
            y1 = tree['sourcePosY1'].array(library='np')

        # Calculate center and radius
        cx1, cy1, r1 = analyze_source(x1, y1)

        # Print and write results
        print(f" → center=({cx1:.2f}, {cy1:.2f}), r={r1:.2f} cm")
        fout.write(f"{fname} {cx1:.2f} {cy1:.2f} {r1:.2f}\n")

print(f"\nAll results saved to {output_txt}")
