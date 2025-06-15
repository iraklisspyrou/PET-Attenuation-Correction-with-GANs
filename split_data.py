import os
import shutil
import random

# Paths
data_ac_dir = r'C:\Users\INFOLYSiS\Desktop\MSc AI\DeepLearning\data_AC'
data_nac_dir = r'C:\Users\INFOLYSiS\Desktop\MSc AI\DeepLearning\data_NAC'

train_ac_dir = r'C:\Users\INFOLYSiS\Desktop\MSc AI\DeepLearning\training_AC'
train_nac_dir = r'C:\Users\INFOLYSiS\Desktop\MSc AI\DeepLearning\training_NAC'
test_ac_dir  = r'C:\Users\INFOLYSiS\Desktop\MSc AI\DeepLearning\testing_AC'
test_nac_dir = r'C:\Users\INFOLYSiS\Desktop\MSc AI\DeepLearning\testing_NAC'

# Δημιουργία φακέλων εξόδου
for folder in [train_ac_dir, train_nac_dir, test_ac_dir, test_nac_dir]:
    os.makedirs(folder, exist_ok=True)

# Λήψη των κοινών ονομάτων εικόνων
ac_files = set([f for f in os.listdir(data_ac_dir) if f.endswith('.png')])
nac_files = set([f for f in os.listdir(data_nac_dir) if f.endswith('.png')])
common_files = list(ac_files.intersection(nac_files))

# Shuffle και split
random.shuffle(common_files)
split_idx = int(len(common_files) * 0.8)
train_files = common_files[:split_idx]
test_files = common_files[split_idx:]

# Συνάρτηση αντιγραφής
def copy_pairs(file_list, src_ac, src_nac, dst_ac, dst_nac):
    for fname in file_list:
        shutil.copy(os.path.join(src_ac, fname), os.path.join(dst_ac, fname))
        shutil.copy(os.path.join(src_nac, fname), os.path.join(dst_nac, fname))

# Εκτέλεση split
copy_pairs(train_files, data_ac_dir, data_nac_dir, train_ac_dir, train_nac_dir)
copy_pairs(test_files, data_ac_dir, data_nac_dir, test_ac_dir, test_nac_dir)

print(f"✅ Έγινε split σε {len(train_files)} training και {len(test_files)} testing ζεύγη.")
