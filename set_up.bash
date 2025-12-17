#!/bin/bash


# ===============================
# Step 1: Install Python dependencies
# ===============================
echo "=== Step 1: Installing required Python packages ==="
pip install --upgrade pip
pip install kaggle numpy pandas torch argparse memory_profiler--quiet

# ===============================
# Step 2: Create folder structure
# ===============================
echo "=== Step 2: Creating folder structure ==="
mkdir -p input/preprocessed_dataset
mkdir -p time_monitor
echo "Folders created:"
echo "- input/preprocessed_dataset"
echo "- time_monitor"

# ===============================
# Step 3: Download Fashion-MNIST from Kaggle
# ===============================
echo "=== Step 3: Downloading Fashion-MNIST dataset from Kaggle ==="
echo "Make sure your kaggle.json is in ~/.kaggle/kaggle.json"
mkdir -p ~/.kaggle
chmod 600 ~/.kaggle/kaggle.json  # ensure proper permissions

cd input
kaggle datasets download -d zalandoresearch/fashionmnist -p . --unzip
cd ..

echo "Dataset downloaded and unzipped into input/"

# ===============================
# Step 4: Done
# ===============================
echo "=== Setup completed successfully ==="
echo "Folder structure:"
tree -L 2  # optional, shows folder tree if tree command is installed
