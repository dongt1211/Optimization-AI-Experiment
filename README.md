## Set up
- Installing required Python packages  
- Creating necessary folders (`input/preprocessed_dataset`, `time_monitor`)  
- Downloading the Fashion-MNIST dataset from Kaggle  

2. **Ensure Kaggle API credentials** are available at:  
   - Linux/macOS: `~/.kaggle/kaggle.json`  
   - Windows: `%USERPROFILE%\.kaggle\kaggle.json`  

4. **Run the setup script**:
```bash
   bash ./set_up.bash
```

# Create preprocessed dataset 
```bash
   python dataset.py
```

# Training Script Documentation

This training script supports GPU acceleration, preprocessed datasets, parallel data loading, mixed precision training, and different optimizer choices.

## How to Run

```bash
python train_official.py --using_gpu True --preprocessed_dataset True --batch_size 128 --parallel_dataloader True --number_of_workers 1 --mixed_precision True --optimizer AdamW
```

## How to run for time monitor

```bash
python -m cProfile -s time .\train_official.py --using_gpu True --prerocessed_dataset True --batch_size 64 --parrallel_dataloader True --number_of_workers 1 --mixed_precision True --optimizer SGD > ./time_monitor/Mixed_precision_training_SGD.txt 
```


## Parameters

- `using_gpu` (bool): Enable GPU acceleration for training. Default: `True`
- `preprocessed_dataset` (bool): Use preprocessed dataset. Default: `True`
- `batch_size` (int): Number of samples per batch. Default: `64`
- `parallel_dataloader` (bool): Enable parallel data loading. Default: `True`
- `number_of_workers` (int): Number of worker processes for data loading. Default: `1`
- `mixed_precision` (bool): Enable mixed precision training for faster computation. Default: `True`
- `optimizer` (str): Optimizer to use for training. Default: `Adam` (Options: `AdamW`, `SGD`, `Adam`)
