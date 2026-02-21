import h5py
import numpy as np
import os
import psutil
import sys

input_dir = '.'
output_file = './oof_preds_all_folds.h5'
dataset_name = 'preds'

h5_files = sorted([
    os.path.join(input_dir, f) for f in os.listdir(input_dir)
    if f.startswith('oof_preds_fold') and f.endswith('_full.h5')
])

# Узнаем итоговые размеры
n_samples = 0
n_classes = None
fold_sample_sizes = []
for f in h5_files:
    with h5py.File(f, 'r') as h5f:
        shape = h5f[dataset_name].shape
        fold_sample_sizes.append(shape[0])
        n_samples += shape[0]
        if n_classes is None:
            n_classes = shape[1]

print(f"Total rows: {n_samples}, num classes: {n_classes}")
# Примерный объем массива (в GB)
total_bytes = n_samples * n_classes * 4  # float32 (4 байта)
print(f"Общий объем данных: {total_bytes/1024/1024/1024:.2f} GB")

BATCH_SIZE = 250_000  # Можешь увеличить при большом запасе RAM

def print_ram_usage():
    mem = psutil.virtual_memory()
    print(f"RAM usage: {mem.used/1024**3:.2f} GB / {mem.total/1024**3:.2f} GB")

with h5py.File(output_file, 'w') as h5out:
    dset = h5out.create_dataset(
        dataset_name,
        shape=(n_samples, n_classes),
        dtype='float32',
        chunks=(min(100_000, n_samples), n_classes),
        compression='gzip'
    )
    pos = 0
    processed_rows = 0
    for i, (f, fold_rows) in enumerate(zip(h5_files, fold_sample_sizes)):
        with h5py.File(f, 'r') as h5f:
            preds_data = h5f[dataset_name]
            n = preds_data.shape[0]
            print(f"\nFold {i+1} ({os.path.basename(f)}): {n} rows")
            for start in range(0, n, BATCH_SIZE):
                end = min(start + BATCH_SIZE, n)
                batch = preds_data[start:end]
                dset[pos:pos + end - start, :] = batch
                processed_rows += end - start
                percent = processed_rows / n_samples * 100
                print(f'  [{start}:{end}] | RAM:', end=' ')
                print_ram_usage()
                print(f'    Обработано: {processed_rows} / {n_samples} ({percent:.2f}%)')
                pos += end - start
print(f"\n✅ Done: {output_file}; обработано {n_samples} строк. Итоговый объем файла будет ~{total_bytes/1024/1024/1024:.2f} GB.")
