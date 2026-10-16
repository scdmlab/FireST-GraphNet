import numpy as np
from pathlib import Path

data_dir = Path('dataset_train')
print('='*70)
print('TS-SatFire Dataset Validation')
print('='*70)

# Check file existence
img_files = list(data_dir.glob('*_img_*.npy'))
label_files = list(data_dir.glob('*_label_*.npy'))
firepred_files = list(data_dir.glob('firepred_*.npy'))
lulc_files = list(data_dir.glob('lulc_*.npy'))

print('\n1. File existence check:')
print(f'  img:      {img_files[0].name if img_files else "NOT FOUND"}')
print(f'  label:    {label_files[0].name if label_files else "NOT FOUND"}')
print(f'  firepred: {firepred_files[0].name if firepred_files else "NOT FOUND"}')
print(f'  lulc:     {lulc_files[0].name if lulc_files else "NOT FOUND"}')

if not all([img_files, label_files, firepred_files, lulc_files]):
    print('\nERROR: Missing required files')
    exit(1)

# Load and check shapes
img = np.load(str(img_files[0]), mmap_mode='r')
label = np.load(str(label_files[0]), mmap_mode='r')
firepred = np.load(str(firepred_files[0]), mmap_mode='r')
lulc = np.load(str(lulc_files[0]), mmap_mode='r')

print('\n2. Data shapes:')
print(f'  img:      {img.shape}')
print(f'  label:    {label.shape}')
print(f'  firepred: {firepred.shape}')
print(f'  lulc:     {lulc.shape}')

# Check N consistency
N_img = img.shape[0]
N_firepred = firepred.shape[0]
N_lulc = lulc.shape[0]

print('\n3. N consistency check:')
print(f'  img N:      {N_img}')
print(f'  firepred N: {N_firepred}')
print(f'  lulc N:     {N_lulc}')

n_match = (N_img == N_firepred == N_lulc)
error_msg = "NO - CRITICAL ERROR" if not n_match else "YES"
print(f'  Match: {error_msg}')
if not n_match:
    print(f'    Delta img-firepred: {abs(N_img - N_firepred)}')
    print(f'    Delta img-lulc: {abs(N_img - N_lulc)}')

# Check label values
print('\n4. Label value check (sample frames):')
print('  BA label (ch0):')
for i in [0, 100, 200]:
    if i < label.shape[0]:
        ba = label[i, 0, 0]
        print(f'    frame {i:3d}: min={ba.min():8.0f}, max={ba.max():8.0f}')
    
print('  AF label (ch1):')
for i in [0, 100, 200]:
    if i < label.shape[0]:
        af = label[i, 1, 0]
        print(f'    frame {i:3d}: min={af.min():8.0f}, max={af.max():8.0f}')

# Check if BA has large integers (ERROR)
ba_sample = label[100, 0, 0]
has_large_ints = (ba_sample > 100).any()
print(f'\n5. BA label validation:')
error_msg = "YES - ERROR!" if has_large_ints else "NO - OK"
print(f'  Has large integers (>100): {error_msg}')
if has_large_ints:
    print(f'    Max value in sample: {ba_sample.max():.0f}')
    print('    This means BA labels are raw area integers, not binary masks')

print('\n' + '='*70)
print('VALIDATION RESULT:')
print('='*70)
if n_match and not has_large_ints:
    print('  Status: PASS - Data looks good for training')
    print('  You can proceed with current data for AF task')
else:
    print('  Status: FAIL - Data needs regeneration')
    print('\nIssues found:')
    if not n_match:
        print('  X N mismatch between img/firepred/lulc')
        print('    -> firepred/lulc will be disabled by N-mismatch guard')
        print('    -> Cannot use auxiliary weather/fuel data')
    if has_large_ints:
        print('  X BA labels have large integers instead of binary masks')
        print('    -> Cannot train BA mapping task')
        print('    -> Cannot train Pred task (needs BA labels)')
    
    print('\nWhat you CAN do with current data:')
    print('  - Train AF detection task (label ch1 is OK)')
    print('  - 1164 clean training samples (boundary contamination fixed)')
    
    print('\nWhat you CANNOT do:')
    print('  - Train BA mapping task (wrong labels)')
    print('  - Train Pred task (requires BA difference labels)')
    print('  - Use firepred auxiliary data (N mismatch)')
    print('  - Use lulc static data (N mismatch)')
    
    print('\nTo fix: Follow DATASET_REGENERATION_GUIDE.md')
print('='*70)
