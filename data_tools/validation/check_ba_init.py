import numpy as np, warnings
warnings.simplefilter('ignore')
from firest_graphnet.data.dataset_forecast import TSSatFireForecastDataset

ds = TSSatFireForecastDataset('dataset_train', T_in=10, horizon=20, label_channel=1)

# Scan first 20 windows for any with positive ba_init pixels
print(f"Total forecast windows: {len(ds)}")
for i in range(min(20, len(ds))):
    samp = ds[i]
    ba = samp['ba_init'].numpy()
    n_pos = int((ba > 0).sum())
    last_hist = ds.windows[i][9]  # T_in=10, so last historical frame index
    if n_pos > 0:
        print(f"Window {i}: last_hist_frame={last_hist}, ba_init positive pixels={n_pos}")

# Also scan sample at window containing frame 11
for i, w in enumerate(ds.windows):
    last = w[9]
    if 10 <= last <= 15:
        samp = ds[i]
        ba = samp['ba_init'].numpy()
        uniq = sorted(set(ba.flatten().tolist()))[:5]
        print(f"Window {i}: last_hist={last}, ba_init unique vals sample: {uniq}")
        break
