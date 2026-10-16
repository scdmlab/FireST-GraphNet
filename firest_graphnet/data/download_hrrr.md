# HRRR 天气数据下载指南（针对 TS-SatFire 美国野火数据集）

## 数据集背景

TS-SatFire 数据集（`gs://ts-satfire`）覆盖美国本土（CONUS），时间 2017-2021 年。  
**HRRR**（High-Resolution Rapid Refresh）完全覆盖 CONUS，3km 分辨率，与本数据集完全匹配。

---

## 快速开始

### 安装依赖

```bash
pip install herbie-data cfgrib xarray scipy
```

### 生成下载脚本 → 执行 → 构建气候态

```bash
# Step 1: 生成脚本（仅下载 DOY 150-300 野火高发季，约 20-40 GB）
python -m firest_graphnet.data.hrrr_alignment gen-download --output download_hrrr.py

# Step 2: 执行下载（GRIB2 存入 hrrr_cache/raw/）
python download_hrrr.py

# Step 3: 构建 DOY 气候态（按 DOY 均值，每个 DOY → doy_NNN.npy (4,256,256)）
python -m firest_graphnet.data.hrrr_alignment build-clim \
    --raw_dir hrrr_cache/raw \
    --out_dir hrrr_cache/climatology
```

---

## 天气通道（4通道）

| 索引 | 变量 | 物理量 |
|------|------|------|
| 0 | TMP:2 m | 2米气温（K） |
| 1 | UGRD:10 m | U 风速（m/s） |
| 2 | VGRD:10 m | V 风速（m/s） |
| 3 | RH:2 m | 相对湿度（%） |

---

## 在训练中使用

```python
from firest_graphnet.data.weather import WeatherProvider

weather = WeatherProvider(
    climatology_dir="hrrr_cache/climatology",
    channels=4,
    spatial_size=(256, 256),
)
doy = 185   # 从 tmp_*_img.npy ch12 提取
w = weather.get_by_doy(doy)   # torch.Tensor (4, 256, 256)
```

更新 `configs/base.yaml`：
```yaml
model:
  in_channels: 12   # 8 光谱通道 + 4 天气通道
```

---

## 对齐策略说明

| 问题 | 解决方案 |
|------|----------|
| 年份未知 | DOY 气候态：2017-2021 同 DOY 均值 |
| 精确位置未知 | 全 CONUS 下载 → 缩放到 256×256 共享场 |
| 时间分辨率 | 使用 18 UTC（≈ 美西 10AM，卫星过境时段） |

---

## 1. Data Source

- **NOAA NOMADS**: <https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod/>
- **AWS Open Data**: `s3://noaa-hrrr-bdp-pds/`
- **Google Cloud**: `gs://high-resolution-rapid-refresh/`

## 2. Download via `herbie` (recommended)

```bash
pip install herbie-data
```

```python
from herbie import Herbie

H = Herbie("2023-08-15 18:00", model="hrrr", product="sfc", fxx=0)

# Download selected variables
H.download(":TMP:2 m above ground")   # 2-m temperature
H.download(":UGRD:10 m above ground")  # 10-m U-wind
H.download(":VGRD:10 m above ground")  # 10-m V-wind
H.download(":RH:2 m above ground")     # 2-m relative humidity
```

## 3. Crop to Event Bounding Box

```python
import rasterio
from rasterio.windows import from_bounds

with rasterio.open("hrrr_sfc.grib2") as src:
    # Define bounding box (west, south, east, north) in EPSG:4326
    bbox = (-120.5, 36.0, -119.0, 37.5)

    # Transform bbox to the HRRR projection (Lambert Conformal Conic)
    from rasterio.warp import transform_bounds
    bounds = transform_bounds("EPSG:4326", src.crs, *bbox)

    window = from_bounds(*bounds, transform=src.transform)
    data = src.read(1, window=window)
```

## 4. Resample to Match Fire Imagery Resolution

```python
from rasterio.warp import reproject, Resampling
import numpy as np

# Target: 256×256 at the fire event's spatial resolution
dst = np.zeros((256, 256), dtype=np.float32)

reproject(
    source=data,
    destination=dst,
    src_transform=src_transform,
    src_crs=src_crs,
    dst_transform=fire_transform,
    dst_crs=fire_crs,
    resampling=Resampling.bilinear,
)
```

## 5. Cache Locally

Save processed weather arrays in the expected directory structure:

```
hrrr_cache/
  {event_id}/
    {YYYYMMDD_HH}_lt{lead_hours}.npy    # shape: (4, 256, 256)
```

Channel order: `[temperature, u_wind, v_wind, relative_humidity]`

## 6. Wire into FireST-GraphNet

Point the `WeatherProvider` to your cache:

```python
from firest_graphnet.data.weather import WeatherProvider

wp = WeatherProvider(cache_dir="hrrr_cache", channels=4, spatial_size=(256, 256))
tensor = wp[event_id, "20230815_18", lead_time=0]  # (4, 256, 256)
```

When the cache is empty or missing, the provider automatically returns zero tensors and logs a warning.

---

## Notes

- HRRR data is available from ~2014 onwards.
- Each GRIB2 file is ~50–200 MB; crop early to save disk.
- Consider forecast lead times (fxx=0 for analysis, fxx=1..18 for forecasts).
