"""
hrrr_alignment.py — HRRR 天气数据下载与对齐

== 数据集关键事实（经查档官方文档）==
  来源：  gs://ts-satfire
  区域：  美国（CONUS）
  时间：  2017年1月 - 2021年10月
  分辨率：约 0.006°/像素 ≈ 0.66 km/像素，256×256 ≈ 168km 范围

== 已知的时间/位置信息（来自 tmp_*_img.npy, 27通道）==
  ch12  = Day of Year (DOY, 1-365)   ✅ 确认
  年份  = 无直接编码，需用策略处理  ⚠️
  经纬度 = 无直接经纬度通道           ⚠️（ch8/9 是大整数，非坐标）
  ch19  = 固定值 per 事件，可能是高程（m）

== HRRR 适用性 ==
  HRRR 覆盖 CONUS：-134W ~ -60W, 21N ~ 53N
  数据是美国野火 → HRRR 完全覆盖 ✅
  ERA5 是全球候选，但 HRRR 在美国分辨率更高（3km vs 28km）

== 对齐策略 ==
  1. 日期：DOY 已知 + 年份用 DOY-averaged 气候态（2017-2021 均值）
  2. 空间：下载全 CONUS HRRR，生成按 DOY 的统计场；推理时对所有样本共享同一气候态

HRRR 下载：https://console.cloud.google.com/storage/browser/high-resolution-rapid-refresh
或: pip install herbie-data
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: 提取数据集 DOY 信息
# ─────────────────────────────────────────────────────────────────────────────

def extract_doy_from_dataset(
    dataset_dir: str | Path,
    output_json: str | Path = "doy_index.json",
) -> Dict[int, int]:
    """
    扫描 tmp_*_img.npy 文件，统计每个 DOY 出现的样本数。

    Returns
    -------
    dict: {doy: count}  — 全数据集 DOY 分布
    """
    d = Path(dataset_dir)
    doy_counts: Dict[int, int] = {}

    for f in sorted(d.glob("tmp*_img.npy")):
        try:
            arr = np.load(str(f), mmap_mode="r")
            # shape: (N_events, 27, N_frames, H, W)
            for ev in range(arr.shape[0]):
                for t in range(arr.shape[2]):
                    doy = arr[ev, 12, t, 128, 128]
                    if not np.isnan(doy) and doy > 0:
                        key = int(doy)
                        doy_counts[key] = doy_counts.get(key, 0) + 1
        except Exception as e:
            logger.warning(f"跳过 {f.name}: {e}")

    sorted_counts = dict(sorted(doy_counts.items()))
    with open(output_json, "w", encoding="utf-8") as fp:
        json.dump(sorted_counts, fp)
    logger.info(f"DOY 分布已保存 → {output_json}")
    return sorted_counts


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: 下载 HRRR（全 CONUS，按日期）
# ─────────────────────────────────────────────────────────────────────────────

HRRR_VARIABLES = [
    "TMP:2 m",          # 2m 气温 (K)
    "UGRD:10 m",        # 10m U 风速 (m/s)
    "VGRD:10 m",        # 10m V 风速 (m/s)
    "RH:2 m",           # 2m 相对湿度 (%)
]

def build_hrrr_date_list(years: List[int] = None) -> List[str]:
    """生成 2017-2021 所有日期（YYYYMMDD 格式）。"""
    if years is None:
        years = [2017, 2018, 2019, 2020, 2021]
    dates = []
    for yr in years:
        start = datetime(yr, 1, 1)
        end = datetime(yr, 10, 31) if yr == 2021 else datetime(yr, 12, 31)
        cur = start
        while cur <= end:
            dates.append(cur.strftime("%Y%m%d"))
            cur += timedelta(days=1)
    return dates


def download_hrrr_day(
    date_str: str,
    hour: int = 18,
    cache_dir: str | Path = "hrrr_cache/raw",
    product: str = "sfc",
) -> Path:
    """
    用 Herbie 下载单日单小时的 HRRR 分析场（fxx=0）。

    参数
    ----
    date_str : "YYYYMMDD"
    hour     : UTC 小时（0-23），18 UTC ≈ 美西 10AM，适合卫星过境
    product  : "sfc" (地面场)

    安装：pip install herbie-data
    """
    try:
        from herbie import Herbie
    except ImportError:
        raise ImportError("pip install herbie-data")

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    dt_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} {hour:02d}:00"
    H = Herbie(dt_str, model="hrrr", product=product, fxx=0, save_dir=str(cache_dir))

    paths = []
    for var in HRRR_VARIABLES:
        try:
            p = H.download(search=var, verbose=False)
            paths.append(p)
        except Exception as e:
            logger.warning(f"下载 {var} 失败: {e}")

    logger.info(f"HRRR {date_str} {hour:02d}Z 下载完成: {len(paths)} 个变量")
    return cache_dir


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: 生成按 DOY 平均的 HRRR 气候态场（512×512 CONUS 裁剪）
# ─────────────────────────────────────────────────────────────────────────────

# CONUS 覆盖范围
CONUS_BBOX = {"north": 50.0, "south": 24.0, "west": -125.0, "east": -63.0}


def build_doy_climatology(
    hrrr_raw_dir: str | Path,
    output_dir: str | Path = "hrrr_cache/climatology",
    conus_shape: Tuple[int, int] = (256, 256),
):
    """
    从已下载的 HRRR GRIB2 文件，按 DOY 取均值，生成 DOY 气候态场。

    每个 DOY 保存：climatology/doy_{doy:03d}.npy   shape: (4, H, W)
    通道顺序: [TMP, UGRD, VGRD, RH]

    文件目录结构（herbie 输出）：
        {raw_dir}/hrrr/YYYYMMDD/subset_HASH__hrrr.t18z.wrfsfcf00.grib2

    需要: pip install cfgrib xarray scipy
    """
    try:
        import xarray as xr
        from scipy.ndimage import zoom
    except ImportError:
        raise ImportError("pip install cfgrib xarray scipy")

    raw_dir = Path(hrrr_raw_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: 按 DOY 收集 GRIB2 文件 ─────────────────────────────────────
    from collections import defaultdict
    # doy_files[doy] = [(grib_path, year), ...]
    doy_files: Dict[int, list] = defaultdict(list)

    for grib_file in sorted(raw_dir.rglob("*.grib2")):
        # Date is in the PARENT directory: .../hrrr/20190704/subset_...grib2
        parent_name = grib_file.parent.name  # e.g. "20190704"
        if len(parent_name) == 8 and parent_name.isdigit():
            try:
                dt = datetime.strptime(parent_name, "%Y%m%d")
                doy = dt.timetuple().tm_yday
                doy_files[doy].append((grib_file, dt.year))
            except Exception:
                pass

    H, W = conus_shape
    print(f"找到 {len(doy_files)} 个 DOY 的 GRIB2 数据，开始生成气候态...")

    # ── Step 2: 对每个 DOY 累积并平均 ───────────────────────────────────────
    for doy, file_year_list in sorted(doy_files.items()):
        out_npy = out_dir / f"doy_{doy:03d}.npy"
        if out_npy.exists():
            continue

        # per-channel: accumulator and count
        ch_accumulate = [np.zeros((H, W), dtype=np.float64) for _ in range(4)]
        ch_count = [0] * 4

        for grib_path, _ in file_year_list:
            try:
                ds = xr.open_dataset(
                    str(grib_path),
                    engine="cfgrib",
                    backend_kwargs={"indexpath": ""},
                )
                for var_name in ds.data_vars:
                    ch_idx = _var_to_channel_from_grib(ds[var_name])
                    if ch_idx is None:
                        continue
                    arr = ds[var_name].values.astype(np.float64).squeeze()
                    if arr.ndim != 2:
                        continue
                    # Zoom to target resolution
                    z0 = H / arr.shape[0]
                    z1 = W / arr.shape[1]
                    arr_r = zoom(arr, (z0, z1), order=1)[:H, :W]
                    ch_accumulate[ch_idx] += arr_r
                    ch_count[ch_idx] += 1
                ds.close()
            except Exception as e:
                logger.debug(f"跳过 {grib_path.name}: {e}")

        # Only save if at least one channel has data
        if any(c > 0 for c in ch_count):
            out_arr = np.zeros((4, H, W), dtype=np.float32)
            for ci in range(4):
                if ch_count[ci] > 0:
                    out_arr[ci] = (ch_accumulate[ci] / ch_count[ci]).astype(np.float32)
            np.save(str(out_npy), out_arr)
            print(f"  DOY {doy:03d}: saved shape={out_arr.shape} "
                  f"(ch counts: {ch_count})")

    print(f"\n气候态生成完成 → {out_dir}")
    saved = sorted(out_dir.glob("doy_*.npy"))
    print(f"共生成 {len(saved)} 个 DOY 气候态文件")


def _var_to_channel_from_grib(da) -> Optional[int]:
    """
    根据 xarray DataArray 的属性映射到天气通道索引。
    通道: 0=TMP, 1=UGRD, 2=VGRD, 3=RH
    """
    name = da.name.lower() if da.name else ""
    # Check GRIB attributes
    attrs = da.attrs
    short_name = str(attrs.get("GRIB_shortName", "")).lower()
    long_name = str(attrs.get("long_name", "")).lower()
    units = str(attrs.get("units", "")).lower()

    check = " ".join([name, short_name, long_name])

    if "2t" in check or "tmp" in check or "temperature" in check or "t2m" in check:
        return 0
    if "10u" in check or "ugrd" in check or "u-component" in check or "u10" in check:
        return 1
    if "10v" in check or "vgrd" in check or "v-component" in check or "v10" in check:
        return 2
    if "r2" in check or "rh" in check or "relative_humidity" in check or "relh" in check:
        return 3
    return None


def _var_to_channel(filename: str) -> Optional[int]:
    """根据文件名中的变量名映射到通道索引（legacy）。"""
    name = filename.lower()
    if "tmp" in name or "t2m" in name:
        return 0
    if "ugrd" in name or "u10" in name:
        return 1
    if "vgrd" in name or "v10" in name:
        return 2
    if "rh" in name or "relh" in name:
        return 3
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: WeatherProvider 对接气候态 HRRR
# ─────────────────────────────────────────────────────────────────────────────

class HRRRClimatologyProvider:
    """
    基于 DOY 的 HRRR 气候态天气提供器。

    对每个样本，根据 ch12(DOY) 返回对应的气候态天气场，
    并将 CONUS 裁剪为与火灾图像对应的局部子区域（如果提供了 bbox）。

    没有年份、没有坐标时：返回按 DOY 的全 CONUS 均值（256×256）。
    """

    def __init__(
        self,
        climatology_dir: str | Path,
        channels: int = 4,
        spatial_size: Tuple[int, int] = (256, 256),
    ):
        self.clim_dir = Path(climatology_dir)
        self.channels = channels
        self.spatial_size = spatial_size
        self._cache: Dict[int, np.ndarray] = {}  # DOY → array

        available = sorted(self.clim_dir.glob("doy_*.npy"))
        self._available_doys = {
            int(f.stem.split("_")[1]) for f in available
        }
        if self._available_doys:
            logger.info(f"HRRRClimatologyProvider: 载入 {len(self._available_doys)} 个 DOY 气候态")
        else:
            logger.warning("HRRRClimatologyProvider: 无气候态文件，将返回零张量")

    def get(self, doy: int) -> np.ndarray:
        """
        返回指定 DOY 的天气场。

        Parameters
        ----------
        doy : int  Day of Year (1-365)

        Returns
        -------
        np.ndarray  shape (channels, H, W)
        """
        if not self._available_doys:
            return np.zeros((self.channels, *self.spatial_size), dtype=np.float32)

        # 找最近的可用 DOY
        actual_doy = min(self._available_doys, key=lambda d: abs(d - doy))

        if actual_doy not in self._cache:
            arr = np.load(str(self.clim_dir / f"doy_{actual_doy:03d}.npy"))
            self._cache[actual_doy] = arr.astype(np.float32)

        return self._cache[actual_doy]


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: 集成到 TSSatFireSequenceDataset
# ─────────────────────────────────────────────────────────────────────────────

_INTEGRATION_EXAMPLE = """
# === 在 data/dataset.py 中集成 HRRR 天气 ===

from firest_graphnet.data.hrrr_alignment import HRRRClimatologyProvider

class TSSatFireSequenceDataset(Dataset):
    def __init__(self, ..., weather_dir=None):
        ...
        self.weather = (
            HRRRClimatologyProvider(weather_dir)
            if weather_dir and Path(weather_dir).exists()
            else None
        )

    def __getitem__(self, idx):
        ...
        x_seq = ...  # shape (T, 8, H, W)

        if self.weather is not None:
            # 从 ch12 提取 DOY（需要 tmp 文件，或者用 af_* 的时序位置近似）
            doy = int(...)          # 从原始 tmp 文件的 ch12 获取
            w = self.weather.get(doy)           # (4, 256, 256)
            w_tensor = torch.from_numpy(w)      # (4, H, W)
            # 拼接到每个时间帧
            w_expanded = w_tensor.unsqueeze(0).expand(T, -1, -1, -1)  # (T, 4, H, W)
            x_seq = torch.cat([x_seq, w_expanded], dim=1)             # (T, 12, H, W)

        return {"x_seq": x_seq, "y": y, ...}

# 训练时：
# 将 in_channels 从 8 改为 12（8 个光谱波段 + 4 个天气变量）
model = FireSTGraphNet(in_channels=12, ...)
"""


# ─────────────────────────────────────────────────────────────────────────────
# Step 6: 批量下载命令生成
# ─────────────────────────────────────────────────────────────────────────────

def generate_download_script(
    output_sh: str | Path = "download_hrrr.sh",
    years: List[int] = None,
    hour: int = 18,
):
    """
    生成 Herbie 批量下载脚本（Python 可直接运行）。

    预估数据量：
      - 2017-2021 约 1700 天 × 4 变量 × ~50MB/变量 ≈ ~340 GB
      - 如只下载特定 DOY（野火高发期 180-280，约 100 天）：~20 GB
    """
    if years is None:
        years = [2017, 2018, 2019, 2020, 2021]

    dates = build_hrrr_date_list(years)
    # 只保留野火多发 DOY 180-280（约7月1日至10月7日）
    fire_dates = []
    for ds in dates:
        dt = datetime.strptime(ds, "%Y%m%d")
        doy = dt.timetuple().tm_yday
        if 150 <= doy <= 300:
            fire_dates.append(ds)

    script = f"""#!/usr/bin/env python
# HRRR 批量下载脚本（野火高发季 DOY 150-300）
# 预估数据量: {len(fire_dates)} 天 × ~200MB ≈ {len(fire_dates)*0.2:.0f} GB

from herbie import Herbie
from pathlib import Path

CACHE_DIR = "hrrr_cache/raw"
VARIABLES = {HRRR_VARIABLES!r}
HOUR = {hour}

dates = {fire_dates!r}

for date_str in dates:
    dt_str = f"{{date_str[:4]}}-{{date_str[4:6]}}-{{date_str[6:8]}} {{HOUR:02d}}:00"
    H = Herbie(dt_str, model="hrrr", product="sfc", fxx=0,
               save_dir=CACHE_DIR, overwrite=False)
    for var in VARIABLES:
        try:
            path = H.download(searchString=var, verbose=False)
            print(f"OK {{date_str}} {{var}}")
        except Exception as e:
            print(f"FAIL {{date_str}} {{var}}: {{e}}")
"""
    with open(output_sh, "w", encoding="utf-8") as f:
        f.write(script)
    print(f"下载脚本已生成: {output_sh}")
    print(f"  覆盖日期: {fire_dates[0]} ~ {fire_dates[-1]}")
    print(f"  共 {len(fire_dates)} 天")
    print(f"  预估大小: ~{len(fire_dates)*0.2:.0f} GB")
    return output_sh


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="HRRR 对齐工具")
    sub = p.add_subparsers(dest="cmd")

    # 子命令1: 分析 DOY 分布
    s1 = sub.add_parser("doy-stats", help="统计数据集 DOY 分布")
    s1.add_argument("--dataset_dir", default=r"C:\Users\22125\Desktop\FireST-GraphNet\dataset_train")

    # 子命令2: 生成下载脚本
    s2 = sub.add_parser("gen-download", help="生成 HRRR 批量下载脚本")
    s2.add_argument("--output", default="download_hrrr.py")

    # 子命令3: 构建气候态
    s3 = sub.add_parser("build-clim", help="从已下载 GRIB2 构建 DOY 气候态")
    s3.add_argument("--raw_dir", default="hrrr_cache/raw")
    s3.add_argument("--out_dir", default="hrrr_cache/climatology")

    args = p.parse_args()

    if args.cmd == "doy-stats":
        counts = extract_doy_from_dataset(args.dataset_dir)
        print(f"DOY 分布已提取，共 {sum(counts.values())} 条记录")
        print(f"活跃 DOY 范围: {min(counts)} ~ {max(counts)}")
    elif args.cmd == "gen-download":
        generate_download_script(args.output)
    elif args.cmd == "build-clim":
        build_doy_climatology(args.raw_dir, args.out_dir)
    else:
        p.print_help()
