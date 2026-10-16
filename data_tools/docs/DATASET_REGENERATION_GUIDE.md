# TS-SatFire 数据集重新生成指南

## 为什么需要重新生成？

当前 npy 文件存在严重问题，无法支持完整的 AF + BA + Pred 三任务训练：

1. ❌ BA 标签错误：`label[:, 0]` 存储的是大整数（面积值），不是二值 mask
2. ❌ Pred 标签缺失：没有 BA[t+1] - BA[t] 差分 mask
3. ❌ firepred/lulc N 不匹配：2634 ≠ 2254，索引错位

## 数据重新生成流程

### 1. 下载原始 GeoTIFF 数据（71 GB）

**Kaggle 数据集**：https://www.kaggle.com/datasets/z789456sx/ts-satfire

```bash
# 在 Kaggle 网站上下载，或使用 Kaggle API
pip install kaggle

# 配置 API key（从 Kaggle Account Settings 获取）
mkdir -p ~/.kaggle
# 将 kaggle.json 放入 ~/.kaggle/

# 下载数据集
kaggle datasets download -d z789456sx/ts-satfire
unzip ts-satfire.zip -d raw_geotiff/
```

**数据结构**：
```
raw_geotiff/
├── fire_event_001/
│   ├── 2017_001_af.tif        # Active Fire GeoTIFF
│   ├── 2017_001_ba.tif        # Burned Area GeoTIFF
│   ├── 2017_001_sr.tif        # Surface Reflectance (8 bands)
│   └── ...
├── fire_event_002/
└── ...
```

### 2. 克隆官方预处理代码

```bash
git clone https://github.com/zhaoyutim/TS-SatFire.git
cd TS-SatFire
```

### 3. 创建预处理环境

```bash
conda create --name ts-satfire python=3.9
conda activate ts-satfire
pip install -r requirements.txt

# 额外依赖（如果 requirements.txt 不完整）
pip install rasterio tqdm scikit-image opencv-python pyyaml
```

### 4. 生成 AF/BA 数据（img + label npy）

**关键参数说明**：
- `-mode`: train | val | test
- `-ts`: 时间序列长度（用 1 = seqtoseq 单帧）
- `-it`: 采样间隔（用 1 = 每天）
- `-uc`: af | ba（用例任务）

```bash
# 生成 AF 检测数据（train）
python dataset_gen_afba.py \
    -mode train \
    -ts 1 \
    -it 1 \
    -uc af

# 生成 BA 映射数据（train）- 会生成正确的二值 BA mask
python dataset_gen_afba.py \
    -mode train \
    -ts 1 \
    -it 1 \
    -uc ba

# 生成 validation 集
python dataset_gen_afba.py -mode val -ts 1 -it 1 -uc af
python dataset_gen_afba.py -mode val -ts 1 -it 1 -uc ba

# 可选：生成 test 集
python dataset_gen_afba.py -mode test -ts 1 -it 1 -uc af
python dataset_gen_afba.py -mode test -ts 1 -it 1 -uc ba
```

**输出文件**（检查这些命名模式）：
```
output/
├── af_train_img_seqtoseq_alll_1i_1.npy      # (N, 8, 1, 256, 256) VIIRS
├── af_train_label_seqtoseq_alll_1i_1.npy    # (N, 3, 1, 256, 256) ch0=BA, ch1=AF, ch2=thermal
├── ba_train_img_seqtoseq_alll_1i_1.npy      # 同 AF img（通用输入）
├── ba_train_label_seqtoseq_alll_1i_1.npy    # (N, 3, 1, 256, 256) 正确的 BA masks
└── ...
```

### 5. 生成 Pred 预测数据（firepred + BA difference 标签）

```bash
# 生成火灾预测数据（train）
python dataset_gen_pred.py \
    -mode train \
    -ts 1 \
    -it 1

# Validation 集
python dataset_gen_pred.py -mode val -ts 1 -it 1

# Test 集（可选）
python dataset_gen_pred.py -mode test -ts 1 -it 1
```

**输出文件**：
```
output/
├── firepred_train.npy         # (N, 19, 1, 256, 256) 气象/地形/燃料辅助数据
├── lulc_train.npy             # (N, 1, 256, 256) 土地覆盖
├── pred_train_label.npy       # (N, 1, 1, 256, 256) BA[t+1] - BA[t] 差分标签
└── ...
```

**⚠️ 确保 firepred/lulc 的 N 与 img/label 的 N 一致！**

### 6. 验证生成的数据

创建验证脚本 `verify_regenerated_data.py`：

```python
import numpy as np

# 加载所有文件
img_train = np.load('af_train_img_seqtoseq_alll_1i_1.npy', mmap_mode='r')
label_train = np.load('af_train_label_seqtoseq_alll_1i_1.npy', mmap_mode='r')
ba_label_train = np.load('ba_train_label_seqtoseq_alll_1i_1.npy', mmap_mode='r')
firepred_train = np.load('firepred_train.npy', mmap_mode='r')
lulc_train = np.load('lulc_train.npy', mmap_mode='r')

print("数据形状检查：")
print(f"  img:      {img_train.shape}")        # 应该是 (N, 8, 1, 256, 256)
print(f"  label:    {label_train.shape}")      # 应该是 (N, 3, 1, 256, 256)
print(f"  ba_label: {ba_label_train.shape}")   # 应该是 (N, 3, 1, 256, 256)
print(f"  firepred: {firepred_train.shape}")   # 应该是 (N, 19, 1, 256, 256)
print(f"  lulc:     {lulc_train.shape}")       # 应该是 (N, 1, 256, 256)

N_img = img_train.shape[0]
N_fire = firepred_train.shape[0]
print(f"\nN 一致性检查：")
print(f"  img N = {N_img}")
print(f"  firepred N = {N_fire}")
print(f"  ✅ 一致" if N_img == N_fire else f"  ❌ 不一致！差值 {abs(N_img - N_fire)}")

# 检查 BA 标签是否是二值
ba_ch0 = ba_label_train[100, 0, 0]  # 取一个样本
print(f"\nBA 标签值范围检查（ch0）：")
print(f"  min: {ba_ch0.min():.2f}, max: {ba_ch0.max():.2f}")
print(f"  unique: {np.unique(ba_ch0)}")
print(f"  ✅ 二值" if set(np.unique(ba_ch0)).issubset({0.0, 1.0}) else "  ❌ 不是二值")

# 检查 AF 标签
af_ch1 = label_train[100, 1, 0]
print(f"\nAF 标签值范围检查（ch1）：")
print(f"  min: {af_ch1.min():.2f}, max: {af_ch1.max():.2f}")
print(f"  unique: {np.unique(af_ch1)}")
print(f"  ✅ 二值" if set(np.unique(af_ch1)).issubset({0.0, 1.0}) else "  ❌ 不是二值")

print("\n✅ 如果所有检查通过，数据集重新生成成功！")
```

运行：
```bash
conda activate ts-satfire
python verify_regenerated_data.py
```

### 7. 替换现有数据

```bash
# 备份旧数据（可选）
mkdir -p old_data_backup
mv dataset_train/*.npy old_data_backup/
mv dataset_val/*.npy old_data_backup/

# 复制新生成的数据
cp output/af_train_*.npy dataset_train/
cp output/af_val_*.npy dataset_val/
cp output/ba_train_*.npy dataset_train/
cp output/ba_val_*.npy dataset_val/
cp output/firepred_*.npy dataset_train/
cp output/firepred_*.npy dataset_val/
cp output/lulc_*.npy dataset_train/
cp output/lulc_*.npy dataset_val/
```

---

## Colab 上传方案

### 方案 A：Google Drive 同步（推荐）

```python
# 在 Colab 中
from google.colab import drive
drive.mount('/content/drive')

# 假设你把数据放在 Drive 的 FireST-GraphNet/ 文件夹
!ls /content/drive/MyDrive/FireST-GraphNet/dataset_train/
```

**优点**：
- 一次上传，永久可用
- 多个 notebook 共享
- 不占用 Colab 临时存储

**缺点**：
- 上传 71GB 原始数据需要很久
- 推荐只上传最终 npy 文件（~20GB）

### 方案 B：Kaggle Dataset 直接下载

```python
# 在 Colab 中
!pip install kaggle
!mkdir -p ~/.kaggle
# 上传 kaggle.json API key
from google.colab import files
uploaded = files.upload()  # 选择 kaggle.json

!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# 下载预处理好的 npy 数据集（如果你上传到 Kaggle）
!kaggle datasets download -d your-username/ts-satfire-npy
!unzip ts-satfire-npy.zip -d dataset/
```

### 方案 C：直接在 Colab 上运行数据生成（不推荐）

Colab 免费版限制：
- ⏱️ 最多 12 小时会话
- 💾 有限磁盘空间
- 🐌 数据生成可能超时

---

## 时间估算

| 步骤 | 时间 | 备注 |
|------|------|------|
| 下载原始 GeoTIFF | 2-6 小时 | 取决于网速 |
| 运行 dataset_gen_afba.py | 1-2 小时 | CPU 密集 |
| 运行 dataset_gen_pred.py | 1-2 小时 | CPU 密集 |
| 验证数据 | 10 分钟 | - |
| 上传到 Google Drive | 1-4 小时 | 取决于网速 |
| **总计** | **5-14 小时** | 大部分是等待时间 |

---

## 快速开始建议

如果你**现在就想开始训练**（不等数据重新生成）：

### 仅训练 AF 任务（当前可用）

```yaml
# config_af_only.yaml
task:
  name: af
  
data:
  label_channel: 1  # 使用现有的 AF 标签
  # firepred/lulc 会被 N-mismatch guard 禁用（返回零值）
```

- ✅ 1164 个干净训练样本（已修复事件边界污染）
- ✅ 可以立即在 Colab 上训练
- ⚠️ 没有辅助气象数据（firepred 被禁用）
- ⚠️ 只能评估 AF 检测性能

### 完整三任务训练（需要重新生成数据）

必须完成上述所有步骤后才能：
- ✅ 训练 AF + BA + Pred 三个任务
- ✅ 使用完整的辅助数据（气象/地形/燃料）
- ✅ 正确的标签语义

---

## 联系数据集作者（备选方案）

如果重新生成数据困难，可以联系作者要预处理好的 npy：

**作者邮箱**：
- Yu Zhao: zhaoyutim@gmail.com
- Sebastian Gerard: sgerard@kth.se

**询问内容**：
> Hi, I'm using the TS-SatFire dataset for my research. I noticed that the 
> firepred_train.npy (N=2634) and af_train_img_*.npy (N=2254) have mismatched 
> sample counts. Could you share the correct generation parameters or provide 
> aligned npy files? Thank you!

---

## 下一步

**请选择：**

1. **现在重新生成数据** → 我继续准备数据验证脚本
2. **先训练 AF 任务** → 我准备 AF-only 的训练配置
3. **联系作者要数据** → 我准备邮件模板

请告诉我你的选择！
