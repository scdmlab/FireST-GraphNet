# TS-SatFire 数据集重新生成 - 完整操作指南

## 概览

预计总时间：**5-14 小时**（大部分是等待时间）

- **下载原始数据**: 2-6 小时（71GB）
- **数据预处理**: 4-8 小时（CPU 密集）
- **上传到 Google Drive**: 1-4 小时

---

## 第一步：准备工作空间

### 1.1 创建工作目录

```powershell
# 在本地创建工作目录（确保至少有 150GB 空闲空间）
cd C:\Users\22125\Desktop
mkdir TS-SatFire-DataGen
cd TS-SatFire-DataGen
```

### 1.2 安装 Kaggle CLI

```powershell
# 激活 conda 环境
conda activate wildfire_gnn

# 安装 Kaggle API
pip install kaggle

# 创建 .kaggle 目录
mkdir $env:USERPROFILE\.kaggle
```

### 1.3 配置 Kaggle API Key

1. 访问 https://www.kaggle.com/
2. 登录后进入 **Account Settings** → **API** → **Create New API Token**
3. 下载 `kaggle.json` 文件
4. 将文件移动到 `C:\Users\22125\.kaggle\kaggle.json`

```powershell
# 验证 API 配置
kaggle datasets list
```

---

## 第二步：下载原始 GeoTIFF 数据

### 2.1 下载数据集（预计 2-6 小时）

```powershell
# 下载 TS-SatFire 数据集（71GB）
kaggle datasets download -d z789456sx/ts-satfire -p raw_data

# 解压（这会花一些时间）
Expand-Archive -Path raw_data\ts-satfire.zip -DestinationPath raw_geotiff -Force
```

**数据结构预览**：
```
raw_geotiff/
├── fire_event_001/
│   ├── 2017_001_af.tif        # Active Fire 标注
│   ├── 2017_001_ba.tif        # Burned Area 标注
│   ├── 2017_001_sr_*.tif      # Surface Reflectance (VIIRS 8 bands)
│   ├── 2017_001_dem.tif       # DEM (高程)
│   ├── 2017_001_*.tif         # 其他辅助数据
│   └── ...
├── fire_event_002/
└── ...（共 179 个火灾事件）
```

---

## 第三步：设置预处理环境

### 3.1 克隆官方预处理代码

```powershell
git clone https://github.com/zhaoyutim/TS-SatFire.git
cd TS-SatFire
```

### 3.2 创建预处理环境

```powershell
# 创建新的 conda 环境（避免依赖冲突）
conda create --name ts-satfire-preprocess python=3.9 -y
conda activate ts-satfire-preprocess

# 安装依赖
pip install -r requirements.txt

# 如果 requirements.txt 不完整，手动安装核心依赖
pip install numpy pandas scikit-image opencv-python tqdm pyyaml
pip install rasterio geopandas fiona shapely pyproj
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 3.3 配置数据路径

编辑预处理脚本，指向你的原始数据目录：

**检查并修改配置文件**（如果有的话）：
- 查找 `dataset_gen_afba.py` 和 `dataset_gen_pred.py` 开头的配置部分
- 确保 `data_root` 或类似变量指向 `C:\Users\22125\Desktop\TS-SatFire-DataGen\raw_geotiff`

---

## 第四步：生成 AF/BA 数据（img + label）

### 4.1 生成训练集

**关键参数说明**：
- `-mode train` : 生成训练集
- `-ts 1` : 时间序列长度=1（seqtoseq 单帧格式）
- `-it 1` : 采样间隔=1天（连续采样）
- `-uc af` : 用例类型 = Active Fire 检测

```powershell
# 确保在 TS-SatFire 目录下，conda 环境已激活
cd C:\Users\22125\Desktop\TS-SatFire-DataGen\TS-SatFire
conda activate ts-satfire-preprocess

# 生成 AF 训练数据
python dataset_gen_afba.py -mode train -ts 1 -it 1 -uc af

# 生成 BA 训练数据（会生成正确的二值 BA mask）
python dataset_gen_afba.py -mode train -ts 1 -it 1 -uc ba
```

**预计时间**: 每个命令 1-2 小时

**输出文件位置**（检查脚本中的 `output_dir` 配置）：
```
output/
├── af_train_img_seqtoseq_alll_1i_1.npy      # (N, 8, 1, 256, 256)
├── af_train_label_seqtoseq_alll_1i_1.npy    # (N, 3, 1, 256, 256)
└── ba_train_label_seqtoseq_alll_1i_1.npy    # (N, 3, 1, 256, 256) - 正确的 BA
```

### 4.2 生成验证集

```powershell
# AF 验证集
python dataset_gen_afba.py -mode val -ts 1 -it 1 -uc af

# BA 验证集
python dataset_gen_afba.py -mode val -ts 1 -it 1 -uc ba
```

### 4.3（可选）生成测试集

```powershell
# AF 测试集
python dataset_gen_afba.py -mode test -ts 1 -it 1 -uc af

# BA 测试集
python dataset_gen_afba.py -mode test -ts 1 -it 1 -uc ba
```

---

## 第五步：生成 Pred 预测数据（firepred + lulc + BA difference 标签）

### 5.1 生成训练集

```powershell
# 生成 firepred 辅助数据 + BA 差分标签
python dataset_gen_pred.py -mode train -ts 1 -it 1
```

**⚠️ 关键**：确保使用**相同的参数** `-ts 1 -it 1`，否则会导致 N 不匹配！

**预计时间**: 1-2 小时

**输出文件**：
```
output/
├── firepred_train.npy         # (N, 19, 1, 256, 256) - 气象/地形/燃料
├── lulc_train.npy             # (N, 1, 256, 256) - 土地覆盖
└── pred_train_label.npy       # (N, 1, 1, 256, 256) - BA[t+1]-BA[t] 差分
```

### 5.2 生成验证集

```powershell
python dataset_gen_pred.py -mode val -ts 1 -it 1
```

### 5.3（可选）生成测试集

```powershell
python dataset_gen_pred.py -mode test -ts 1 -it 1
```

---

## 第六步：验证生成的数据

### 6.1 运行验证脚本

将验证脚本复制到数据输出目录：

```powershell
# 假设输出在 TS-SatFire/output/
cd C:\Users\22125\Desktop\TS-SatFire-DataGen\TS-SatFire\output

# 复制验证脚本
cp C:\Users\22125\Desktop\FireST-GraphNet\verify_regenerated_data.py .

# 运行验证（需要 utf-8 编码支持）
$env:PYTHONIOENCODING='utf-8'
python verify_regenerated_data.py --data_dir .
```

### 6.2 手动检查关键指标

```powershell
# 激活环境
conda activate ts-satfire-preprocess

# 运行快速验证
python -c "
import numpy as np

# 加载所有文件
img_train = np.load('af_train_img_seqtoseq_alll_1i_1.npy', mmap_mode='r')
label_train = np.load('af_train_label_seqtoseq_alll_1i_1.npy', mmap_mode='r')
ba_label_train = np.load('ba_train_label_seqtoseq_alll_1i_1.npy', mmap_mode='r')
firepred_train = np.load('firepred_train.npy', mmap_mode='r')
lulc_train = np.load('lulc_train.npy', mmap_mode='r')

print('='*70)
print('DATA VALIDATION RESULTS')
print('='*70)

# 检查形状
print('\n1. Shapes:')
print(f'   img:      {img_train.shape}')
print(f'   label:    {label_train.shape}')
print(f'   ba_label: {ba_label_train.shape}')
print(f'   firepred: {firepred_train.shape}')
print(f'   lulc:     {lulc_train.shape}')

# 检查 N 一致性
N_img = img_train.shape[0]
N_fire = firepred_train.shape[0]
N_lulc = lulc_train.shape[0]

print('\n2. N Consistency:')
print(f'   img N:      {N_img}')
print(f'   firepred N: {N_fire}')
print(f'   lulc N:     {N_lulc}')

if N_img == N_fire == N_lulc:
    print('   --> PASS: All N values match!')
else:
    print('   --> FAIL: N mismatch detected!')
    exit(1)

# 检查 BA 标签是否二值
ba_sample = ba_label_train[100, 0, 0]
print('\n3. BA Label Validation (ch0):')
print(f'   Min: {ba_sample.min():.2f}')
print(f'   Max: {ba_sample.max():.2f}')
print(f'   Unique values: {np.unique(ba_sample)[:10]}')

if (ba_sample > 100).any():
    print('   --> FAIL: BA labels have large integers (should be binary)')
    exit(1)
else:
    print('   --> PASS: BA labels are binary')

# 检查 AF 标签
af_sample = label_train[100, 1, 0]
print('\n4. AF Label Validation (ch1):')
print(f'   Min: {af_sample.min():.2f}')
print(f'   Max: {af_sample.max():.2f}')
print(f'   Unique values: {np.unique(af_sample)}')

if set(np.unique(af_sample)).issubset({0.0, 1.0}):
    print('   --> PASS: AF labels are binary')
else:
    print('   --> FAIL: AF labels are not binary')
    exit(1)

print('\n' + '='*70)
print('ALL VALIDATION CHECKS PASSED!')
print('Data is ready for training.')
print('='*70)
"
```

**期望输出**：
```
======================================================================
DATA VALIDATION RESULTS
======================================================================

1. Shapes:
   img:      (2254, 8, 1, 256, 256)       # 或其他 N 值
   label:    (2254, 3, 1, 256, 256)
   ba_label: (2254, 3, 1, 256, 256)
   firepred: (2254, 19, 1, 256, 256)      # ← 必须与 img 的 N 一致！
   lulc:     (2254, 1, 256, 256)

2. N Consistency:
   img N:      2254
   firepred N: 2254                       # ← 关键：必须匹配
   lulc N:     2254
   --> PASS: All N values match!

3. BA Label Validation (ch0):
   Min: 0.00
   Max: 1.00
   Unique values: [0. 1.]
   --> PASS: BA labels are binary

4. AF Label Validation (ch1):
   Min: 0.00
   Max: 1.00
   Unique values: [0. 1.]
   --> PASS: AF labels are binary

======================================================================
ALL VALIDATION CHECKS PASSED!
Data is ready for training.
======================================================================
```

---

## 第七步：整理并替换数据

### 7.1 备份当前数据

```powershell
# 返回到 FireST-GraphNet 目录
cd C:\Users\22125\Desktop\FireST-GraphNet

# 备份旧数据
mkdir old_data_backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')
cp dataset_train\*.npy old_data_backup_*\
cp dataset_val\*.npy old_data_backup_*\
```

### 7.2 复制新生成的数据

```powershell
# 从输出目录复制到 FireST-GraphNet
$output_dir = "C:\Users\22125\Desktop\TS-SatFire-DataGen\TS-SatFire\output"

# 复制训练集
cp $output_dir\af_train_img_seqtoseq_alll_1i_1.npy dataset_train\
cp $output_dir\af_train_label_seqtoseq_alll_1i_1.npy dataset_train\
cp $output_dir\firepred_train.npy dataset_train\
cp $output_dir\lulc_train.npy dataset_train\

# 复制验证集
cp $output_dir\af_val_img_seqtoseq_alll_1i_1.npy dataset_val\
cp $output_dir\af_val_label_seqtoseq_alll_1i_1.npy dataset_val\
cp $output_dir\firepred_val.npy dataset_val\
cp $output_dir\lulc_val.npy dataset_val\
```

### 7.3 最终验证

```powershell
# 在 FireST-GraphNet 目录下验证
cd C:\Users\22125\Desktop\FireST-GraphNet
conda activate wildfire_gnn
python quick_verify.py
```

**期望看到**：
```
Status: PASS - Data looks good for training
```

---

## 第八步：准备上传到 Google Drive（用于 Colab）

### 8.1 压缩数据集（可选，加快上传）

```powershell
# 压缩训练和验证数据
Compress-Archive -Path dataset_train\* -DestinationPath FireST_GraphNet_Data_Train.zip
Compress-Archive -Path dataset_val\* -DestinationPath FireST_GraphNet_Data_Val.zip
```

### 8.2 上传到 Google Drive

**方法 A：手动上传（推荐）**

1. 访问 https://drive.google.com/
2. 创建文件夹 `FireST-GraphNet/`
3. 上传 `dataset_train/` 和 `dataset_val/` 文件夹（或 zip 文件）

**方法 B：使用 rclone（自动化）**

```powershell
# 安装 rclone
choco install rclone

# 配置 Google Drive
rclone config
# 按提示配置 Google Drive 连接

# 上传数据
rclone copy dataset_train gdrive:FireST-GraphNet/dataset_train -P
rclone copy dataset_val gdrive:FireST-GraphNet/dataset_val -P
```

---

## 故障排查

### 问题 1：Kaggle API 下载失败

```powershell
# 检查 API key 是否正确配置
cat $env:USERPROFILE\.kaggle\kaggle.json

# 手动下载（如果 API 失败）
# 访问 https://www.kaggle.com/datasets/z789456sx/ts-satfire
# 点击 Download 按钮
```

### 问题 2：预处理脚本找不到数据

检查 `dataset_gen_afba.py` 中的数据路径配置：
```python
# 常见配置变量名
data_root = "..."
raw_data_dir = "..."
```

### 问题 3：N 不匹配（img ≠ firepred）

**原因**：AF/BA 和 Pred 数据生成使用了不同的参数

**解决**：
```powershell
# 删除所有输出
rm -r output\*

# 重新生成，确保使用相同参数
python dataset_gen_afba.py -mode train -ts 1 -it 1 -uc af
python dataset_gen_afba.py -mode train -ts 1 -it 1 -uc ba
python dataset_gen_pred.py -mode train -ts 1 -it 1  # ← 相同的 -ts 和 -it
```

### 问题 4：磁盘空间不足

**临时解决方案**：
```powershell
# 只生成训练集和验证集，跳过测试集
# 生成完一个模态后立即压缩和移动

# 生成 → 验证 → 移动 → 清理
python dataset_gen_afba.py -mode train -ts 1 -it 1 -uc af
cp output\*.npy C:\Users\22125\Desktop\FireST-GraphNet\dataset_train\
rm output\*.npy
```

---

## 预计完成时间表

| 步骤 | 操作 | 预计时间 | 可后台运行 |
|------|------|---------|-----------|
| 1-3 | 准备环境 | 30 分钟 | ❌ |
| 4 | 下载原始数据 | 2-6 小时 | ✅ 可过夜 |
| 5-6 | 安装依赖 | 15 分钟 | ❌ |
| 7 | 生成 AF/BA 训练数据 | 2-4 小时 | ✅ |
| 8 | 生成 AF/BA 验证数据 | 30-60 分钟 | ✅ |
| 9 | 生成 Pred 训练数据 | 1-2 小时 | ✅ |
| 10 | 生成 Pred 验证数据 | 20-40 分钟 | ✅ |
| 11 | 验证数据 | 10 分钟 | ❌ |
| 12 | 上传到 Google Drive | 1-4 小时 | ✅ |
| **总计** | | **8-18 小时** | 大部分可后台 |

**优化建议**：
- 第 4 步下载可以**过夜运行**
- 第 7-10 步数据生成可以在**工作/学习时后台运行**
- 第 12 步上传也可以**后台运行**

实际需要你操作的时间：**约 1-2 小时**

---

## 下一步

完成数据生成后，我会帮你：
1. ✅ 准备完整的三任务训练配置（AF + BA + Pred）
2. ✅ 创建 Colab notebook 模板
3. ✅ 实现多任务代码重构
4. ✅ 准备训练脚本和评估工具

**现在开始操作吧！遇到问题随时问我。**
