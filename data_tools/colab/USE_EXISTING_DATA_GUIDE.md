# 🔧 使用已有原始数据的修改说明

## ✅ 你的情况

你已经在 Google Drive 中有完整的原始数据：
- 位置：`My Drive/FireST-GraphNet/ts-satfire_raw/`
- 结构：192 个火灾事件，10,811 个 GeoTIFF 文件，71GB

**无需重新下载！跳过 2-4 小时的下载时间！**

---

## 📝 Notebook 修改步骤

打开 `data_tools/colab/colab_data_generation_and_training.ipynb`，按以下方式修改：

---

### 修改 1: 更新标题和说明（Cell 1）

将第 1 个单元格改为：

```markdown
# 🔥 FireST-GraphNet: Colab 端到端流程

**功能**：
1. ✅ 使用 Google Drive 中已有的原始数据（跳过下载）
2. ✅ 运行数据预处理生成训练数据
3. ✅ 验证数据正确性
4. ✅ 训练模型（AF + BA + Pred）
5. ✅ 保存结果到 Google Drive

**估计时间**（使用已有数据）：
- ~~下载数据~~：**已跳过** ✅
- 数据生成：3-5 小时
- 训练（30 epochs）：2-4 小时
- **总计**：5-9 小时（节省 2-4 小时！）
```

---

### 修改 2: 更新准备工作说明（Cell 2）

```markdown
## 📋 准备工作

确保以下文件夹已在 Google Drive：

```
My Drive/
└── FireST-GraphNet/
    ├── FireST-GraphNet-Clean/     # 项目代码
    │   ├── firest_graphnet/
    │   ├── data_tools/validation/
    │   └── config_run11.yaml
    └── ts-satfire_raw/            # 原始数据（已有）
        ├── 20562846/              # 事件1
        ├── 20568194/              # 事件2
        └── ... (共192个事件)
```
```

---

### 修改 3: 更新挂载 Google Drive（Cell 5）

将原来的 Cell 5 修改为：

```python
# 挂载 Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 设置路径
PROJECT_PATH = "/content/drive/MyDrive/FireST-GraphNet/FireST-GraphNet-Clean"
RAW_DATA_PATH = "/content/drive/MyDrive/FireST-GraphNet/ts-satfire_raw"

# 验证路径
from pathlib import Path

assert Path(PROJECT_PATH).exists(), f"❌ 未找到项目: {PROJECT_PATH}"
assert Path(RAW_DATA_PATH).exists(), f"❌ 未找到原始数据: {RAW_DATA_PATH}"

print(f"✅ 项目路径: {PROJECT_PATH}")
print(f"✅ 原始数据: {RAW_DATA_PATH}")
```

---

### 修改 4: **删除或跳过** Cells 6-9（Kaggle 和下载相关）

**不要运行以下单元格**（或直接删除）：
- Cell 6: 上传 Kaggle API 密钥
- Cell 7: Step 2 标题（下载数据）
- Cell 8: 下载数据
- Cell 9: 解压数据
- Cell 10: 删除 zip 文件

---

### 修改 5: 替换为验证原始数据（新 Cell）

在 **Step 3 之前**添加新单元格：

```python
# === Step 2: 验证原始数据（跳过下载）===

DATA_DIR = RAW_DATA_PATH

print("=== 使用已有的原始数据 ===\n")
print(f"路径: {DATA_DIR}\n")

# 验证数据完整性
from pathlib import Path

raw_data = Path(DATA_DIR)
if not raw_data.exists():
    raise FileNotFoundError(f"❌ 未找到原始数据: {DATA_DIR}")

# 统计事件
event_dirs = [d for d in raw_data.iterdir() if d.is_dir() and d.name.isdigit()]
print(f"✅ 找到 {len(event_dirs)} 个火灾事件")

# 检查数据类型
if event_dirs:
    sample = event_dirs[0]
    subdirs = [d.name for d in sample.iterdir() if d.is_dir()]
    print(f"✅ 数据类型: {', '.join(subdirs)}")
    
    # 统计文件
    total_tif = sum(1 for _ in raw_data.rglob('*.tif'))
    total_size = sum(f.stat().st_size for f in raw_data.rglob('*.tif')) / (1024**3)
    print(f"✅ GeoTIFF 文件: {total_tif:,} 个")
    print(f"✅ 总大小: {total_size:.2f} GB")

print(f"\n🎉 数据验证完成！")
print(f"⏱️  节省时间：2-4 小时（跳过下载）")
```

---

### 修改 6: 配置预处理脚本路径

在 **Step 4 开始前**（生成 AF 数据之前），添加配置单元格：

```python
# 配置 TS-SatFire 预处理脚本的数据路径

import subprocess

print("=== 配置数据路径 ===\n")

# 进入预处理目录
import os
os.chdir(PREPROCESS_DIR)

# 方法1: 创建软链接（推荐）
link_path = Path(PREPROCESS_DIR) / "data"

print(f"创建软链接: {link_path} -> {DATA_DIR}\n")

# 删除旧链接
if link_path.exists() and link_path.is_symlink():
    link_path.unlink()

# 创建新链接
if not link_path.exists():
    result = subprocess.run(
        ["ln", "-s", str(DATA_DIR), str(link_path)],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(f"✅ 软链接创建成功")
    else:
        print(f"⚠️  软链接失败，使用环境变量")
        os.environ['DATA_DIR'] = str(DATA_DIR)

# 验证
print(f"\n验证数据访问:")
if link_path.exists():
    print(f"✅ {link_path} -> {DATA_DIR}")
else:
    print(f"⚠️  将使用完整路径: {DATA_DIR}")

print(f"\n💡 如果预处理失败，请手动修改 TS-SatFire 脚本中的数据路径")
```

---

### 修改 7: 更新预处理命令（可选）

如果 TS-SatFire 脚本支持 `--data_dir` 参数，修改生成命令：

```python
# 生成训练集 AF 数据
!python dataset_gen_afba.py --data_dir {DATA_DIR} -mode train -ts 1 -it 1 -uc af
```

**注意**：如果脚本不支持此参数，软链接方法应该可以工作。

---

## 🚀 完整运行流程

1. ✅ 打开修改后的 notebook
2. ✅ 运行 Cell 1-3（环境设置）
3. ✅ 运行 Cell 4（挂载 Drive）
4. ✅ **跳过** Cells 5-9（Kaggle 下载相关）
5. ✅ 运行新添加的验证数据单元格
6. ✅ 运行 Step 3（安装依赖）
7. ✅ 运行配置路径单元格
8. ✅ 运行 Step 4-9（预处理、验证、训练）

---

## ⏱️ 新的时间估算

| 步骤 | 原计划 | 使用已有数据 | 节省 |
|------|--------|-------------|------|
| 下载数据 | 2-4小时 | **0分钟** ✅ | 2-4小时 |
| 解压数据 | 15-30分钟 | **0分钟** ✅ | 15-30分钟 |
| 数据生成 | 3-5小时 | 3-5小时 | - |
| 训练 | 2-4小时 | 2-4小时 | - |
| **总计** | **7-13小时** | **5-9小时** | **2-4.5小时** |

---

## 🆘 故障排除

### 问题 1: 预处理脚本找不到数据

**症状**：运行 `dataset_gen_afba.py` 时报错找不到文件

**解决方案**：

1. **检查软链接是否成功**：
   ```python
   !ls -la {PREPROCESS_DIR}/data
   ```

2. **手动修改 TS-SatFire 脚本**：
   - 打开 `dataset_gen_afba.py`
   - 找到数据路径定义（通常在文件开头）
   - 修改为：`DATA_DIR = "/content/drive/MyDrive/FireST-GraphNet/ts-satfire_raw"`

3. **使用绝对路径**：
   ```python
   # 如果脚本支持命令行参数
   !python dataset_gen_afba.py --data_root {DATA_DIR} -mode train -ts 1 -it 1 -uc af
   ```

### 问题 2: 权限错误

**症状**：`Permission denied` 访问 Google Drive

**解决方案**：
```python
# 重新挂载 Drive
from google.colab import drive
drive.flush_and_unmount()
drive.mount('/content/drive')
```

### 问题 3: 内存不足

**症状**：预处理时 Colab 崩溃

**解决方案**：
- 使用 Colab Pro（更多内存）
- 分批处理（先只生成 train，再生成 val）

---

## ✅ 验证成功

运行后应该看到：
- ✅ 找到 192 个火灾事件
- ✅ 数据类型: VIIRS_Day, VIIRS_Night, FirePred, ESRI_LULC
- ✅ GeoTIFF 文件: 10,811 个
- ✅ 总大小: 71.07 GB
- ✅ 软链接创建成功

---

**现在开始运行吧！** 🚀
