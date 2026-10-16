# 数据重新生成 - 快速开始指南

## 🎯 目标

重新生成完整的 TS-SatFire 数据集，解决以下问题：
1. ❌ firepred/lulc N 不匹配（2634 ≠ 2254）
2. ❌ BA 标签格式错误（大整数而非二值 mask）
3. ❌ Pred 任务缺失 BA 差分标签

## ⚡ 快速开始（两种方式）

### 方式 1：自动化脚本（推荐）

**一键运行**PowerShell 自动化脚本：

```powershell
cd C:\Users\22125\Desktop\FireST-GraphNet
.\automated_data_generation.ps1
```

脚本会自动完成：
- ✅ 检查 Kaggle API 配置
- ✅ 下载原始 GeoTIFF 数据（71GB）
- ✅ 克隆预处理代码
- ✅ 创建 conda 环境
- ✅ 运行数据生成（AF + BA + Pred）
- ✅ 验证生成的数据
- ✅ 复制到项目目录

**预计时间**: 8-18 小时（大部分可后台运行）

---

### 方式 2：手动分步操作

按照详细指南逐步操作：

**查看完整指南**：
```
DATA_REGENERATION_STEP_BY_STEP.md
```

**主要步骤**：
1. 配置 Kaggle API
2. 下载原始数据（2-6 小时）
3. 运行预处理脚本（4-8 小时）
4. 验证数据正确性
5. 上传到 Google Drive

---

## 📋 前置要求

### 1. Kaggle API 配置

1. 访问 https://www.kaggle.com/me/account
2. 点击 **Create New API Token**
3. 将 `kaggle.json` 保存到 `C:\Users\22125\.kaggle\`

### 2. 磁盘空间

确保至少有 **150GB** 可用空间：
- 原始数据: ~71GB
- 预处理输出: ~30GB
- 临时文件: ~50GB

### 3. 软件依赖

- Python 3.9+
- Conda
- Git
- PowerShell 5.1+

---

## 🔍 验证数据

### 在本地验证

```powershell
cd C:\Users\22125\Desktop\FireST-GraphNet
conda activate wildfire_gnn
python quick_verify.py
```

**期望输出**：
```
Status: PASS - Data looks good for training
✓ img N = firepred N = lulc N
✓ BA labels are binary
✓ AF labels are binary
```

### 关键检查项

| 检查项 | 要求 | 当前状态 |
|--------|------|---------|
| img N = firepred N | ✅ 必须相等 | ❌ 2254 ≠ 2634 |
| img N = lulc N | ✅ 必须相等 | ❌ 2254 ≠ 2634 |
| BA label 二值 | ✅ {0, 1} | ❌ 大整数 |
| AF label 二值 | ✅ {0, 1} | ✅ 已正确 |

---

## 📤 上传到 Google Drive（用于 Colab）

### 方式 A：使用压缩文件（推荐）

```powershell
# 准备压缩文件
conda activate wildfire_gnn
python prepare_for_colab.py --mode compress

# 上传生成的 zip 文件到 Google Drive
# 文件位置: ./colab_upload/
#   - dataset_train.zip
#   - dataset_val.zip
```

**优点**：
- ✅ 上传更快（压缩后 ~15-20GB）
- ✅ 文件数量少

**缺点**：
- ⚠️ 需要在 Colab 解压（5-10 分钟）

### 方式 B：直接上传原始文件夹

```powershell
# 准备清单
python prepare_for_colab.py --mode direct

# 手动上传整个文件夹到 Google Drive:
#   FireST-GraphNet/
#   ├── dataset_train/
#   └── dataset_val/
```

**优点**：
- ✅ Colab 可直接使用，无需解压

**缺点**：
- ⚠️ 上传较慢（~25GB 原始文件）
- ⚠️ 文件数量多

---

## 🚀 在 Colab 上训练

### 1. 上传 Colab Notebook

将 `colab_training_template.ipynb` 上传到 Google Colab

### 2. 运行 Notebook

按照 notebook 中的步骤：
1. ✅ 挂载 Google Drive
2. ✅ 加载数据集
3. ✅ 配置训练任务（AF/BA/Pred）
4. ✅ 开始训练
5. ✅ 保存结果回 Drive

---

## 🛠️ 故障排查

### 问题 1：Kaggle API 下载失败

**症状**：`kaggle datasets download` 报错

**解决**：
```powershell
# 检查 API key
cat $env:USERPROFILE\.kaggle\kaggle.json

# 测试连接
kaggle datasets list
```

### 问题 2：N 不匹配（img ≠ firepred）

**症状**：验证脚本显示 N 不一致

**原因**：AF/BA 和 Pred 使用了不同的生成参数

**解决**：
```powershell
# 删除输出重新生成，确保使用相同参数
cd TS-SatFire-DataGen\TS-SatFire
rm -r output\*

# 重新生成（注意 -ts 和 -it 必须相同）
python dataset_gen_afba.py -mode train -ts 1 -it 1 -uc af
python dataset_gen_afba.py -mode train -ts 1 -it 1 -uc ba
python dataset_gen_pred.py -mode train -ts 1 -it 1  # ← 相同的参数！
```

### 问题 3：BA 标签仍然是大整数

**症状**：验证脚本显示 "Has large integers: YES"

**原因**：使用了错误的 npy 文件（AF 的 label 而非 BA 的 label）

**解决**：
```powershell
# 确保使用正确的 BA 标签文件
# 检查文件名: ba_train_label_*.npy（不是 af_train_label_*.npy）

# 如果 BA 生成失败，重新运行
python dataset_gen_afba.py -mode train -ts 1 -it 1 -uc ba
```

### 问题 4：磁盘空间不足

**临时解决方案**：
```powershell
# 分批生成和清理
python dataset_gen_afba.py -mode train -ts 1 -it 1 -uc af
cp output\*.npy C:\Users\22125\Desktop\FireST-GraphNet\dataset_train\
rm output\*.npy

python dataset_gen_afba.py -mode train -ts 1 -it 1 -uc ba
cp output\*.npy C:\Users\22125\Desktop\FireST-GraphNet\dataset_train\
rm output\*.npy

# ... 继续其他模态
```

---

## 📊 时间估算

| 步骤 | 操作 | 预计时间 | 可后台运行 |
|------|------|---------|-----------|
| 1 | 配置环境 | 30 分钟 | ❌ |
| 2 | 下载原始数据 (71GB) | 2-6 小时 | ✅ 可过夜 |
| 3 | 生成 AF/BA 训练数据 | 2-4 小时 | ✅ |
| 4 | 生成 AF/BA 验证数据 | 30-60 分钟 | ✅ |
| 5 | 生成 Pred 训练数据 | 1-2 小时 | ✅ |
| 6 | 生成 Pred 验证数据 | 20-40 分钟 | ✅ |
| 7 | 验证数据 | 10 分钟 | ❌ |
| 8 | 上传到 Google Drive | 1-4 小时 | ✅ |
| **总计** | | **8-18 小时** | ~80% 可后台 |

**实际操作时间**：约 1-2 小时

---

## 📚 相关文件

- **`automated_data_generation.ps1`** - 自动化脚本（推荐）
- **`DATA_REGENERATION_STEP_BY_STEP.md`** - 详细步骤指南
- **`quick_verify.py`** - 数据验证脚本
- **`prepare_for_colab.py`** - Colab 上传准备工具
- **`colab_training_template.ipynb`** - Colab 训练模板

---

## ✅ 检查清单

在开始训练前，确保：

- [ ] Kaggle API 已配置
- [ ] 原始 GeoTIFF 数据已下载（71GB）
- [ ] 预处理脚本已运行完成
- [ ] **`python quick_verify.py` 显示 PASS**
- [ ] 数据已上传到 Google Drive
- [ ] Colab notebook 已准备好

---

## 🆘 需要帮助？

如果遇到问题：

1. 查看 `DATA_REGENERATION_STEP_BY_STEP.md` 详细步骤
2. 运行 `python quick_verify.py` 诊断问题
3. 检查上述故障排查部分
4. 或考虑**联系数据集作者**要对齐的数据：
   - Yu Zhao: zhaoyutim@gmail.com
   - Sebastian Gerard: sgerard@kth.se

---

## 下一步

数据生成完成后：

1. ✅ 验证数据正确性
2. ✅ 上传到 Google Drive
3. ✅ 在 Colab 上开始训练
4. ✅ 实现多任务代码重构（我会帮你完成）

**现在就开始吧！运行自动化脚本或按照详细指南操作。**

```powershell
# 开始！
.\automated_data_generation.ps1
```
