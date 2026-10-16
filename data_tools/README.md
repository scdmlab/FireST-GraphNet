# 数据工具包 (Data Tools)

这个文件夹包含所有用于 TS-SatFire 数据集重新生成、验证和 Colab 训练准备的工具。

---

## 📁 文件夹结构

```
data_tools/
├── docs/                           📚 文档
│   ├── DATA_REGENERATION_QUICKSTART.md          快速开始指南（推荐先看）
│   ├── DATA_REGENERATION_STEP_BY_STEP.md        详细分步教程
│   └── DATASET_REGENERATION_GUIDE.md            完整背景文档
│
├── scripts/                        🔧 自动化脚本
│   ├── automated_data_generation.ps1            一键生成数据（推荐）
│   └── prepare_for_colab.py                     准备 Colab 上传
│
├── validation/                     ✅ 验证工具
│   ├── quick_verify.py                          快速验证（推荐）
│   ├── verify_regenerated_data.py               完整验证报告
│   ├── audit_dataset_alignment.py               数据对齐审计
│   └── check_ba_init.py                         BA 初始化检查
│
└── colab/                          ☁️ Colab 支持
    └── colab_training_template.ipynb            Colab 训练模板
```

---

## 🚀 快速开始

### 1️⃣ 重新生成数据集（推荐自动化脚本）

```powershell
cd C:\Users\22125\Desktop\FireST-GraphNet
.\data_tools\scripts\automated_data_generation.ps1
```

**这个脚本会自动完成**：
- ✅ 检查 Kaggle API 配置
- ✅ 下载原始 GeoTIFF 数据（71GB）
- ✅ 克隆预处理代码
- ✅ 创建 conda 环境
- ✅ 运行数据生成（AF + BA + Pred）
- ✅ 验证生成的数据
- ✅ 复制到项目目录

**预计时间**：8-18 小时（大部分可后台运行）

---

### 2️⃣ 验证数据正确性

```powershell
# 激活环境
conda activate wildfire_gnn

# 快速验证（推荐）
python data_tools\validation\quick_verify.py

# 完整验证报告
python data_tools\validation\verify_regenerated_data.py --data_dir dataset_train

# 审计数据对齐
python data_tools\validation\audit_dataset_alignment.py
```

**期望输出**：
```
Status: PASS - Data looks good for training
✓ img N = firepred N = lulc N
✓ BA labels are binary
✓ AF labels are binary
```

---

### 3️⃣ 准备 Colab 上传

```powershell
# 压缩数据集（推荐，上传更快）
python data_tools\scripts\prepare_for_colab.py --mode compress

# 或直接准备原始文件夹
python data_tools\scripts\prepare_for_colab.py --mode direct
```

生成的文件在 `./colab_upload/` 文件夹中，上传到 Google Drive。

---

### 4️⃣ 在 Colab 训练

1. 上传 `data_tools\colab\colab_training_template.ipynb` 到 Google Colab
2. 挂载 Google Drive
3. 运行 notebook 开始训练

---

## 📚 文档说明

### 🔰 新手入门

**先看这个**：`docs/DATA_REGENERATION_QUICKSTART.md`
- 快速了解整个流程
- 时间估算
- 常见问题

### 📖 详细教程

**需要分步指导**：`docs/DATA_REGENERATION_STEP_BY_STEP.md`
- 每一步的详细命令
- 故障排查
- 验证检查点

### 📑 完整参考

**深入了解背景**：`docs/DATASET_REGENERATION_GUIDE.md`
- 为什么需要重新生成
- 数据结构解析
- 联系作者信息

---

## 🛠️ 工具使用说明

### 自动化脚本 (automated_data_generation.ps1)

**功能**：一键完成整个数据生成流程

**使用方法**：
```powershell
.\data_tools\scripts\automated_data_generation.ps1
```

**交互式选项**：
- Kaggle API 配置检查
- 数据下载确认
- 生成模式选择（train+val 或 train+val+test）
- 数据验证
- 自动复制到项目目录

---

### 验证工具对比

| 工具 | 速度 | 详细程度 | 使用场景 |
|------|------|---------|---------|
| **quick_verify.py** | ⚡ 快 | 核心检查 | 日常快速验证 |
| **verify_regenerated_data.py** | 🐢 慢 | 完整报告 | 重新生成后全面检查 |
| **audit_dataset_alignment.py** | 🐢 慢 | 最详细 | 诊断复杂问题 |
| **check_ba_init.py** | ⚡ 快 | BA 专项 | 检查 BA 初始化 |

**推荐使用**：
- 日常检查：`quick_verify.py`
- 数据生成后：`verify_regenerated_data.py`
- 发现问题时：`audit_dataset_alignment.py`

---

### Colab 上传准备 (prepare_for_colab.py)

**功能**：准备数据集以上传到 Google Drive

**使用方法**：
```powershell
# 压缩模式（推荐，上传更快）
python data_tools\scripts\prepare_for_colab.py --mode compress

# 直接模式（Colab 无需解压）
python data_tools\scripts\prepare_for_colab.py --mode direct

# 两种模式都准备
python data_tools\scripts\prepare_for_colab.py --mode both
```

**输出**：
- `colab_upload/dataset_train.zip` （压缩模式）
- `colab_upload/dataset_val.zip`
- `colab_upload/UPLOAD_INSTRUCTIONS.txt` （详细说明）
- `colab_upload/file_manifest.json` （文件清单）

---

## ✅ 工作流程

### 完整流程（首次使用）

```
1. [docs] 阅读 DATA_REGENERATION_QUICKSTART.md
         ↓
2. [scripts] 运行 automated_data_generation.ps1
         ↓（8-18 小时，大部分可后台）
         ↓
3. [validation] 运行 quick_verify.py
         ↓
4. [scripts] 运行 prepare_for_colab.py --mode compress
         ↓（1-4 小时上传）
         ↓
5. [colab] 上传 colab_training_template.ipynb 到 Colab
         ↓
6. ✅ 开始训练！
```

### 快速检查流程（日常使用）

```powershell
# 验证当前数据
conda activate wildfire_gnn
python data_tools\validation\quick_verify.py
```

---

## 🆘 故障排查

### 问题 1：Kaggle API 失败

```powershell
# 检查配置
cat $env:USERPROFILE\.kaggle\kaggle.json

# 测试连接
kaggle datasets list
```

**解决方法**：参考 `docs/DATA_REGENERATION_STEP_BY_STEP.md` 第 1.3 节

---

### 问题 2：N 不匹配（img ≠ firepred）

**症状**：`quick_verify.py` 显示 "Match: NO"

**解决方法**：
1. 查看 `docs/DATA_REGENERATION_QUICKSTART.md` 故障排查部分
2. 确保 AF/BA 和 Pred 使用相同的 `-ts` 和 `-it` 参数重新生成

---

### 问题 3：BA 标签包含大整数

**症状**：`quick_verify.py` 显示 "Has large integers: YES"

**原因**：使用了错误的 label 文件（AF 的而非 BA 的）

**解决方法**：
```powershell
# 重新生成 BA 标签
python dataset_gen_afba.py -mode train -ts 1 -it 1 -uc ba
```

---

### 问题 4：磁盘空间不足

**临时方案**：分批生成和清理

查看详细解决方案：`docs/DATA_REGENERATION_QUICKSTART.md` 故障排查部分

---

## 📝 当前数据状态

运行 `quick_verify.py` 查看当前数据状态：

```
当前状态（重新生成前）：
✅ AF 任务标签正确（label ch1）
❌ firepred N 不匹配（2634 ≠ 2254）
❌ lulc N 不匹配（2634 ≠ 2254）
❌ BA 标签格式错误
❌ Pred 标签缺失

重新生成后（预期）：
✅ AF 任务标签正确
✅ firepred N 匹配
✅ lulc N 匹配  
✅ BA 标签二值化
✅ Pred BA 差分标签
```

---

## 🎯 下一步

完成数据生成和验证后：

1. ✅ 数据上传到 Google Drive
2. ✅ 在 Colab 开始训练 AF 任务
3. ✅ 等待多任务代码重构（AF + BA + Pred）
4. ✅ 进行完整的三任务训练和评估

---

## 📧 联系与支持

**数据集作者**（如需要预处理好的数据）：
- Yu Zhao: zhaoyutim@gmail.com
- Sebastian Gerard: sgerard@kth.se

**TS-SatFire GitHub**：
- https://github.com/zhaoyutim/TS-SatFire

**Kaggle 数据集**：
- https://www.kaggle.com/datasets/z789456sx/ts-satfire

---

## 📌 重要提示

⚠️ **数据重新生成是必须的**，当前数据存在以下问题无法修复：
1. firepred/lulc N 不匹配导致索引错位
2. BA 标签格式错误（大整数而非二值 mask）
3. Pred 任务缺少 BA 差分标签

❌ **不要跳过这一步**，否则：
- 无法使用辅助气象数据
- 无法训练 BA 任务
- 无法训练 Pred 任务
- 只能训练受限的 AF 任务

✅ **重新生成后可获得**：
- 完整的三任务支持（AF + BA + Pred）
- 正确对齐的辅助数据
- 可靠的训练结果

---

**开始使用：**

```powershell
# 阅读快速入门指南
cat data_tools\docs\DATA_REGENERATION_QUICKSTART.md

# 运行自动化脚本
.\data_tools\scripts\automated_data_generation.ps1
```

**祝你顺利！** 🚀
