# 🔥 Colab 端到端训练指南

## 📋 目标

在 Google Colab 上完成从数据下载到模型训练的全流程，**无需在本地处理任何大文件**。

---

## 🚀 快速开始

### 1. 准备 Google Drive

上传以下文件夹到 Google Drive：

```
My Drive/
└── FireST-GraphNet/
    └── FireST-GraphNet-Clean/        # 项目代码根目录
        ├── firest_graphnet/          # 必需：模型代码
        │   ├── __init__.py
        │   ├── models/
        │   ├── data/
        │   ├── training/
        │   ├── scripts/
        │   └── utils/
        ├── data_tools/validation/    # 必需：验证脚本
        │   └── quick_verify.py
        ├── config_run11.yaml         # 必需：训练配置
        └── outputs/                  # 可选：结果保存位置
```

**上传方式**：
- 方式 1：直接拖拽文件夹到 Google Drive 网页
- 方式 2：使用 Google Drive 桌面客户端同步

---

### 2. 准备 Kaggle API 密钥

1. 访问 https://www.kaggle.com/me/account
2. 滚动到 "API" 部分
3. 点击 "Create New API Token"
4. 下载 `kaggle.json` 文件（稍后在 Colab 中上传）

---

### 3. 打开 Colab Notebook

在 Colab 中打开刚创建的 notebook：

```
FireST-GraphNet/data_tools/colab/colab_data_generation_and_training.ipynb
```

或直接上传到 Colab。

---

### 4. 按顺序执行所有单元格

**重要步骤**：

1. **检查 GPU**（Step 1）
   - 确保分配到 GPU（T4 或更好）
   - 如果没有 GPU，点击"运行时" → "更改运行时类型" → 选择 GPU

2. **挂载 Google Drive**（Step 1）
   - 授权访问 Google Drive

3. **上传 Kaggle 密钥**（Step 1）
   - 上传下载的 `kaggle.json` 文件

4. **下载数据**（Step 2，2-4 小时）
   - 可以让单元格运行并去做其他事情
   - **不要关闭浏览器标签页**

5. **生成数据**（Step 4，3-5 小时）
   - 分别运行 AF、BA、Pred 数据生成
   - 每个步骤显示预计时间

6. **验证数据**（Step 5）
   - 检查 `Status: PASS`
   - 确保 N 一致

7. **训练模型**（Step 7，2-4 小时）
   - 选择任务（AF、BA 或 Pred）
   - 监控训练进度

8. **保存结果**（Step 8）
   - 自动保存到 Google Drive

---

## ⏱️ 总时间估算

| 步骤 | 时间 | 可暂停 |
|------|------|--------|
| 环境设置 | 5 分钟 | ❌ |
| 下载数据 | 2-4 小时 | ✅ 可后台 |
| 解压数据 | 15-30 分钟 | ❌ |
| 安装依赖 | 5 分钟 | ❌ |
| 生成 AF 数据 | 2-2.5 小时 | ✅ 可后台 |
| 生成 BA 数据 | 2-2.5 小时 | ✅ 可后台 |
| 生成 Pred 数据 | 1-2 小时 | ✅ 可后台 |
| 验证数据 | 10 分钟 | ❌ |
| 训练模型 | 2-4 小时 | ✅ 可后台 |
| **总计** | **10-18 小时** | 85% 可后台 |

**实际操作时间**：约 **1-2 小时**（其余为自动运行）

---

## 💡 重要提示

### Colab 会话时间限制

- **免费版**：最长 12 小时后自动断开
- **Colab Pro**：最长 24 小时

**建议策略**：

**方案 1**：分两次运行
```
第1次会话（6-10 小时）：
  Step 1-5: 下载 + 生成 + 验证数据
  → 将数据保存到 Google Drive（Step 8，设置 SAVE_DATA=True）

第2次会话（2-4 小时）：
  Step 6-9: 从 Drive 加载数据 + 训练
```

**方案 2**：一次性运行（需要 Pro 或运气好）
```
连续运行 Step 1-9（10-18 小时）
```

### 保持会话活跃

- **不要关闭浏览器标签页**
- 定期点击任何单元格（防止自动断开）
- 使用浏览器扩展（如 Colab Keep Alive）

### 磁盘空间管理

Colab 临时磁盘约 **80-100GB**：

```
原始数据（解压后）：    ~50 GB
生成的训练数据：        ~15 GB
生成的验证数据：        ~2 GB
训练输出：              ~1 GB
剩余空间：              ~20 GB
```

如果空间不足：
1. 删除原始数据（Step 2 最后一个单元格）
2. 只生成 train+val（不生成 test）

---

## 🔍 故障排除

### 问题 1：下载数据失败

**错误信息**：`401 Unauthorized` 或 `403 Forbidden`

**解决方案**：
1. 确认 Kaggle 账号已验证（需要绑定手机号）
2. 重新下载 `kaggle.json`
3. 检查 `~/.kaggle/kaggle.json` 权限（应为 600）

### 问题 2：找不到生成的 .npy 文件

**可能原因**：
- 预处理脚本执行失败（检查单元格输出）
- 输出目录不同（检查 `TS-SatFire` 代码）

**解决方案**：
```python
# 在 Colab 中运行
!find /content -name "*.npy" -type f
```

### 问题 3：N 不匹配

**错误信息**：`img N != firepred N`

**原因**：预处理脚本参数不一致

**解决方案**：
- 确保所有 `dataset_gen_*.py` 都使用 `-ts 1 -it 1`
- 重新运行所有生成步骤（Step 4）

### 问题 4：会话断开

**如果在数据生成阶段断开**：
1. 检查 `/content/ts-satfire-data` 是否有数据
2. 从断开的步骤继续运行
3. 或者将已下载的数据保存到 Drive

**如果在训练阶段断开**：
1. 检查 `/content/outputs` 是否有 checkpoint
2. 修改训练脚本从 checkpoint 恢复
3. 或者重新开始训练

### 问题 5：GPU 内存不足

**错误信息**：`CUDA out of memory`

**解决方案**：
1. 减小 batch size（config 中修改或命令行参数）
2. 使用梯度累积
3. 重启运行时释放内存

---

## 📊 数据验证检查点

### 必须通过的检查

在开始训练前，确保：

- [ ] `Status: PASS`
- [ ] `img N = firepred N = lulc N`（例如都是 2254）
- [ ] `BA labels: Binary YES`（不是 large integers）
- [ ] `AF labels: Binary YES`
- [ ] 所有 4 个文件存在（img, label, firepred, lulc）

**验证命令**：
```python
python data_tools/validation/quick_verify.py \
  --train_dir /content/dataset_train \
  --val_dir /content/dataset_val
```

---

## 🎯 优势总结

与本地生成再上传相比：

| 方面 | Colab 方案 | 本地方案 |
|------|-----------|---------|
| 下载 71GB 数据 | ✅ 快速（Kaggle → Colab） | ❌ 慢（Kaggle → 本地） |
| 预处理时间 | ✅ 快（多核 CPU） | ⚠️ 取决于本地配置 |
| 上传数据 | ✅ 无需上传 | ❌ 需上传 15GB（1-4 小时） |
| 磁盘占用 | ✅ 临时空间 | ❌ 占用本地 ~70GB |
| 总时间 | **10-18 小时** | **15-25 小时** |
| 实际操作 | **1-2 小时** | **3-5 小时** |

---

## 📚 相关文档

- [data_tools/README.md](../README.md) - 数据工具总览
- [docs/DATA_REGENERATION_QUICKSTART.md](../docs/DATA_REGENERATION_QUICKSTART.md) - 本地生成快速指南
- [validation/quick_verify.py](../validation/quick_verify.py) - 数据验证脚本

---

## 🆘 需要帮助？

1. 检查 Colab notebook 中的输出和错误信息
2. 参考上方"故障排除"部分
3. 查看 TS-SatFire 仓库的 Issues：https://github.com/zhaoyutim/TS-SatFire/issues

---

**开始使用**：上传项目到 Google Drive，打开 Colab notebook，按顺序执行！
