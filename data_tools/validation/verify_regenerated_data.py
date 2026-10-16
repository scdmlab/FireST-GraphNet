"""
验证重新生成的数据集是否正确对齐并且标签格式正确。

运行此脚本以确保：
1. 所有 npy 文件的 N（样本数）一致
2. BA 标签是二值的（不是大整数）
3. AF 标签是二值的
4. firepred/lulc 的 N 与 img/label 匹配

使用方法：
    python verify_regenerated_data.py --data_dir dataset_train
"""

import argparse
import numpy as np
from pathlib import Path


def verify_dataset(data_dir: Path):
    """验证数据集的对齐性和标签正确性。"""
    
    print(f"{'='*70}")
    print(f"  TS-SatFire 数据集验证")
    print(f"  数据目录: {data_dir}")
    print(f"{'='*70}\n")
    
    errors = []
    warnings = []
    
    # ═══════════════════════════════════════════════════════════════════
    #  1. 检查文件是否存在
    # ═══════════════════════════════════════════════════════════════════
    print("1. 文件存在性检查")
    print("-" * 70)
    
    required_files = {
        'img': '*_img_*.npy',
        'label': '*_label_*.npy',
        'firepred': 'firepred_*.npy',
        'lulc': 'lulc_*.npy',
    }
    
    files_found = {}
    for key, pattern in required_files.items():
        matches = list(data_dir.glob(pattern))
        if matches:
            files_found[key] = matches[0]
            print(f"  ✅ {key:10s}: {matches[0].name}")
        else:
            files_found[key] = None
            errors.append(f"缺失文件: {pattern}")
            print(f"  ❌ {key:10s}: 未找到匹配 {pattern} 的文件")
    
    if errors:
        print(f"\n❌ 文件缺失，无法继续验证。")
        for e in errors:
            print(f"   - {e}")
        return False
    
    # ═══════════════════════════════════════════════════════════════════
    #  2. 加载数据并检查形状
    # ═══════════════════════════════════════════════════════════════════
    print("\n2. 数据形状检查")
    print("-" * 70)
    
    try:
        img = np.load(str(files_found['img']), mmap_mode='r')
        label = np.load(str(files_found['label']), mmap_mode='r')
        firepred = np.load(str(files_found['firepred']), mmap_mode='r')
        lulc = np.load(str(files_found['lulc']), mmap_mode='r')
        
        print(f"  img:      {img.shape}")
        print(f"  label:    {label.shape}")
        print(f"  firepred: {firepred.shape}")
        print(f"  lulc:     {lulc.shape}")
        
        # 检查预期形状
        expected_img_shape = (None, 8, 1, 256, 256)
        expected_label_shape = (None, 3, 1, 256, 256)
        expected_firepred_shape = (None, 19, 1, 256, 256)
        expected_lulc_shape = (None, 1, 256, 256)
        
        if img.shape[1:] != expected_img_shape[1:]:
            errors.append(f"img 形状不正确: {img.shape} (预期 (N, 8, 1, 256, 256))")
        
        if label.shape[1:] != expected_label_shape[1:]:
            errors.append(f"label 形状不正确: {label.shape} (预期 (N, 3, 1, 256, 256))")
        
        if firepred.shape[1:] != expected_firepred_shape[1:]:
            warnings.append(f"firepred 形状异常: {firepred.shape} (预期 (N, 19, 1, 256, 256))")
        
        if lulc.shape[1:] != expected_lulc_shape[1:]:
            warnings.append(f"lulc 形状异常: {lulc.shape} (预期 (N, 1, 256, 256))")
        
    except Exception as e:
        errors.append(f"加载数据失败: {e}")
        print(f"\n❌ 数据加载失败: {e}")
        return False
    
    # ═══════════════════════════════════════════════════════════════════
    #  3. 检查 N（样本数）一致性
    # ═══════════════════════════════════════════════════════════════════
    print("\n3. 样本数 N 一致性检查")
    print("-" * 70)
    
    N_img = img.shape[0]
    N_label = label.shape[0]
    N_firepred = firepred.shape[0]
    N_lulc = lulc.shape[0]
    
    print(f"  img N:      {N_img}")
    print(f"  label N:    {N_label}")
    print(f"  firepred N: {N_firepred}")
    print(f"  lulc N:     {N_lulc}")
    
    if N_img != N_label:
        errors.append(f"img 和 label 的 N 不一致: {N_img} vs {N_label}")
    
    if N_img != N_firepred:
        errors.append(f"⚠️ CRITICAL: img 和 firepred 的 N 不一致: {N_img} vs {N_firepred}")
        print(f"  ❌ img N ({N_img}) ≠ firepred N ({N_firepred})  差值: {abs(N_img - N_firepred)}")
    else:
        print(f"  ✅ img N = firepred N = {N_img}")
    
    if N_img != N_lulc:
        errors.append(f"⚠️ CRITICAL: img 和 lulc 的 N 不一致: {N_img} vs {N_lulc}")
        print(f"  ❌ img N ({N_img}) ≠ lulc N ({N_lulc})  差值: {abs(N_img - N_lulc)}")
    else:
        print(f"  ✅ img N = lulc N = {N_img}")
    
    # ═══════════════════════════════════════════════════════════════════
    #  4. 检查标签值范围和类型
    # ═══════════════════════════════════════════════════════════════════
    print("\n4. 标签值范围检查（抽样 10 个样本）")
    print("-" * 70)
    
    sample_indices = np.linspace(0, N_img - 1, min(10, N_img), dtype=int)
    
    # 检查 AF 标签（channel 1）
    print("  AF 标签（channel 1）：")
    af_all_binary = True
    for i in sample_indices[:3]:  # 只打印前 3 个
        af_mask = label[i, 1, 0]
        unique_vals = np.unique(af_mask)
        is_binary = set(unique_vals).issubset({0.0, 1.0})
        af_all_binary = af_all_binary and is_binary
        print(f"    样本 {i}: min={af_mask.min():.2f}, max={af_mask.max():.2f}, "
              f"unique={list(unique_vals)[:5]} {'✅' if is_binary else '❌'}")
    
    if not af_all_binary:
        errors.append("AF 标签（channel 1）不是二值的")
    else:
        print(f"    ✅ 所有抽样的 AF 标签都是二值的")
    
    # 检查 BA 标签（channel 0）
    print("\n  BA 标签（channel 0）：")
    ba_all_binary = True
    ba_has_large_ints = False
    for i in sample_indices[:3]:
        ba_mask = label[i, 0, 0]
        unique_vals = np.unique(ba_mask)
        is_binary = set(unique_vals).issubset({0.0, 1.0, -1.0})  # 允许 -1 哨兵值
        has_large = (ba_mask > 100).any()
        ba_all_binary = ba_all_binary and is_binary
        ba_has_large_ints = ba_has_large_ints or has_large
        print(f"    样本 {i}: min={ba_mask.min():.0f}, max={ba_mask.max():.0f}, "
              f"unique={list(unique_vals)[:5]} {'✅' if is_binary else '❌'}")
    
    if ba_has_large_ints:
        errors.append("⚠️ CRITICAL: BA 标签（channel 0）包含大整数，不是二值 mask！")
        print(f"    ❌ BA 标签包含大整数（> 100），这是**错误的数据**")
        print(f"       正确的 BA 标签应该是二值 {{0.0, 1.0}} 或包含哨兵值 -1")
    elif ba_all_binary:
        print(f"    ✅ 所有抽样的 BA 标签都是二值的")
    
    # 检查 channel 2（热辐射，仅警告）
    print("\n  标签 Channel 2（应为热辐射数据）：")
    ch2_sample = label[sample_indices[0], 2, 0]
    print(f"    样本 0: min={ch2_sample.min():.1f}, max={ch2_sample.max():.1f}")
    if ch2_sample.min() < 100 or ch2_sample.max() > 400:
        warnings.append("Channel 2 值范围异常（预期 270-385K 热辐射）")
    
    # ═══════════════════════════════════════════════════════════════════
    #  5. 汇总报告
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("  验证结果汇总")
    print("="*70)
    
    if not errors:
        print("\n✅ 所有关键检查通过！数据集可以用于训练。\n")
        if warnings:
            print("⚠️  警告（不影响训练）：")
            for w in warnings:
                print(f"   - {w}")
        return True
    else:
        print("\n❌ 发现 {} 个错误：\n".format(len(errors)))
        for e in errors:
            print(f"   - {e}")
        print("\n建议：")
        print("  1. 如果 firepred/lulc N 不匹配：重新生成数据，确保使用相同参数")
        print("  2. 如果 BA 标签包含大整数：使用 dataset_gen_afba.py -uc ba 重新生成")
        print("  3. 参考 DATASET_REGENERATION_GUIDE.md 获取完整指南")
        print()
        return False


def main():
    parser = argparse.ArgumentParser(description="验证 TS-SatFire 数据集")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="dataset_train",
        help="数据目录路径"
    )
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"❌ 数据目录不存在: {data_dir}")
        return
    
    success = verify_dataset(data_dir)
    
    if success:
        print("="*70)
        print("  下一步：在 Colab 上训练")
        print("="*70)
        print("  1. 将数据上传到 Google Drive")
        print("  2. 在 Colab 中挂载 Drive")
        print("  3. 运行训练脚本")
        print()


if __name__ == "__main__":
    main()
