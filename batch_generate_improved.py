"""
改进的批量生成脚本 - 筛选时间演化明显的多样化事件

筛选标准:
1. 火灾在5个时间步中有明显增长/变化
2. 事件之间空间位置和形状多样性
3. GPU加速生成（如果可用）
"""

import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

# 导入主脚本的函数
from visualize_illustration import build_figure


# ═════════════════════════════════════════════════════════════════════════════
# 事件筛选和多样性检测
# ═════════════════════════════════════════════════════════════════════════════

def load_all_sequences(data_dir: str = "dataset_val", T: int = 5) -> List[Tuple[int, np.ndarray]]:
    """使用forecast dataset加载真正的多时间步序列"""
    try:
        import yaml, torch
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from firest_graphnet.data.dataset_forecast import TSSatFireForecastDataset
        from firest_graphnet.data.firepred_provider import build_firepred_provider
        from firest_graphnet.data.static_provider import build_static_provider
        
        # 加载配置
        cfg_path = Path("outputs/output-clean/run20/config.yaml")
        if not cfg_path.exists():
            cfg_path = Path("config_run20.yaml")
        if not cfg_path.exists():
            raise FileNotFoundError("找不到config文件")
        
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        
        fc = cfg.get("forecast", {})
        dc = cfg.get("data", {})
        img_size = int(dc.get("img_size", 256))
        
        fp = build_firepred_provider(data_dir, spatial_size=(img_size, img_size))
        st = build_static_provider(data_dir, spatial_size=(img_size, img_size))
        
        ds = TSSatFireForecastDataset(
            data_dir=data_dir,
            T_in=int(fc.get("T_in", 10)),
            horizon=max(int(fc.get("horizon", 20)), T),
            label_channel=int(dc.get("label_channel", 1)),
            firepred_provider=fp,
            static_provider=st,
        )
        
        if ds.firepred_provider is None and getattr(fp, "_N", 0) > 0:
            ds.firepred_provider = fp
        if ds.static_provider is None and getattr(st, "_N", 0) > 0:
            ds.static_provider = st
        
        print(f"📊 Dataset size: {len(ds)} events")
        
        sequences = []
        for i in range(len(ds)):
            try:
                sample = ds[i]
                af_fut = sample["af_fut"]  # (Horizon, 1, H, W)
                
                if isinstance(af_fut, torch.Tensor):
                    af_fut = af_fut.numpy()
                
                af_fut = af_fut.squeeze(1)  # (Horizon, H, W)
                af_fut = np.clip(af_fut, 0, 1)
                
                # 选择均匀间隔的T帧
                total_H = af_fut.shape[0]
                if total_H >= T:
                    indices = np.linspace(0, total_H - 1, T, dtype=int)
                    gt = af_fut[indices]
                else:
                    gt = af_fut[:T]
                
                # 基本检查
                total_fire = (gt >= 0.5).sum()
                if total_fire < 100:
                    continue
                
                sequences.append((i, gt))
                
            except Exception as e:
                continue
        
        print(f"✅ 从forecast dataset加载了 {len(sequences)} 个序列\n")
        return sequences
        
    except Exception as e:
        print(f"⚠️  Forecast dataset加载失败: {e}")
        print("降级到基础label文件...\n")
        
        # 降级方案：使用原始label文件
        data_dir = Path(data_dir)
        label_file = None
        for name in ["af_val_label_seqtoseq_alll_1i_1.npy",
                     "af_train_label_seqtoseq_alll_1i_1.npy"]:
            lf = data_dir / name
            if lf.exists():
                label_file = lf
                break
        
        if label_file is None:
            raise FileNotFoundError(f"No label file in {data_dir}")
        
        labels = np.load(str(label_file), mmap_mode="r")
        print(f"📊 Labels shape: {labels.shape}")
        
        sequences = []
        label_channel = 1
        
        for i in range(labels.shape[0]):
            if labels.ndim == 5:
                gt = np.array(labels[i, label_channel], dtype=np.float32)
            elif labels.ndim == 4:
                gt = np.array(labels[i, label_channel], dtype=np.float32)[None]
            else:
                continue
            
            gt = np.clip(gt, 0, 1)
            total_fire = (gt >= 0.5).sum()
            if total_fire < 100:
                continue
            
            if gt.shape[0] < T:
                pad = T - gt.shape[0]
                gt = np.concatenate([gt] + [gt[-1:]] * pad, axis=0)
            
            sequences.append((i, gt[:T]))
        
        print(f"✅ 加载了 {len(sequences)} 个序列\n")
        return sequences


def compute_temporal_dynamics(gt: np.ndarray, threshold: float = 0.5) -> dict:
    """
    计算时间动态特征
    
    Returns
    -------
    dict with keys:
        - fire_areas: 每帧火灾面积
        - growth_rate: 增长率 (max - min) / min
        - variance: 面积方差
        - is_dynamic: 是否有明显时间变化
    """
    T = gt.shape[0]
    fire_areas = [(gt[t] >= threshold).sum() for t in range(T)]
    
    min_area = min(fire_areas)
    max_area = max(fire_areas)
    mean_area = np.mean(fire_areas)
    
    # 增长率
    growth_rate = (max_area - min_area) / (min_area + 1e-6)
    
    # 归一化方差 (coefficient of variation)
    variance = np.std(fire_areas) / (mean_area + 1e-6)
    
    # 判断是否动态：增长率>5% 或 方差>2%（降低阈值以适应真实数据）
    is_dynamic = (growth_rate > 0.05) or (variance > 0.02)
    
    return {
        'fire_areas': fire_areas,
        'growth_rate': growth_rate,
        'variance': variance,
        'mean_area': mean_area,
        'is_dynamic': is_dynamic,
    }


def compute_spatial_features(gt: np.ndarray, threshold: float = 0.5) -> dict:
    """
    计算空间特征用于多样性检测
    
    Returns
    -------
    dict with keys:
        - centroid: (y, x) 重心位置
        - bbox: (y0, y1, x0, x1) 边界框
        - aspect_ratio: 长宽比
        - compactness: 紧凑度
    """
    # 使用最后一帧（最大火灾范围）
    mask = (gt[-1] >= threshold)
    
    if mask.sum() == 0:
        return None
    
    ys, xs = np.where(mask)
    
    # 重心
    centroid = (ys.mean(), xs.mean())
    
    # 边界框
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    
    # 长宽比
    h, w = y1 - y0 + 1, x1 - x0 + 1
    aspect_ratio = max(h, w) / (min(h, w) + 1e-6)
    
    # 紧凑度 (面积 / 边界框面积)
    compactness = mask.sum() / (h * w + 1e-6)
    
    return {
        'centroid': centroid,
        'bbox': (y0, y1, x0, x1),
        'aspect_ratio': aspect_ratio,
        'compactness': compactness,
        'height': h,
        'width': w,
    }


def compute_diversity_score(features1: dict, features2: dict, img_size: int = 256) -> float:
    """
    计算两个事件之间的多样性分数（越大越不同）
    """
    if features1 is None or features2 is None:
        return 0.0
    
    # 重心距离（归一化到0-1）
    cy1, cx1 = features1['centroid']
    cy2, cx2 = features2['centroid']
    centroid_dist = np.sqrt((cy1 - cy2)**2 + (cx1 - cx2)**2) / (img_size * np.sqrt(2))
    
    # 长宽比差异
    ar_diff = abs(features1['aspect_ratio'] - features2['aspect_ratio']) / 10.0
    
    # 尺寸差异
    size1 = features1['height'] * features1['width']
    size2 = features2['height'] * features2['width']
    size_diff = abs(size1 - size2) / max(size1, size2)
    
    # 紧凑度差异
    comp_diff = abs(features1['compactness'] - features2['compactness'])
    
    # 综合多样性分数
    diversity = (
        centroid_dist * 0.4 +   # 位置差异最重要
        ar_diff * 0.25 +         # 形状差异
        size_diff * 0.20 +       # 尺寸差异
        comp_diff * 0.15         # 紧凑度差异
    )
    
    return min(diversity, 1.0)


def select_diverse_dynamic_events(
    sequences: List[Tuple[int, np.ndarray]],
    n_events: int = 20,
    min_diversity: float = 0.15,
) -> List[Tuple[int, np.ndarray, dict, dict]]:
    """
    选择时间动态明显且空间多样的事件
    
    Returns
    -------
    List of (event_idx, gt_seq, temporal_features, spatial_features)
    """
    print(f"🔍 分析事件时间动态性和空间特征...\n")
    
    # 1️⃣ 计算所有事件的特征
    candidates = []
    for idx, gt in tqdm(sequences, desc="特征提取"):
        temp_feat = compute_temporal_dynamics(gt)
        spat_feat = compute_spatial_features(gt)
        
        if not temp_feat['is_dynamic'] or spat_feat is None:
            continue
        
        candidates.append((idx, gt, temp_feat, spat_feat))
    
    print(f"✅ 找到 {len(candidates)} 个动态事件\n")
    
    if len(candidates) == 0:
        raise ValueError("没有找到任何动态事件！")
    
    # 2️⃣ 按增长率排序，优先选择变化明显的
    candidates.sort(key=lambda x: x[2]['growth_rate'], reverse=True)
    
    # 3️⃣ 贪心选择多样化事件
    selected = []
    selected.append(candidates[0])  # 选择变化最大的作为第一个
    
    print(f"📌 Event {candidates[0][0]:04d}: 增长率={candidates[0][2]['growth_rate']:.2%}, "
          f"方差={candidates[0][2]['variance']:.3f}")
    
    for idx, gt, temp_f, spat_f in candidates[1:]:
        if len(selected) >= n_events:
            break
        
        # 计算与已选事件的最小多样性
        min_div = min(
            compute_diversity_score(spat_f, s[3]) 
            for s in selected
        )
        
        # 如果足够多样，添加
        if min_div >= min_diversity:
            selected.append((idx, gt, temp_f, spat_f))
            print(f"📌 Event {idx:04d}: 增长率={temp_f['growth_rate']:.2%}, "
                  f"方差={temp_f['variance']:.3f}, 多样性={min_div:.3f}")
    
    print(f"\n✅ 最终选择了 {len(selected)} 个多样化动态事件\n")
    return selected


# ═════════════════════════════════════════════════════════════════════════════
# GPU加速批量生成
# ═════════════════════════════════════════════════════════════════════════════

def batch_generate(
    n_samples: int = 20,
    quality_levels: int = 10,
    fragmentation_levels: int = 5,
    data_dir: str = "dataset_val",
    out_dir: str = "figures/batch_illustration",
    horizon: int = 5,
    dpi: int = 150,
    mode: str = "presentation",
    seed: int = 42,
    min_diversity: float = 0.15,
):
    """改进的批量生成"""
    
    print("="*80)
    print("🚀 改进的批量可视化生成")
    print("="*80)
    print(f"目标: {n_samples} 事件 × {quality_levels} 质量 × {fragmentation_levels} 碎片化")
    print(f"     = {n_samples * quality_levels * fragmentation_levels} 张图片\n")
    
    # 1️⃣ 加载并筛选事件
    sequences = load_all_sequences(data_dir, T=horizon)
    selected_events = select_diverse_dynamic_events(
        sequences, n_events=n_samples, min_diversity=min_diversity
    )
    
    if len(selected_events) < n_samples:
        print(f"⚠️  只找到 {len(selected_events)} 个符合条件的事件（目标{n_samples}）")
        n_samples = len(selected_events)
    
    # 2️⃣ 定义质量和碎片化网格
    quality_grid = np.linspace(0.1, 1.0, quality_levels)
    fragmentation_grid = np.linspace(0.0, 1.0, fragmentation_levels)
    
    print(f"🎯 质量维度: {quality_grid}")
    print(f"🔪 碎片化维度: {fragmentation_grid}\n")
    
    # 3️⃣ 创建输出目录
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # 4️⃣ 批量生成
    total_images = len(selected_events) * len(quality_grid) * len(fragmentation_grid)
    results = []
    
    print(f"🎨 开始生成...\n")
    with tqdm(total=total_images, desc="总进度") as pbar:
        for event_rank, (event_idx, gt_seq, temp_f, spat_f) in enumerate(selected_events):
            event_dir = out_path / f"event_{event_idx:04d}"
            event_dir.mkdir(exist_ok=True)
            
            for q_idx, quality in enumerate(quality_grid):
                for f_idx, fragmentation in enumerate(fragmentation_grid):
                    filename = f"event_{event_idx:04d}_q{quality:.2f}_f{fragmentation:.2f}.png"
                    save_path = event_dir / filename
                    
                    try:
                        fig, pred, ious = build_figure(
                            gt_seq,
                            seed=seed + event_idx * 1000 + q_idx * 10 + f_idx,
                            quality=quality,
                            smoothness=2.0,
                            noise_level=0.1,
                            fragmentation=fragmentation,
                            mode=mode,
                            save_path=str(save_path),
                            dpi=dpi,
                            show=False,
                            auto_crop=True,
                        )
                        
                        results.append({
                            'event_idx': event_idx,
                            'event_rank': event_rank,
                            'quality': quality,
                            'fragmentation': fragmentation,
                            'mean_iou': np.mean(ious),
                            'ious': ious,
                            'growth_rate': temp_f['growth_rate'],
                            'variance': temp_f['variance'],
                            'filename': str(save_path.relative_to(out_path)),
                        })
                        
                    except Exception as e:
                        print(f"\n❌ 错误 [{filename}]: {e}")
                    
                    pbar.update(1)
    
    # 5️⃣ 保存结果摘要
    print("\n📊 保存结果摘要...")
    
    summary_file = out_path / "batch_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("改进的批量生成结果摘要\n")
        f.write("="*80 + "\n\n")
        f.write(f"配置:\n")
        f.write(f"  实际事件数: {len(selected_events)}\n")
        f.write(f"  质量维度: {quality_levels} (范围 {quality_grid[0]:.2f}-{quality_grid[-1]:.2f})\n")
        f.write(f"  碎片化维度: {fragmentation_levels} (范围 {fragmentation_grid[0]:.2f}-{fragmentation_grid[-1]:.2f})\n")
        f.write(f"  总图片数: {len(results)}\n")
        f.write(f"  模式: {mode}\n")
        f.write(f"  DPI: {dpi}\n")
        f.write(f"  最小多样性阈值: {min_diversity:.3f}\n\n")
        
        f.write("事件统计:\n")
        for rank, (ev_idx, gt, temp_f, spat_f) in enumerate(selected_events):
            event_results = [r for r in results if r['event_idx'] == ev_idx]
            if event_results:
                avg_iou = np.mean([r['mean_iou'] for r in event_results])
                f.write(
                    f"  Event {ev_idx:04d} (rank {rank+1}): "
                    f"{len(event_results)} 张, 平均IoU={avg_iou:.3f}, "
                    f"增长率={temp_f['growth_rate']:.2%}, "
                    f"方差={temp_f['variance']:.3f}\n"
                )
        
        f.write(f"\n按质量分组:\n")
        for q in quality_grid:
            q_results = [r for r in results if abs(r['quality'] - q) < 0.01]
            if q_results:
                avg_iou = np.mean([r['mean_iou'] for r in q_results])
                f.write(f"  质量 {q:.2f}: {len(q_results)} 张, 平均IoU={avg_iou:.3f}\n")
        
        f.write(f"\n按碎片化分组:\n")
        for frag in fragmentation_grid:
            f_results = [r for r in results if abs(r['fragmentation'] - frag) < 0.01]
            if f_results:
                avg_iou = np.mean([r['mean_iou'] for r in f_results])
                f.write(f"  碎片化 {frag:.2f}: {len(f_results)} 张, 平均IoU={avg_iou:.3f}\n")
    
    # CSV详细数据
    csv_file = out_path / "batch_results.csv"
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write("event_idx,event_rank,quality,fragmentation,mean_iou,iou_t0,iou_t1,iou_t2,iou_t3,iou_t4,"
                "growth_rate,variance,filename\n")
        for r in results:
            ious_str = ','.join([f"{x:.4f}" for x in r['ious']])
            f.write(
                f"{r['event_idx']},{r['event_rank']},{r['quality']:.2f},"
                f"{r['fragmentation']:.2f},{r['mean_iou']:.4f},"
                f"{ious_str},{r['growth_rate']:.4f},{r['variance']:.4f},"
                f"{r['filename']}\n"
            )
    
    print(f"\n✅ 全部完成!")
    print(f"📁 输出目录: {out_path.resolve()}")
    print(f"📄 摘要: {summary_file.name}")
    print(f"📊 数据: {csv_file.name}")
    print(f"🎨 总计: {len(results)} 张图片\n")
    
    return results, selected_events


if __name__ == "__main__":
    import argparse
    import os
    
    # 设置环境变量
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    parser = argparse.ArgumentParser(description="改进的批量生成")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--quality-levels", type=int, default=10)
    parser.add_argument("--frag-levels", type=int, default=5)
    parser.add_argument("--data-dir", type=str, default="dataset_val")
    parser.add_argument("--out-dir", type=str, default="figures/batch_illustration")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--mode", choices=["presentation", "realistic"], default="presentation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-diversity", type=float, default=0.15,
                       help="最小多样性阈值 (0.0-1.0)")
    
    args = parser.parse_args()
    
    results, events = batch_generate(
        n_samples=args.samples,
        quality_levels=args.quality_levels,
        fragmentation_levels=args.frag_levels,
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        dpi=args.dpi,
        mode=args.mode,
        seed=args.seed,
        min_diversity=args.min_diversity,
    )
