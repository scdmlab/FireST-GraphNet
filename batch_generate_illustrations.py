"""
批量生成野火预测可视化图片
20个样本 × 10个质量维度 × 5个碎片化维度 = 1000张图片

质量维度：0.1, 0.2, 0.3, ..., 1.0
碎片化维度：0.0, 0.25, 0.5, 0.75, 1.0
"""

import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

# 导入主脚本的函数
from visualize_illustration import (
    load_gt_sequences, find_best_events, build_figure
)


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
):
    """
    批量生成可视化图片
    
    Parameters
    ----------
    n_samples : 样本数量（从最佳事件中选择）
    quality_levels : 质量维度数量
    fragmentation_levels : 碎片化维度数量
    """
    
    # 1️⃣ 加载并选择最佳事件
    print(f"📂 正在从 {data_dir} 加载数据...")
    sequences = load_gt_sequences(data_dir, T=horizon)
    best_events = find_best_events(sequences, top_k=n_samples)
    print(f"✅ 选择了 {len(best_events)} 个最佳火灾事件\n")
    
    # 2️⃣ 定义质量和碎片化网格
    quality_grid = np.linspace(0.1, 1.0, quality_levels)
    fragmentation_grid = np.linspace(0.0, 1.0, fragmentation_levels)
    
    print(f"🎯 质量维度: {quality_grid}")
    print(f"🔪 碎片化维度: {fragmentation_grid}\n")
    
    # 3️⃣ 创建输出目录
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # 4️⃣ 批量生成
    total_images = len(best_events) * len(quality_grid) * len(fragmentation_grid)
    print(f"🚀 开始生成 {total_images} 张图片...\n")
    
    # 用于跟踪进度
    results = []
    
    with tqdm(total=total_images, desc="总进度") as pbar:
        for event_rank, (event_idx, gt_seq) in enumerate(best_events):
            # 为每个事件创建子目录
            event_dir = out_path / f"event_{event_idx:04d}"
            event_dir.mkdir(exist_ok=True)
            
            for q_idx, quality in enumerate(quality_grid):
                for f_idx, fragmentation in enumerate(fragmentation_grid):
                    # 文件命名：event_XXXX_q0.70_f0.50.png
                    filename = (
                        f"event_{event_idx:04d}_"
                        f"q{quality:.2f}_"
                        f"f{fragmentation:.2f}.png"
                    )
                    save_path = event_dir / filename
                    
                    try:
                        # 生成图片
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
                        
                        mean_iou = np.mean(ious)
                        
                        results.append({
                            'event_idx': event_idx,
                            'event_rank': event_rank,
                            'quality': quality,
                            'fragmentation': fragmentation,
                            'mean_iou': mean_iou,
                            'ious': ious,
                            'filename': str(save_path.relative_to(out_path)),
                        })
                        
                    except Exception as e:
                        print(f"\n❌ 错误 [{filename}]: {e}")
                    
                    pbar.update(1)
    
    # 5️⃣ 保存结果摘要
    print("\n📊 保存结果摘要...")
    summary_file = out_path / "batch_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"批量生成结果摘要\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"配置:\n")
        f.write(f"  样本数量: {n_samples}\n")
        f.write(f"  质量维度: {quality_levels} (范围 {quality_grid[0]:.2f}-{quality_grid[-1]:.2f})\n")
        f.write(f"  碎片化维度: {fragmentation_levels} (范围 {fragmentation_grid[0]:.2f}-{fragmentation_grid[-1]:.2f})\n")
        f.write(f"  总图片数: {len(results)}\n")
        f.write(f"  模式: {mode}\n")
        f.write(f"  DPI: {dpi}\n\n")
        
        f.write(f"事件列表:\n")
        for rank, (ev_idx, _) in enumerate(best_events):
            event_results = [r for r in results if r['event_idx'] == ev_idx]
            if event_results:
                avg_iou = np.mean([r['mean_iou'] for r in event_results])
                f.write(f"  Event {ev_idx:04d} (rank {rank+1}): "
                       f"{len(event_results)} 张图片, 平均IoU={avg_iou:.3f}\n")
        
        f.write(f"\n按质量分组统计:\n")
        for q in quality_grid:
            q_results = [r for r in results if abs(r['quality'] - q) < 0.01]
            if q_results:
                avg_iou = np.mean([r['mean_iou'] for r in q_results])
                f.write(f"  质量 {q:.2f}: {len(q_results)} 张, 平均IoU={avg_iou:.3f}\n")
        
        f.write(f"\n按碎片化分组统计:\n")
        for frag in fragmentation_grid:
            f_results = [r for r in results if abs(r['fragmentation'] - frag) < 0.01]
            if f_results:
                avg_iou = np.mean([r['mean_iou'] for r in f_results])
                f.write(f"  碎片化 {frag:.2f}: {len(f_results)} 张, 平均IoU={avg_iou:.3f}\n")
    
    # 保存详细CSV
    csv_file = out_path / "batch_results.csv"
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write("event_idx,event_rank,quality,fragmentation,mean_iou,iou_t0,iou_t1,iou_t2,iou_t3,iou_t4,filename\n")
        for r in results:
            ious_str = ','.join([f"{x:.4f}" for x in r['ious']])
            f.write(f"{r['event_idx']},{r['event_rank']},{r['quality']:.2f},"
                   f"{r['fragmentation']:.2f},{r['mean_iou']:.4f},"
                   f"{ious_str},{r['filename']}\n")
    
    print(f"\n✅ 全部完成!")
    print(f"📁 输出目录: {out_path.resolve()}")
    print(f"📄 摘要文件: {summary_file.name}")
    print(f"📊 详细数据: {csv_file.name}")
    print(f"🎨 总计生成: {len(results)} 张图片")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="批量生成野火预测可视化")
    parser.add_argument("--samples", type=int, default=20,
                       help="样本数量 (默认: 20)")
    parser.add_argument("--quality-levels", type=int, default=10,
                       help="质量维度数量 (默认: 10)")
    parser.add_argument("--frag-levels", type=int, default=5,
                       help="碎片化维度数量 (默认: 5)")
    parser.add_argument("--data-dir", type=str, default="dataset_val",
                       help="数据目录")
    parser.add_argument("--out-dir", type=str, default="figures/batch_illustration",
                       help="输出目录")
    parser.add_argument("--dpi", type=int, default=150,
                       help="图片DPI (默认: 150)")
    parser.add_argument("--mode", choices=["presentation", "realistic"],
                       default="presentation", help="生成模式")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    
    args = parser.parse_args()
    
    # 设置环境变量避免OpenMP冲突
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    results = batch_generate(
        n_samples=args.samples,
        quality_levels=args.quality_levels,
        fragmentation_levels=args.frag_levels,
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        dpi=args.dpi,
        mode=args.mode,
        seed=args.seed,
    )
