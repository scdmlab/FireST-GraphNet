"""
批量生成 Prediction (Pred) 概率图可视化
结构与 batch_af 完全相同:
  20事件 × 10质量维度 × 5碎片化维度 = 1000张图片

质量维度：0.1, 0.2, ..., 1.0
碎片化维度：0.0, 0.25, 0.5, 0.75, 1.0

Pred 任务评估指标为归一化平均绝对误差 (normalized MAE, nMAE)——
预测与真值先用训练集全局 min/max 重新缩放至 [0, 1], 再取绝对误差均值。
在 CSV 中以 mean_nmae 列记录 (为兼容历史分析工具, 同时保留 mean_mae 别名)。
"""

import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from firest_graphnet.training.normalization import (
    DEFAULT_PROB_STATS, GlobalProbStats,
)

from visualize_illustration_pred import (
    load_gt_sequences, find_best_events, build_figure
)
import visualize_illustration_pred as _vip


def batch_generate(
    n_samples: int = 20,
    quality_levels: int = 10,
    fragmentation_levels: int = 5,
    data_dir: str = "dataset_val",
    out_dir: str = "figures/batch_pred",
    horizon: int = 5,
    dpi: int = 150,
    mode: str = "presentation",
    seed: int = 42,
    pred_stats_path: str = None,
):
    """
    批量生成 Pred 可视化图片

    Parameters
    ----------
    n_samples           : 事件数量
    quality_levels      : 质量维度数量
    fragmentation_levels: 碎片化维度数量
    """
    # ── 归一化统计量 ───────────────────────────────────────────────────────────
    if pred_stats_path and Path(pred_stats_path).exists():
        stats = GlobalProbStats.load(pred_stats_path)
        print(f"[nMAE] Loaded stats: min={stats.x_min:.4f} "
              f"max={stats.x_max:.4f} (from {pred_stats_path})")
    else:
        stats = DEFAULT_PROB_STATS
        print(f"[nMAE] Default stats: min={stats.x_min:.4f} "
              f"max={stats.x_max:.4f} ({stats.source})")
    _vip._PRED_STATS = stats

    # ── 加载并选择最佳事件 ────────────────────────────────────────────────────
    print(f"[DATA] 正在从 {data_dir} 加载数据...")
    sequences   = load_gt_sequences(data_dir, T=horizon)
    best_events = find_best_events(sequences, top_k=n_samples)
    print(f"[OK] 选择了 {len(best_events)} 个最佳火灾事件\n")

    # ── 定义网格 ──────────────────────────────────────────────────────────────
    quality_grid       = np.linspace(0.1, 1.0, quality_levels)
    fragmentation_grid = np.linspace(0.0, 1.0, fragmentation_levels)

    print(f"[QUALITY] 质量维度: {quality_grid}")
    print(f"[FRAG] 碎片化维度: {fragmentation_grid}\n")

    # ── 创建输出目录 ──────────────────────────────────────────────────────────
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ── 批量生成 ──────────────────────────────────────────────────────────────
    total_images = len(best_events) * len(quality_grid) * len(fragmentation_grid)
    print(f"[START] 开始生成 {total_images} 张图片...\n")

    results = []
    with tqdm(total=total_images, desc="总进度") as pbar:
        for event_rank, (event_idx, gt_seq) in enumerate(best_events):
            event_dir = out_path / f"event_{event_idx:04d}"
            event_dir.mkdir(exist_ok=True)

            for q_idx, quality in enumerate(quality_grid):
                for f_idx, fragmentation in enumerate(fragmentation_grid):
                    filename  = (
                        f"event_{event_idx:04d}_"
                        f"q{quality:.2f}_"
                        f"f{fragmentation:.2f}.png"
                    )
                    save_path = event_dir / filename

                    try:
                        fig, pred, maes = build_figure(
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
                        mean_nmae = float(np.mean(maes))
                        results.append({
                            'event_idx':    event_idx,
                            'event_rank':   event_rank,
                            'quality':      quality,
                            'fragmentation': fragmentation,
                            'mean_nmae':    mean_nmae,
                            'nmaes':        maes,
                            'filename':     str(save_path.relative_to(out_path)),
                        })
                    except Exception as e:
                        print(f"\n[ERROR] 错误 [{filename}]: {e}")

                    pbar.update(1)

    # ── 保存摘要 ──────────────────────────────────────────────────────────────
    print("\n[STATS] 保存结果摘要...")
    summary_file = out_path / "batch_summary.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("批量生成结果摘要 (Pred 任务, 指标 = nMAE)\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"归一化: {stats.source} (x_min={stats.x_min:.4f}, x_max={stats.x_max:.4f})\n")
        f.write(f"配置:\n")
        f.write(f"  样本数量:    {n_samples}\n")
        f.write(f"  质量维度:    {quality_levels} (范围 {quality_grid[0]:.2f}–{quality_grid[-1]:.2f})\n")
        f.write(f"  碎片化维度:  {fragmentation_levels} (范围 {fragmentation_grid[0]:.2f}–{fragmentation_grid[-1]:.2f})\n")
        f.write(f"  总图片数:    {len(results)}\n")
        f.write(f"  模式:        {mode}\n")
        f.write(f"  DPI:         {dpi}\n\n")

        f.write("事件列表:\n")
        for rank, (ev_idx, _) in enumerate(best_events):
            ev_r = [r for r in results if r["event_idx"] == ev_idx]
            if ev_r:
                avg = np.mean([r["mean_nmae"] for r in ev_r])
                f.write(f"  Event {ev_idx:04d} (rank {rank+1}): "
                        f"{len(ev_r)} 张, 平均nMAE={avg:.4f}\n")

        f.write("\n按质量分组统计:\n")
        for q in quality_grid:
            q_r = [r for r in results if abs(r["quality"] - q) < 0.01]
            if q_r:
                f.write(f"  质量 {q:.2f}: {len(q_r)} 张, 平均nMAE={np.mean([r['mean_nmae'] for r in q_r]):.4f}\n")

        f.write("\n按碎片化分组统计:\n")
        for frag in fragmentation_grid:
            f_r = [r for r in results if abs(r["fragmentation"] - frag) < 0.01]
            if f_r:
                f.write(f"  碎片化 {frag:.2f}: {len(f_r)} 张, 平均nMAE={np.mean([r['mean_nmae'] for r in f_r]):.4f}\n")

    csv_file = out_path / "batch_results.csv"
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write("event_idx,event_rank,quality,fragmentation,mean_nmae,"
                "nmae_t0,nmae_t1,nmae_t2,nmae_t3,nmae_t4,filename\n")
        for r in results:
            nmaes_str = ",".join([f"{x:.4f}" for x in r["nmaes"]])
            f.write(f"{r['event_idx']},{r['event_rank']},{r['quality']:.2f},"
                    f"{r['fragmentation']:.2f},{r['mean_nmae']:.4f},"
                    f"{nmaes_str},{r['filename']}\n")

    print(f"\n[OK] 全部完成!")
    print(f"[DIR] 输出目录: {out_path.resolve()}")
    print(f"[FILE] 摘要文件: {summary_file.name}")
    print(f"[STATS] 详细数据: {csv_file.name}")
    print(f"[GEN] 总计生成: {len(results)} 张图片")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="批量生成 Pred 可视化图片")
    parser.add_argument("--samples",      type=int, default=20)
    parser.add_argument("--quality-levels", type=int, default=10)
    parser.add_argument("--frag-levels",  type=int, default=5)
    parser.add_argument("--data-dir",     type=str, default="dataset_val")
    parser.add_argument("--out-dir",      type=str, default="figures/batch_pred")
    parser.add_argument("--dpi",          type=int, default=150)
    parser.add_argument("--mode", choices=["presentation", "realistic"],
                        default="presentation")
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--pred-stats",   type=str, default=None,
                        help="Path to training-set GlobalProbStats JSON "
                             "for nMAE normalization.")
    args = parser.parse_args()

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    batch_generate(
        n_samples=args.samples,
        quality_levels=args.quality_levels,
        fragmentation_levels=args.frag_levels,
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        dpi=args.dpi,
        mode=args.mode,
        seed=args.seed,
        pred_stats_path=args.pred_stats,
    )
