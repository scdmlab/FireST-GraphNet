# ─── EMA (delayed-init) ──────────────────────────────────────────────────
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = None
        self._initialised = False

    def maybe_init(self, model, epoch, start_epoch):
        if (not self._initialised) and epoch >= start_epoch:
            self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
            self._initialised = True
            print(f'  [EMA] shadow initialised at epoch {epoch}')

    def update(self, model):
        if not self._initialised:
            return
        d = self.decay
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    self.shadow[k].mul_(d).add_(v.detach(), alpha=1 - d)
                else:
                    self.shadow[k].copy_(v)

    def swap_in(self, model):
        if not self._initialised:
            return None
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=False)
        return backup

    def swap_out(self, model, backup):
        if backup is not None:
            model.load_state_dict(backup, strict=False)


# ─── Composite single-step loss ──────────────────────────────────────────
class AFLoss(nn.Module):
    def __init__(self, pos_weight=3.0, focal_gamma=2.0, lovasz_w=0.5,
                 dice_w=0.15, tversky_w=0.3, tversky_a=0.4, tversky_b=0.6,
                 boundary_w=0.2, label_smoothing=0.02,
                 hard_mining=False, hard_top_k_frac=0.75):
        super().__init__()
        self.focal = FocalBCELoss(pos_weight=pos_weight, gamma=focal_gamma,
                                  label_smoothing=label_smoothing)
        self.tversky = TverskyLoss(alpha=tversky_a, beta=tversky_b)
        self.boundary = BoundaryAwareLoss()
        self.lovasz = LovaszHingeLoss(per_image=True)
        self.lovasz_w, self.dice_w, self.tversky_w, self.boundary_w = \
            lovasz_w, dice_w, tversky_w, boundary_w
        self.hard_mining = hard_mining
        self.hard_top_k_frac = hard_top_k_frac

    def _soft_dice(self, logits, tgt):
        p = torch.sigmoid(logits).reshape(logits.shape[0], -1)
        t = tgt.reshape(tgt.shape[0], -1)
        inter = (p * t).sum(dim=1)
        return (1.0 - (2.0 * inter + 1.0) / (p.sum(dim=1) + t.sum(dim=1) + 1.0)).mean()

    def _hard_focal(self, logits, tgt):
        probs = torch.sigmoid(logits)
        p_t = probs * tgt + (1.0 - probs) * (1.0 - tgt)
        focal_w = (1.0 - p_t).pow(2.0)
        pw = 3.0
        px_w = tgt * pw + (1.0 - tgt)
        bce = F.binary_cross_entropy_with_logits(logits, tgt, reduction='none')
        loss_map = focal_w * px_w * bce
        B = loss_map.shape[0]
        flat = loss_map.view(B, -1)
        k = max(1, int(flat.shape[1] * self.hard_top_k_frac))
        top, _ = flat.topk(k, dim=1, largest=True)
        return top.mean()

    def forward(self, logits, tgt):
        loss_focal = self._hard_focal(logits, tgt) if self.hard_mining else self.focal(logits, tgt)
        loss_lovasz = self.lovasz(logits, tgt)
        loss_dice = self._soft_dice(logits, tgt)
        loss_tversky = self.tversky(logits, tgt)
        loss_boundary = self.boundary(logits, tgt)
        total = (loss_focal
                 + self.lovasz_w   * loss_lovasz
                 + self.dice_w     * loss_dice
                 + self.tversky_w  * loss_tversky
                 + self.boundary_w * loss_boundary)
        return dict(total=total,
                    focal=loss_focal.detach(),
                    lovasz=loss_lovasz.detach(),
                    dice=loss_dice.detach(),
                    tversky=loss_tversky.detach(),
                    boundary=loss_boundary.detach())


# ─── Checkpoint helpers ───────────────────────────────────────────────────
def find_latest_checkpoint(run_dir, resume_path=None):
    """Return path to best full-state checkpoint, or None if none found."""
    run_dir = Path(run_dir)
    if resume_path is not None:
        rp = Path(resume_path)
        if rp.exists():
            return rp
        print(f'  [resume] WARNING: RESUME_PATH {resume_path!r} not found — falling back to auto.')
    last = run_dir / 'last.pt'
    if last.exists():
        return last
    periodic = sorted(run_dir.glob('checkpoint_epoch*.pt'))
    if periodic:
        return periodic[-1]
    return None


def save_checkpoint(path, epoch, model, optimizer, scheduler, scaler,
                    ema, best_iou, patience_ctr, history,
                    val_iou_best_thr, val_thr_best, cfg):
    """Atomically save full training state (model/opt/sched/scaler/EMA/history)."""
    ema_shadow = None
    if ema is not None and ema._initialised:
        ema_shadow = {k: v.clone().cpu() for k, v in ema.shadow.items()}
    payload = {
        'epoch':            epoch,
        'model_state':      {k: v.cpu() for k, v in model.state_dict().items()},
        'optimizer_state':  optimizer.state_dict(),
        'scheduler_state':  scheduler.state_dict() if scheduler is not None else None,
        'scaler_state':     scaler.state_dict()    if scaler    is not None else None,
        'ema_state':        ema_shadow,
        'ema_initialised':  (ema._initialised if ema is not None else False),
        'best_iou':         best_iou,
        'patience_ctr':     patience_ctr,
        'history':          history,
        'cfg':              cfg,
        'val_iou_best_thr': val_iou_best_thr,
        'val_thr_best':     val_thr_best,
    }
    path = Path(path)
    tmp  = path.with_suffix('.tmp')
    torch.save(payload, tmp)
    tmp.replace(path)   # atomic rename — safe if Colab kills mid-write


def load_checkpoint(ckpt_path, model, optimizer, scheduler, scaler, ema, device):
    """Load full training state. Handles missing/mismatched keys gracefully."""
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except Exception as exc:
        print(f'  [resume] ERROR reading {ckpt_path}: {exc}')
        return None

    # Model weights — strict=False tolerates minor key mismatches
    try:
        res = model.load_state_dict(ckpt['model_state'], strict=False)
        if res.missing_keys:
            print(f'  [resume] model missing keys ({len(res.missing_keys)}): {res.missing_keys[:4]}')
        if res.unexpected_keys:
            print(f'  [resume] model unexpected keys ({len(res.unexpected_keys)}): {res.unexpected_keys[:4]}')
    except Exception as exc:
        print(f'  [resume] WARNING: model weights not loaded: {exc}')

    restored = {}

    def _try_load(obj, key, label):
        if obj is None or key not in ckpt or ckpt[key] is None:
            return
        try:
            obj.load_state_dict(ckpt[key])
            restored[label] = True
        except Exception as exc:
            print(f'  [resume] WARNING: {label} state load failed: {exc}')

    _try_load(optimizer, 'optimizer_state', 'optimizer')
    _try_load(scheduler, 'scheduler_state', 'scheduler')
    _try_load(scaler,    'scaler_state',    'scaler')

    # EMA — custom restore (not a nn.Module, no load_state_dict)
    if ema is not None and 'ema_state' in ckpt and ckpt['ema_state'] is not None:
        try:
            ema.shadow       = {k: v.to(device) for k, v in ckpt['ema_state'].items()}
            ema._initialised = bool(ckpt.get('ema_initialised', True))
            restored['ema']  = True
        except Exception as exc:
            print(f'  [resume] WARNING: EMA state load failed: {exc}')

    restored['epoch']            = int(ckpt.get('epoch', 0))
    restored['best_iou']         = float(ckpt.get('best_iou', -1.0))
    restored['patience_ctr']     = int(ckpt.get('patience_ctr', 0))
    restored['history']          = list(ckpt.get('history', []))
    restored['val_iou_best_thr'] = float(ckpt.get('val_iou_best_thr', -1.0))
    restored['val_thr_best']     = float(ckpt.get('val_thr_best', 0.40))
    return restored


# ─── Instantiate model, loss, optimizer, scheduler, scaler, EMA ──────────
mcfg, tcfg = cfg['model'], cfg['training']
train_loader, val_loader, train_base, val_base, C_aux, C_static = build_loaders(cfg)

model = FireAFNetV5(
    in_ch_img=int(cfg['data']['num_bands']),
    in_ch_aux=int(C_aux),
    in_ch_static=int(C_static),
    encoder_channels=tuple(mcfg['encoder_channels']),
    gnn_patches=tuple(mcfg['gnn_patches']),
    gnn_heads=int(mcfg['gnn_heads']),
    gnn_layers=int(mcfg['gnn_layers']),
    gnn_knn=int(mcfg['gnn_knn']),
    gnn_content_knn=bool(mcfg['gnn_content_knn']),
    lstm_hidden=int(mcfg['lstm_hidden']),
    lstm_layers=int(mcfg['lstm_layers']),
    lstm_dropout=float(mcfg['lstm_dropout']),
    lstm_attn_heads=int(mcfg['lstm_attn_heads']),
    weather_indices=tuple(mcfg['weather_indices']),
    weather_emb=int(mcfg['weather_emb']),
    cross_attn_heads=int(mcfg['cross_attn_heads']),
    decoder_dropout=float(mcfg['decoder_dropout']),
    use_film_skip=bool(mcfg['use_film_skip']),
    use_cbam=bool(mcfg['use_cbam']),
    use_input_norm=bool(mcfg['use_input_norm']),
    out_ch=int(mcfg['out_channels']),
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f'Model: FireAFNetV5  | params={n_params:.2f} M  | device={DEVICE}')

criterion = AFLoss(
    pos_weight=float(tcfg['pos_weight']),
    focal_gamma=float(tcfg['focal_gamma']),
    lovasz_w=float(tcfg['lovasz_weight']),
    dice_w=float(tcfg['dice_weight']),
    tversky_w=float(tcfg['tversky_weight']),
    tversky_a=float(tcfg['tversky_alpha']),
    tversky_b=float(tcfg['tversky_beta']),
    boundary_w=float(tcfg['boundary_weight']),
    label_smoothing=float(tcfg['label_smoothing']),
    hard_mining=False,
    hard_top_k_frac=float(tcfg['hard_example_top_k_frac']),
).to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=float(tcfg['lr']),
    weight_decay=float(tcfg['weight_decay']),
    betas=(float(tcfg['adam_beta1']), float(tcfg['adam_beta2'])),
)

EPOCHS = int(tcfg['epochs'])
W      = int(tcfg['warmup_epochs_lr'])

def lr_lambda(ep):
    if ep < W:
        return tcfg['warmup_start_factor'] + (1 - tcfg['warmup_start_factor']) * (ep / max(1, W))
    progress = (ep - W) / max(1, EPOCHS - W)
    return (tcfg['eta_min'] / tcfg['lr']
            + 0.5 * (1 - tcfg['eta_min'] / tcfg['lr'])
            * (1 + math.cos(math.pi * progress)))

scheduler  = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
amp_dtype  = torch.bfloat16 if tcfg.get('amp_dtype', 'bfloat16') == 'bfloat16' else torch.float16
use_scaler = (amp_dtype == torch.float16)
scaler     = torch.amp.GradScaler('cuda') if use_scaler else None
ema        = EMA(model, decay=float(tcfg['ema_decay']))


# ─── Metric helpers (global TP/FP/FN accumulator) ────────────────────────
def _run_validation(model, loader, criterion, thr_grid, ema=None):
    model.eval()
    bak = ema.swap_in(model) if ema is not None else None
    try:
        tp = {t: 0.0 for t in thr_grid}
        fp = {t: 0.0 for t in thr_grid}
        fn = {t: 0.0 for t in thr_grid}
        losses = dict(total=0.0, focal=0.0, lovasz=0.0, dice=0.0,
                      tversky=0.0, boundary=0.0)
        n_batches = 0
        with torch.no_grad():
            for batch in loader:
                for k in batch:
                    batch[k] = batch[k].to(DEVICE, non_blocking=True)
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    out  = model(batch)
                    loss = criterion(out['logits'], batch['y'])
                logits = out['logits'].float()
                gt     = (batch['y'] > 0.5).float()
                probs  = torch.sigmoid(logits)
                for t in thr_grid:
                    pred   = (probs > t).float()
                    tp[t] += float((pred * gt).sum())
                    fp[t] += float((pred * (1 - gt)).sum())
                    fn[t] += float(((1 - pred) * gt).sum())
                for k in losses:
                    losses[k] += float(loss[k])
                n_batches += 1
        best_iou_v, best_t, best_p, best_r = -1.0, thr_grid[0], 0.0, 0.0
        for t in thr_grid:
            iou_t = tp[t] / max(tp[t] + fp[t] + fn[t], 1e-8)
            if iou_t > best_iou_v:
                best_iou_v = iou_t; best_t = t
                best_p = tp[t] / max(tp[t] + fp[t], 1e-8)
                best_r = tp[t] / max(tp[t] + fn[t], 1e-8)
        losses = {k: v / max(n_batches, 1) for k, v in losses.items()}
        return dict(**losses, iou_best_thr=best_iou_v, thr_best=best_t,
                    prec_best=best_p, rec_best=best_r,
                    f1_best=2 * best_p * best_r / max(best_p + best_r, 1e-8))
    finally:
        if bak is not None:
            ema.swap_out(model, bak)


def _mixup(batch, alpha):
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)
    idx = torch.randperm(batch['af_hist'].shape[0], device=DEVICE)
    for k in ['af_hist', 'aux_hist', 'static', 'y']:
        batch[k] = lam * batch[k] + (1 - lam) * batch[k][idx]
    return batch


# ─── Fit-loop setup ──────────────────────────────────────────────────────
history      = []
best_iou     = -1.0
patience_ctr = 0
PATIENCE     = int(tcfg['patience'])
thr_grid     = list(tcfg['val_threshold_grid'])
mixup_p, mixup_a = float(tcfg['mixup_prob']), float(tcfg['mixup_alpha'])
grad_clip    = float(tcfg['grad_clip'])
out_dir      = Path(OUTPUT_LOCAL); out_dir.mkdir(parents=True, exist_ok=True)
hme_start    = int(tcfg['hard_example_start_epoch'])
start_epoch  = 1
_last_val_thr = 0.40

# ─── Auto-resume probe ───────────────────────────────────────────────────
_ckpt_path = find_latest_checkpoint(out_dir,
                                     RESUME_PATH if AUTO_RESUME else None)
if _ckpt_path is not None and AUTO_RESUME:
    print(f'\n{"="*60}')
    print(f'  RESUMING from: {_ckpt_path}')
    _res = load_checkpoint(_ckpt_path, model, optimizer, scheduler,
                           scaler, ema, device=str(DEVICE))
    if _res is not None:
        start_epoch   = _res['epoch'] + 1
        best_iou      = _res['best_iou']
        patience_ctr  = _res['patience_ctr']
        history       = _res['history']
        _last_val_thr = _res['val_thr_best']
        print(f'  Restored epoch   : {_res["epoch"]}  -> continuing from ep {start_epoch}')
        print(f'  Best val IoU     : {best_iou:.4f}  @thr={_last_val_thr:.2f}')
        print(f'  Patience counter : {patience_ctr}/{PATIENCE}')
        print(f'  History rows     : {len(history)}')
        print(f'  Optimizer        : {"OK" if _res.get("optimizer") else "not found in ckpt"}')
        print(f'  Scheduler        : {"OK" if _res.get("scheduler") else "not found/skipped"}')
        print(f'  Scaler (fp16)    : {"OK" if _res.get("scaler") else "not found/skipped"}')
        print(f'  EMA              : {"OK" if _res.get("ema") else "not found/skipped"}')
        print(f'{"="*60}')
    else:
        print('  [resume] Checkpoint corrupted — starting fresh from epoch 1.')
        start_epoch = 1
elif not AUTO_RESUME:
    print('AUTO_RESUME=False — starting fresh from epoch 1.')
else:
    print('No checkpoint found — starting fresh from epoch 1.')

# ─── CSV continuity ──────────────────────────────────────────────────────
# If resuming but metrics.csv was lost (Colab /content cleared), rebuild it.
import csv as _csv
_col_names = list(history[0].keys()) if history else None
if history and not (out_dir / 'metrics.csv').exists():
    with open(out_dir / 'metrics.csv', 'w', newline='') as _f:
        w = _csv.DictWriter(_f, fieldnames=_col_names)
        w.writeheader(); w.writerows(history)
    print(f'  [resume] Reconstructed metrics.csv from history ({len(history)} rows).')

# ─── Fit loop ────────────────────────────────────────────────────────────
for epoch in range(start_epoch, EPOCHS + 1):
    criterion.hard_mining = bool(tcfg['hard_example_mining']) and (epoch >= hme_start)
    ema.maybe_init(model, epoch, int(tcfg['ema_start_epoch']))
    model.train()
    run = dict(total=0.0, focal=0.0, lovasz=0.0, dice=0.0, tversky=0.0, boundary=0.0, n=0)

    pbar = tqdm(train_loader, desc=f'ep {epoch:02d}/{EPOCHS}', leave=False)
    for batch in pbar:
        for k in batch:
            batch[k] = batch[k].to(DEVICE, non_blocking=True)
        if torch.rand(1).item() < mixup_p:
            batch = _mixup(batch, mixup_a)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=amp_dtype):
            out  = model(batch)
            loss = criterion(out['logits'], batch['y'])

        if use_scaler:
            scaler.scale(loss['total']).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer); scaler.update()
        else:
            loss['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        ema.update(model)

        for k in run:
            if k == 'n': run[k] += 1
            else:         run[k] += float(loss[k])
        pbar.set_postfix(total=loss['total'].item(), lovasz=loss['lovasz'].item(),
                         lr=optimizer.param_groups[0]['lr'])

    tr = {k: v / max(run['n'], 1) for k, v in run.items() if k != 'n'}
    scheduler.step()

    # ── Validation (EMA weights if ready, else raw) ───────────────────────
    va = _run_validation(model, val_loader, criterion, thr_grid,
                         ema=ema if ema._initialised else None)

    row = dict(epoch=epoch, lr=optimizer.param_groups[0]['lr'],
               **{'tr_' + k: v for k, v in tr.items()},
               **{'va_' + k: v for k, v in va.items()})
    history.append(row)

    print(f'ep {epoch:02d}  '
          f'tr_total={tr["total"]:.4f}  '
          f'va_total={va["total"]:.4f}  '
          f'val_iou_best={va["iou_best_thr"]:.4f}@t={va["thr_best"]:.2f}  '
          f'P={va["prec_best"]:.3f}  R={va["rec_best"]:.3f}  '
          f'lr={optimizer.param_groups[0]["lr"]:.2e}')

    # ── Best-model checkpoint (inference weights — EMA preferred) ─────────
    if va['iou_best_thr'] > best_iou:
        best_iou = va['iou_best_thr']; patience_ctr = 0
        _last_val_thr = va['thr_best']
        torch.save({
            'model_state':      model.state_dict(),
            'ema_state':        (ema.shadow if ema._initialised else None),
            'cfg':              cfg,
            'epoch':            epoch,
            'val_iou_best_thr': best_iou,
            'val_thr_best':     _last_val_thr,
        }, out_dir / 'best_model.pt')
        print(f'  * new best iou={best_iou:.4f}  -> best_model.pt saved')
    else:
        patience_ctr += 1

    # ── Full training-state checkpoint (last.pt — always overwritten) ─────
    save_checkpoint(
        out_dir / 'last.pt',
        epoch=epoch, model=model, optimizer=optimizer,
        scheduler=scheduler, scaler=scaler, ema=ema,
        best_iou=best_iou, patience_ctr=patience_ctr, history=history,
        val_iou_best_thr=best_iou, val_thr_best=_last_val_thr, cfg=cfg,
    )

    # ── Periodic dated checkpoint (full state) ────────────────────────────
    if KEEP_PERIODIC_CHECKPOINTS and (epoch % CHECKPOINT_EVERY == 0):
        save_checkpoint(
            out_dir / f'checkpoint_epoch{epoch:03d}.pt',
            epoch=epoch, model=model, optimizer=optimizer,
            scheduler=scheduler, scaler=scaler, ema=ema,
            best_iou=best_iou, patience_ctr=patience_ctr, history=history,
            val_iou_best_thr=best_iou, val_thr_best=_last_val_thr, cfg=cfg,
        )

    # ── CSV log — append one row per epoch (header written once) ──────────
    if _col_names is None:
        _col_names = list(row.keys())
        with open(out_dir / 'metrics.csv', 'w', newline='') as _f:
            _csv.DictWriter(_f, fieldnames=_col_names).writeheader()
    with open(out_dir / 'metrics.csv', 'a', newline='') as _f:
        _csv.DictWriter(_f, fieldnames=_col_names).writerow(row)

    # ── Early stopping ────────────────────────────────────────────────────
    if patience_ctr >= PATIENCE:
        print(f'Early stopping at ep {epoch} — no improvement for {PATIENCE} epochs.')
        break

print(f'\nTraining done.  best val_iou_best_thr = {best_iou:.4f}')
BEST_CKPT = out_dir / 'best_model.pt'
print(f'Best checkpoint : {BEST_CKPT}')
print(f'Last checkpoint : {out_dir / "last.pt"}')
