"""Build colab_run29.ipynb by patching colab_run28.ipynb."""
import json, shutil

SRC = 'C:/Users/22125/OneDrive - UW-Madison/FireST-GraphNet-Clean/data_tools/colab/colab_run28.ipynb'
DST = 'C:/Users/22125/OneDrive - UW-Madison/FireST-GraphNet-Clean/data_tools/colab/colab_run29.ipynb'
shutil.copy2(SRC, DST)

with open(DST, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# ── CELL 05 — point CONFIG_PATH to run29 ───────────────────────────────────
src5 = ''.join(nb['cells'][5]['source'])
src5 = src5.replace('config_run28.yaml', 'config_run29.yaml')
nb['cells'][5]['source'] = [src5]
nb['cells'][5]['outputs'] = []
nb['cells'][5]['execution_count'] = None

# ── CELL 16 — replace ForecastDecoderV2 with V3, update main model ────────
src16 = ''.join(nb['cells'][16]['source'])

OLD_DEC_START = "class ForecastDecoderV2(nn.Module):"
OLD_DEC_END   = "# -------------------------------------------- FireSTGraphNetV2 (top) ----"
assert OLD_DEC_START in src16
assert OLD_DEC_END in src16

NEW_DEC = r'''class ForecastDecoderV3(nn.Module):
    # V3: per-horizon pos_emb + autoregressive feedback + FiLM-modulate skips.
    # Fixes V2 bug where all horizons produced near-identical outputs.
    def __init__(self, spatial_ch, lstm_hidden, horizon,
                 skip_channels=(32, 64),
                 weather_ch=4, weather_emb=64,
                 dropout=0.15, heads=4, aux_ba=True,
                 use_pos_emb=True, use_feedback=True, use_skip_film=True):
        super().__init__()
        self.H = horizon
        self.spatial_ch = spatial_ch
        self.use_pos_emb   = use_pos_emb
        self.use_feedback  = use_feedback
        self.use_skip_film = use_skip_film

        self.weather_embed = nn.Sequential(
            nn.Linear(weather_ch, 32), nn.GELU(),
            nn.Linear(32, weather_emb),
        )
        self.cross_attn = CrossAttnWeatherFusion(spatial_ch, weather_emb, heads=heads)

        self.lstm_cell = nn.LSTMCell(spatial_ch, lstm_hidden)
        self.proj_cond = nn.Linear(lstm_hidden, spatial_ch)

        if use_pos_emb:
            self.pos_emb = nn.Parameter(torch.randn(horizon, spatial_ch) * 0.02)

        if use_feedback:
            self.feedback_conv = nn.Conv2d(1, spatial_ch, kernel_size=3, padding=1)

        if use_skip_film:
            self.film1 = nn.Linear(lstm_hidden, 2 * skip_channels[1])  # for skip2
            self.film2 = nn.Linear(lstm_hidden, 2 * skip_channels[0])  # for skip1

        self.up1 = nn.ConvTranspose2d(spatial_ch, 64, 2, stride=2)
        self.fuse1 = nn.Sequential(
            nn.Conv2d(64 + skip_channels[1], 64, 3, padding=1),
            nn.InstanceNorm2d(64, affine=True), nn.GELU(),
            nn.Dropout2d(dropout),
        )
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.fuse2 = nn.Sequential(
            nn.Conv2d(32 + skip_channels[0], 32, 3, padding=1),
            nn.InstanceNorm2d(32, affine=True), nn.GELU(),
            nn.Dropout2d(dropout),
        )
        self.head_af = nn.Conv2d(32, 1, 1)
        self.aux_ba = aux_ba
        if aux_ba:
            self.head_ba = nn.Conv2d(32, 1, 1)

    def forward(self, spatial, lstm_state, env_fut, skip1, skip2):
        B, C, Hs, Ws = spatial.shape
        H_out = env_fut.shape[1]
        h, c = lstm_state

        af_seq, ba_seq = [], []
        ba_cum = torch.zeros(B, 1, skip1.shape[-2], skip1.shape[-1],
                             device=spatial.device, dtype=spatial.dtype)
        prev_af = torch.zeros(B, 1, skip1.shape[-2], skip1.shape[-1],
                              device=spatial.device, dtype=spatial.dtype)

        for t in range(H_out):
            sp_vec = F.adaptive_avg_pool2d(spatial, 1).flatten(1)
            h, c = self.lstm_cell(sp_vec, (h, c))
            cond = self.proj_cond(h)
            spatial_t = spatial + cond.view(B, C, 1, 1)

            if self.use_pos_emb:
                t_idx = min(t, self.pos_emb.shape[0] - 1)
                spatial_t = spatial_t + self.pos_emb[t_idx].view(1, C, 1, 1)

            if self.use_feedback:
                fb = F.adaptive_avg_pool2d(prev_af, (Hs, Ws))
                spatial_t = spatial_t + self.feedback_conv(fb)

            w_now = env_fut[:, t]
            w_vec = F.adaptive_avg_pool2d(w_now, 1).flatten(1)
            w_tok = self.weather_embed(w_vec).unsqueeze(1)
            sp_tok = spatial_t.flatten(2).transpose(1, 2)
            sp_tok = self.cross_attn(sp_tok, w_tok)
            spatial_t = sp_tok.transpose(1, 2).reshape(B, C, Hs, Ws)

            if self.use_skip_film:
                g1, b1 = self.film1(h).chunk(2, dim=-1)
                skip2_t = skip2 * (1 + g1.view(B, -1, 1, 1)) + b1.view(B, -1, 1, 1)
                g2, b2 = self.film2(h).chunk(2, dim=-1)
                skip1_t = skip1 * (1 + g2.view(B, -1, 1, 1)) + b2.view(B, -1, 1, 1)
            else:
                skip1_t, skip2_t = skip1, skip2

            x = self.up1(spatial_t)
            if x.shape[-2:] != skip2_t.shape[-2:]:
                x = F.interpolate(x, size=skip2_t.shape[-2:], mode='bilinear', align_corners=False)
            x = self.fuse1(torch.cat([x, skip2_t], dim=1))
            x = self.up2(x)
            if x.shape[-2:] != skip1_t.shape[-2:]:
                x = F.interpolate(x, size=skip1_t.shape[-2:], mode='bilinear', align_corners=False)
            x = self.fuse2(torch.cat([x, skip1_t], dim=1))

            af_logit = self.head_af(x)
            af_seq.append(af_logit)

            if self.use_feedback:
                prev_af = torch.sigmoid(af_logit).detach()

            if self.aux_ba:
                ba_delta = torch.sigmoid(self.head_ba(x))
                ba_cum = torch.clamp(ba_cum + ba_delta, 0.0, 1.0)
                ba_seq.append(ba_cum)

        af_pred = torch.stack(af_seq, dim=1)
        out = {'af_pred': af_pred}
        if self.aux_ba:
            out['ba_pred'] = torch.stack(ba_seq, dim=1)
        return out


'''

before = src16[:src16.index(OLD_DEC_START)]
after  = src16[src16.index(OLD_DEC_END):]
src16 = before + NEW_DEC + after

src16 = src16.replace(
    "self.decoder = ForecastDecoderV2(",
    "self.decoder = ForecastDecoderV3("
)
src16 = src16.replace(
    "dropout=dropout, heads=gat_heads, aux_ba=aux_ba,\n        )",
    "dropout=dropout, heads=gat_heads, aux_ba=aux_ba,\n"
    "            use_pos_emb=use_pos_emb, use_feedback=use_feedback,\n"
    "            use_skip_film=use_skip_film,\n        )"
)
src16 = src16.replace(
    "horizon=15,\n                 dropout=0.15, aux_ba=True):",
    "horizon=15,\n                 dropout=0.15, aux_ba=True,\n"
    "                 use_pos_emb=True, use_feedback=True, use_skip_film=True):"
)

nb['cells'][16]['source'] = [src16]
nb['cells'][16]['outputs'] = []
nb['cells'][16]['execution_count'] = None

# ── CELL 20 — EMA fix + v3 flags in model construction ─────────────────────
src20 = ''.join(nb['cells'][20]['source'])

OLD_EMA = '''class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }

    def update(self, model):
        d = self.decay
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in self.shadow:
                    self.shadow[k].mul_(d).add_(v.detach().to(self.shadow[k].dtype),
                                                alpha=1.0 - d)

    def swap_in(self, model):
        live = model.state_dict()
        self.backup = {k: live[k].detach().clone() for k in self.shadow}
        model.load_state_dict({**live, **self.shadow}, strict=False)

    def swap_out(self, model):
        live = model.state_dict()
        model.load_state_dict({**live, **self.backup}, strict=False)'''

NEW_EMA = '''class EMA:
    # Delayed-init EMA: shadow is populated from CURRENT weights at
    # ema_start_epoch, NOT at model construction. This fixes the run27/run28
    # ep10/ep20 val-IoU crash where swap_in loaded 20-ep-old random weights.
    def __init__(self, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.initialized = False

    def maybe_init(self, model, epoch, start_epoch):
        if not self.initialized and epoch >= start_epoch:
            self.shadow = {
                k: v.detach().clone()
                for k, v in model.state_dict().items()
                if v.dtype.is_floating_point
            }
            self.initialized = True

    def update(self, model):
        if not self.initialized:
            return
        d = self.decay
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in self.shadow:
                    self.shadow[k].mul_(d).add_(v.detach().to(self.shadow[k].dtype),
                                                alpha=1.0 - d)

    def swap_in(self, model):
        if not self.initialized:
            return False
        live = model.state_dict()
        self.backup = {k: live[k].detach().clone() for k in self.shadow}
        model.load_state_dict({**live, **self.shadow}, strict=False)
        return True

    def swap_out(self, model):
        if not self.initialized:
            return
        live = model.state_dict()
        model.load_state_dict({**live, **self.backup}, strict=False)'''

assert OLD_EMA in src20, "Old EMA class not found verbatim"
src20 = src20.replace(OLD_EMA, NEW_EMA)

src20 = src20.replace(
    "ema = EMA(model, decay=float(tr['ema_decay']))",
    "ema = EMA(decay=float(tr['ema_decay']))"
)

src20 = src20.replace(
    "            if epoch >= ema_start:\n                ema.update(model)",
    "            ema.maybe_init(model, epoch, ema_start)\n"
    "            if epoch >= ema_start:\n                ema.update(model)"
)

src20 = src20.replace(
    "swap_ctx = (epoch >= ema_start)\n        if swap_ctx:\n            ema.swap_in(model)",
    "swap_ctx = False\n        if epoch >= ema_start and ema.initialized:\n            swap_ctx = ema.swap_in(model)"
)

OLD_MODEL_INIT = """    model = FireSTGraphNetV2(
        in_ch_img=8, in_ch_aux=aux_ch, in_ch_static=1,
        encoder_channels=tuple(mcfg['encoder_channels']),
        gnn_patches=tuple(mcfg['v2_graph_pyramid_patches']),
        gat_heads=int(mcfg['v2_gat_heads']),
        lstm_hidden=int(mcfg['lstm_hidden']),
        lstm_layers=int(mcfg['lstm_layers']),
        lstm_dropout=float(mcfg['lstm_dropout']),
        lstm_attn_heads=int(mcfg['v2_lstm_attn_heads']),
        weather_ch=int(cfg['forecast']['weather_channels']),
        weather_emb=int(cfg['forecast']['weather_emb_dim']),
        horizon=int(cfg['forecast']['horizon']),
        dropout=float(mcfg['decoder_dropout']),
        aux_ba=bool(mcfg['v2_aux_ba_head']),
    ).to(DEVICE)"""

NEW_MODEL_INIT = """    model = FireSTGraphNetV2(
        in_ch_img=8, in_ch_aux=aux_ch, in_ch_static=1,
        encoder_channels=tuple(mcfg['encoder_channels']),
        gnn_patches=tuple(mcfg['v2_graph_pyramid_patches']),
        gat_heads=int(mcfg['v2_gat_heads']),
        lstm_hidden=int(mcfg['lstm_hidden']),
        lstm_layers=int(mcfg['lstm_layers']),
        lstm_dropout=float(mcfg['lstm_dropout']),
        lstm_attn_heads=int(mcfg['v2_lstm_attn_heads']),
        weather_ch=int(cfg['forecast']['weather_channels']),
        weather_emb=int(cfg['forecast']['weather_emb_dim']),
        horizon=int(cfg['forecast']['horizon']),
        dropout=float(mcfg['decoder_dropout']),
        aux_ba=bool(mcfg['v2_aux_ba_head']),
        use_pos_emb   = bool(mcfg.get('v3_pos_emb',        True)),
        use_feedback  = bool(mcfg.get('v3_autoregressive', True)),
        use_skip_film = bool(mcfg.get('v3_skip_film',      True)),
    ).to(DEVICE)"""

assert OLD_MODEL_INIT in src20, "Model init block not found"
src20 = src20.replace(OLD_MODEL_INIT, NEW_MODEL_INIT)

nb['cells'][20]['source'] = [src20]
nb['cells'][20]['outputs'] = []
nb['cells'][20]['execution_count'] = None

# ── CELL 24 — inference: pass v3 flags when rebuilding model_inf ──────────
src24 = ''.join(nb['cells'][24]['source'])
src24 = src24.replace(
    "aux_ba=mcfg['v2_aux_ba_head'],\n).to(DEVICE)",
    "aux_ba=mcfg['v2_aux_ba_head'],\n"
    "    use_pos_emb   = bool(mcfg.get('v3_pos_emb',        True)),\n"
    "    use_feedback  = bool(mcfg.get('v3_autoregressive', True)),\n"
    "    use_skip_film = bool(mcfg.get('v3_skip_film',      True)),\n"
    ").to(DEVICE)"
)
nb['cells'][24]['source'] = [src24]
nb['cells'][24]['outputs'] = []
nb['cells'][24]['execution_count'] = None

# ── CELL 22 + 26 — update baseline labels to include run28 ────────────────
src22 = ''.join(nb['cells'][22]['source'])
src22 = src22.replace("label='run27 0.591'", "label='run28 0.585'")
src22 = src22.replace("label='run27 thr pegged @0.6'", "label='run28 thr 0.55'")
src22 = src22.replace("label='run27 prec=0.80'", "label='run28 prec=0.75'")
nb['cells'][22]['source'] = [src22]

src26 = ''.join(nb['cells'][26]['source'])
src26 = src26.replace(
    'rint(f"  vs run25=0.580   run26=0.522   run27=0.591")',
    'rint(f"  vs run26=0.522   run27=0.591   run28=0.585")'
)
src26 = src26.replace(
    "delta run27    : {best['val_iou_best_thr']-0.591:+.",
    "delta run28    : {best['val_iou_best_thr']-0.585:+."
)
nb['cells'][26]['source'] = [src26]

with open(DST, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('colab_run29.ipynb created and patched.')
