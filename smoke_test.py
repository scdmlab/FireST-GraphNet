"""Smoke test for FireST-GraphNet -- semantically aligned multi-source pipeline.

Validates the UCGIS-2026 architecture with SEMANTIC ALIGNMENT:
  - Encoder: fire(8 VIIRS) + aux(19 FirePred) + static(1 LULC) = 28ch
  - Decoder: ALL sources -> 4ch (temp, u, v, rh) -> per-source adapter (4->32) -> backbone
      TRAINING:     FirePred 4ch (extracted from 19-band) -> "firepred"     adapter
      INFERENCE A:  HRRR 4ch     (TMP, UGRD, VGRD, RH)   -> "hrrr"          adapter
      INFERENCE B:  WN2 4ch      (temp, u10, v10, rh)     -> "weathernext2"  adapter
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, ".")
import torch
import numpy as np

print(f"CUDA: {torch.cuda.is_available()}")

# === Legacy model forward pass ===
from firest_graphnet.models.firest_graphnet import FireSTGraphNet

model_legacy = FireSTGraphNet(
    in_channels=8, forecast_mode=False,
    img_h=64, img_w=64,
    encoder_channels=[16, 32, 64],
    decoder_channels=[64, 32, 16],
    lstm_hidden=64, gnn_hidden=64,
)
x = torch.randn(2, 9, 8, 64, 64)
model_legacy.eval()
with torch.no_grad():
    out = model_legacy(x)
print(f"Legacy output: {out.shape}")
assert out.shape == (2, 1, 64, 64), f"Expected (2,1,64,64) got {out.shape}"

# === Forecast model — 28ch encoder + 4ch semantic-aligned decoder ===
FIRE_CH   = 8   # VIIRS Day(6) + Night(2)
AUX_CH    = 19  # FirePred weather/topo/fuel (historical)
STATIC_CH = 1   # ESRI LULC
FUT_CH    = 4   # Canonical: [temp, u_wind, v_wind, rh] — ALL sources

model_fc = FireSTGraphNet(
    in_channels=FIRE_CH + AUX_CH + STATIC_CH,  # 28
    forecast_mode=True,
    future_channels=FUT_CH,
    future_emb_dim=32,
    img_h=64, img_w=64,
    encoder_channels=[16, 32, 64],
    decoder_channels=[64, 32, 16],
    lstm_hidden=64, gnn_hidden=64,
)
model_fc.eval()

af_hist  = torch.randn(2, 10, FIRE_CH, 64, 64)    # VIIRS 8ch
aux_hist = torch.randn(2, 10, AUX_CH, 64, 64)      # FirePred 19ch hist
static   = torch.randn(2, STATIC_CH, 64, 64)        # ESRI LULC 1ch

# --- Test 1: TRAINING — FirePred dynamic 4ch future ---
env_fut_fp = torch.randn(2, 5, FUT_CH, 64, 64)  # 4ch extracted from FirePred
with torch.no_grad():
    logits_fp = model_fc(
        af_hist, aux_hist=aux_hist, env_fut=env_fut_fp,
        env_source="firepred", static=static,
    )
print(f"Forecast [firepred 4ch]: {logits_fp.shape}")
assert logits_fp.shape == (2, 5, 1, 64, 64)

# --- Test 2: INFERENCE Mode A — HRRR 4ch future ---
env_fut_hrrr = torch.randn(2, 5, FUT_CH, 64, 64)  # HRRR 4ch
with torch.no_grad():
    logits_hrrr = model_fc(
        af_hist, aux_hist=aux_hist, env_fut=env_fut_hrrr,
        env_source="hrrr", static=static,
    )
print(f"Forecast [hrrr 4ch]:     {logits_hrrr.shape}")
assert logits_hrrr.shape == (2, 5, 1, 64, 64)

# --- Test 3: INFERENCE Mode B — WeatherNext2 4ch future ---
env_fut_wn2 = torch.randn(2, 5, FUT_CH, 64, 64)  # WN2 4ch
with torch.no_grad():
    logits_wn2 = model_fc(
        af_hist, aux_hist=aux_hist, env_fut=env_fut_wn2,
        env_source="weathernext2", static=static,
    )
print(f"Forecast [weathernext2]: {logits_wn2.shape}")
assert logits_wn2.shape == (2, 5, 1, 64, 64)

# --- Test 4: Backward compat — w_fut alias ---
with torch.no_grad():
    logits_bc = model_fc(
        af_hist, aux_hist=aux_hist, w_fut=env_fut_fp, static=static,
    )
print(f"Backward compat (w_fut): {logits_bc.shape}")
assert logits_bc.shape == (2, 5, 1, 64, 64)

# --- Test 5: Fire-only (no aux/static) ---
model_fc_fire = FireSTGraphNet(
    in_channels=FIRE_CH,  # 8 only
    forecast_mode=True,
    future_channels=FUT_CH,
    future_emb_dim=32,
    img_h=64, img_w=64,
    encoder_channels=[16, 32, 64],
    decoder_channels=[64, 32, 16],
    lstm_hidden=64, gnn_hidden=64,
)
model_fc_fire.eval()
with torch.no_grad():
    logits_fo = model_fc_fire(af_hist, env_fut=env_fut_fp)
print(f"Fire-only forecast:      {logits_fo.shape}")
assert logits_fo.shape == (2, 5, 1, 64, 64)

# === FutureConditionEncoder adapter shapes — all 4ch ===
from firest_graphnet.models.decoder import FutureConditionEncoder

fce = FutureConditionEncoder(future_channels=4, emb_dim=32)
emb_fp   = fce(torch.randn(2, 4, 64, 64), source="firepred")
emb_hrrr = fce(torch.randn(2, 4, 64, 64), source="hrrr")
emb_wn2  = fce(torch.randn(2, 4, 64, 64), source="weathernext2")
assert emb_fp.shape == emb_hrrr.shape == emb_wn2.shape == (2, 32)
print(f"FutureConditionEncoder adapters: all 4ch → (2, 32)  ✓")

# === Dynamic extraction from FirePred ===
from firest_graphnet.data.future_providers import (
    extract_dynamic_from_firepred,
    extract_dynamic_from_firepred_torch,
    DYNAMIC_FUTURE_CHANNELS,
    FIREPRED_BAND_NAMES,
    FIREPRED_CLASSIFICATION,
)

# NumPy extraction
frame_19 = np.random.randn(19, 64, 64).astype(np.float32)
frame_4 = extract_dynamic_from_firepred(frame_19)
assert frame_4.shape == (4, 64, 64), f"Expected (4,64,64) got {frame_4.shape}"
print(f"Dynamic extraction (numpy): (19,64,64) → {frame_4.shape}  ✓")

# Batch extraction
batch_19 = np.random.randn(5, 19, 64, 64).astype(np.float32)
batch_4 = extract_dynamic_from_firepred(batch_19)
assert batch_4.shape == (5, 4, 64, 64), f"Expected (5,4,64,64) got {batch_4.shape}"
print(f"Dynamic extraction (batch): (5,19,64,64) → {batch_4.shape}  ✓")

# Torch extraction
torch_19 = torch.randn(2, 5, 19, 64, 64)
torch_4 = extract_dynamic_from_firepred_torch(torch_19)
assert torch_4.shape == (2, 5, 4, 64, 64)
print(f"Dynamic extraction (torch): (2,5,19,64,64) → {torch_4.shape}  ✓")

# Verify band constants
assert DYNAMIC_FUTURE_CHANNELS == 4
assert len(FIREPRED_BAND_NAMES) == 19
assert FIREPRED_CLASSIFICATION["static"] == [0, 1, 2]
print(f"FirePred band constants: {len(FIREPRED_BAND_NAMES)} bands, {DYNAMIC_FUTURE_CHANNELS} dynamic  ✓")

# === Future providers — all return 4ch ===
from firest_graphnet.data.future_providers import (
    PlaceholderFutureProvider, extract_wind,
)

pfp = PlaceholderFutureProvider(channels=4, source_name="firepred")
env = pfp.get_future_env(tuple(range(5)))
print(f"Placeholder future env: {env.shape}")
assert env.shape == (5, 4, 256, 256)

pfp4 = PlaceholderFutureProvider(channels=4, source_name="hrrr", spatial_size=(64, 64))
env4 = pfp4.get_future_env(tuple(range(5)))
assert env4.shape == (5, 4, 64, 64)
print(f"Placeholder HRRR future: {env4.shape}")

# === Wind extraction — UNIFORM for all sources ===
env_t = torch.randn(2, 5, 4, 64, 64)
wind_fp = extract_wind(env_t, "firepred")
assert wind_fp.shape == (2, 5, 2, 64, 64)
print(f"Wind from FirePred: {wind_fp.shape}")

wind_hr = extract_wind(env_t, "hrrr")
assert wind_hr.shape == (2, 5, 2, 64, 64)
print(f"Wind from HRRR:     {wind_hr.shape}")

wind_wn = extract_wind(env_t, "weathernext2")
assert wind_wn.shape == (2, 5, 2, 64, 64)
print(f"Wind from WN2:      {wind_wn.shape}")

# Verify wind is always channels 1,2 (u,v) regardless of source
assert torch.equal(wind_fp, wind_hr), "Wind should be identical across sources"
assert torch.equal(wind_fp, wind_wn), "Wind should be identical across sources"
print("Wind extraction: uniform across all sources  ✓")

# === Physics constraints (with wind) ===
from firest_graphnet.physics.constraints import PhysicsConstraints

phys = PhysicsConstraints()
ba_init = torch.zeros(2, 1, 64, 64)
# env_fut_fp is (2, 5, 4, 64, 64); wind = channels 1,2
wind_phys = env_fut_fp[:, :, 1:3]  # (2, 5, 2, 64, 64)
losses = phys(logits_fp, ba_init, wind=wind_phys)
print(f"Physics losses (with wind): mono={losses['mono']:.4f} mask={losses['mask']:.4f} pde={losses['pde']:.4f}")

# === Physics constraints (without wind) ===
losses_nw = phys(logits_fp, ba_init)
print(f"Physics losses (no wind):   mono={losses_nw['mono']:.4f} mask={losses_nw['mask']:.4f} pde={losses_nw['pde']:.4f}")

# === PINN loss (with wind) ===
from firest_graphnet.training.losses import ForecastPINNLoss

criterion = ForecastPINNLoss()
criterion.set_epoch(10)
af_fut = torch.randint(0, 2, (2, 5, 1, 64, 64)).float()
loss_dict = criterion(logits_fp, af_fut, ba_init=ba_init, wind=wind_phys)
print(f"PINN loss: total={loss_dict['total']:.4f} sup={loss_dict['sup']:.4f}")

# === Per-horizon metrics ===
from firest_graphnet.training.metrics import compute_metrics_per_horizon

per_h = compute_metrics_per_horizon(logits_fp, af_fut)
print(f"Per-horizon metrics: {len(per_h)} steps, h=0 IoU={per_h[0]['iou']:.4f}")

# === Schedule ===
from firest_graphnet.physics.schedules import PINNSchedule

s = PINNSchedule()
w0 = s.get_weights(0)
w10 = s.get_weights(10)
w15 = s.get_weights(15)
print(f"Schedule: ep0={w0}")
print(f"Schedule: ep10={w10}")
print(f"Schedule: ep15={w15}")

# === FirePred provider (historical) ===
from firest_graphnet.data.firepred_provider import PlaceholderFirePred

fp = PlaceholderFirePred(channels=19, spatial_size=(64, 64))
aux = fp.get_window(tuple(range(10)))
print(f"Placeholder FirePred aux: {aux.shape}")
assert aux.shape == (10, 19, 64, 64)

# === BA update ===
from firest_graphnet.physics.ba_update import BAStateUpdate

ba = BAStateUpdate(gamma=0.3)
ba_seq = ba(logits_fp, ba_init)
print(f"BA sequence: {ba_seq.shape}")
assert ba_seq.shape == (2, 5, 1, 64, 64)

# Verify monotonicity
for i in range(ba_seq.shape[1] - 1):
    diff = ba_seq[:, i+1] - ba_seq[:, i]
    assert diff.min() >= -1e-6, f"BA not monotonic at step {i}"
print("BA monotonicity: PASS")

# === Semantic alignment verification ===
print("\n" + "="*60)
print("  SEMANTIC ALIGNMENT VERIFICATION")
print("="*60)
print(f"  Future channels: {FUT_CH}  (temp, u, v, rh)")
print(f"  FirePred adapter:  4 → 32  (extracted dynamic)")
print(f"  HRRR adapter:      4 → 32  (direct TMP/UGRD/VGRD/RH)")
print(f"  WN2 adapter:       4 → 32  (direct temp/u10/v10/rh)")
print(f"  Wind extraction:   channels 1,2 (uniform for all sources)")
print(f"  ✓ No 19ch→4ch mismatch between training and inference")
print("="*60)

print("\n=== ALL SMOKE TESTS PASSED ===")
