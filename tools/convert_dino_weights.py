"""
DINOv3 .pth → HuggingFace state_dict key 변환기

문제:
  로컬 .pth 파일의 key 이름(blocks.N.attn.qkv.weight 등)이
  HuggingFace AutoModel의 key 이름(model.layer.N.attention.q_proj.weight 등)과
  달라서 610개 key 중 2개만 로드되고 나머지는 랜덤 초기화 상태로 남아있었음.

변환 규칙:
  cls_token                     → embeddings.cls_token
  storage_tokens                → embeddings.register_tokens
  mask_token  [1,D]             → embeddings.mask_token  [1,1,D] (unsqueeze)
  patch_embed.proj.weight/bias  → embeddings.patch_embeddings.weight/bias
  blocks.N.norm1/2.weight/bias  → model.layer.N.norm1/2.weight/bias
  blocks.N.ls1/2.gamma          → model.layer.N.layer_scale1/2.lambda1
  blocks.N.attn.qkv.weight      → model.layer.N.attention.q/k/v_proj.weight  (split)
  blocks.N.attn.proj.weight/bias→ model.layer.N.attention.o_proj.weight/bias
  blocks.N.mlp.w1.weight/bias   → model.layer.N.mlp.gate_proj.weight/bias
  blocks.N.mlp.w2.weight/bias   → model.layer.N.mlp.up_proj.weight/bias
  blocks.N.mlp.w3.weight/bias   → model.layer.N.mlp.down_proj.weight/bias
  norm.weight/bias              → norm.weight/bias  (동일)

Usage:
  python convert_dino_weights.py
  → 변환된 weight를 저장 후 DINOFeatureExtractor에서 바로 로드 가능
"""

from pathlib import Path
import torch


SRC_PTH  = "/home/ssu/Downloads/isaac-sim-standalone-6.0.0-linux-x86_64/th_ws/model/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"
DST_PTH  = "/home/ssu/Downloads/isaac-sim-standalone-6.0.0-linux-x86_64/th_ws/model/dinov3_vit7b16_hf_converted.pth"


def convert_state_dict(src_sd: dict, hidden_size: int = 4096) -> dict:
    """
    .pth 원본 state_dict → HuggingFace Dinov3Model state_dict 변환.
    hidden_size: 모델의 D (기본 4096)
    """
    dst = {}

    for k, v in src_sd.items():

        # ── embeddings ────────────────────────────────────────────────────────
        if k == "cls_token":
            dst["embeddings.cls_token"] = v
        elif k == "storage_tokens":
            dst["embeddings.register_tokens"] = v
        elif k == "mask_token":
            # .pth: [1, D] → HF: [1, 1, D]
            dst["embeddings.mask_token"] = v.unsqueeze(1) if v.dim() == 2 else v
        elif k == "patch_embed.proj.weight":
            dst["embeddings.patch_embeddings.weight"] = v
        elif k == "patch_embed.proj.bias":
            dst["embeddings.patch_embeddings.bias"] = v

        # ── final norm ────────────────────────────────────────────────────────
        elif k in ("norm.weight", "norm.bias"):
            dst[k] = v   # 동일

        # ── rope_embed 등 HF에 없는 key → 무시 ────────────────────────────────
        elif k in ("rope_embed.periods", "local_cls_norm.weight", "local_cls_norm.bias"):
            pass  # HF 모델에 대응 없음

        # ── per-layer key 처리 ────────────────────────────────────────────────
        elif k.startswith("blocks."):
            parts = k.split(".")          # ["blocks", "N", "sub", ...]
            N     = parts[1]              # layer index (문자열)
            rest  = ".".join(parts[2:])   # e.g. "norm1.weight", "attn.qkv.weight"
            prefix = f"model.layer.{N}"

            # norm1, norm2
            if rest in ("norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias"):
                dst[f"{prefix}.{rest}"] = v

            # layer_scale
            elif rest == "ls1.gamma":
                dst[f"{prefix}.layer_scale1.lambda1"] = v
            elif rest == "ls2.gamma":
                dst[f"{prefix}.layer_scale2.lambda1"] = v

            # attention: merged QKV → split q / k / v
            elif rest == "attn.qkv.weight":
                # shape: [3*D, D]
                q, k_, vv = v.chunk(3, dim=0)
                dst[f"{prefix}.attention.q_proj.weight"] = q
                dst[f"{prefix}.attention.k_proj.weight"] = k_
                dst[f"{prefix}.attention.v_proj.weight"] = vv

            elif rest == "attn.qkv.bias":
                q, k_, vv = v.chunk(3, dim=0)
                dst[f"{prefix}.attention.q_proj.bias"] = q
                dst[f"{prefix}.attention.k_proj.bias"] = k_
                dst[f"{prefix}.attention.v_proj.bias"] = vv

            # attention output projection
            elif rest == "attn.proj.weight":
                dst[f"{prefix}.attention.o_proj.weight"] = v
            elif rest == "attn.proj.bias":
                dst[f"{prefix}.attention.o_proj.bias"] = v

            # MLP: w1=gate_proj, w2=up_proj, w3=down_proj
            elif rest == "mlp.w1.weight":
                dst[f"{prefix}.mlp.gate_proj.weight"] = v
            elif rest == "mlp.w1.bias":
                dst[f"{prefix}.mlp.gate_proj.bias"] = v
            elif rest == "mlp.w2.weight":
                dst[f"{prefix}.mlp.up_proj.weight"] = v
            elif rest == "mlp.w2.bias":
                dst[f"{prefix}.mlp.up_proj.bias"] = v
            elif rest == "mlp.w3.weight":
                dst[f"{prefix}.mlp.down_proj.weight"] = v
            elif rest == "mlp.w3.bias":
                dst[f"{prefix}.mlp.down_proj.bias"] = v

            else:
                print(f"  [SKIP] 미처리 key: {k}")

        else:
            print(f"  [SKIP] 알 수 없는 key: {k}")

    return dst


def verify(dst_sd: dict, model_sd: dict):
    """변환 결과를 HuggingFace 모델 state_dict와 대조."""
    dst_keys = set(dst_sd.keys())
    hf_keys  = set(model_sd.keys())

    matched   = dst_keys & hf_keys
    only_dst  = dst_keys - hf_keys
    only_hf   = hf_keys - dst_keys

    print(f"\n  매칭 성공 : {len(matched)} / {len(hf_keys)}")
    if only_dst:
        print(f"  변환 결과에만 있음 ({len(only_dst)}개): {list(only_dst)[:5]}")
    if only_hf:
        print(f"  HF에만 있음 ({len(only_hf)}개): {list(only_hf)[:5]}")

    # shape 불일치 확인
    mismatch = []
    for k in matched:
        if dst_sd[k].shape != model_sd[k].shape:
            mismatch.append((k, dst_sd[k].shape, model_sd[k].shape))
    if mismatch:
        print(f"  shape 불일치 ({len(mismatch)}개):")
        for k, ds, hs in mismatch:
            print(f"    {k}: 변환={ds}  HF={hs}")
    else:
        print(f"  shape 불일치: 없음 ✓")

    return len(matched), len(only_hf)


if __name__ == "__main__":
    from transformers import AutoConfig, AutoModel

    print(f"[1] 원본 .pth 로드: {SRC_PTH}")
    raw  = torch.load(SRC_PTH, map_location="cpu")
    src_sd = raw.get("model", raw) if isinstance(raw, dict) else raw
    print(f"    원본 key 수: {len(src_sd)}")

    print("[2] HuggingFace 모델 구조 로드 (가중치 없이)")
    config   = AutoConfig.from_pretrained("facebook/dinov3-vit7b16-pretrain-lvd1689m")
    hf_model = AutoModel.from_config(config)
    hf_sd    = hf_model.state_dict()
    print(f"    HF key 수: {len(hf_sd)}")

    print("[3] key 변환 중 ...")
    dst_sd = convert_state_dict(src_sd, hidden_size=config.hidden_size)
    print(f"    변환 후 key 수: {len(dst_sd)}")

    print("[4] 변환 결과 검증 ...")
    n_matched, n_missing = verify(dst_sd, hf_sd)

    print("\n[5] 변환된 weight를 HuggingFace 모델에 로드 테스트 ...")
    missing, unexpected = hf_model.load_state_dict(dst_sd, strict=False)
    print(f"    missing keys   : {len(missing)}")
    print(f"    unexpected keys: {len(unexpected)}")
    if missing:
        print(f"    missing 예시   : {missing[:5]}")

    if len(missing) < 10:
        print(f"\n[6] 저장: {DST_PTH}")
        torch.save(dst_sd, DST_PTH)
        print(f"    완료  ({Path(DST_PTH).stat().st_size / 1e9:.2f} GB)")
    else:
        print(f"\n[WARNING] missing이 {len(missing)}개 — 저장 전 mapping 확인 필요")
