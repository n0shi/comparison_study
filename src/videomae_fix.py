"""Load VideoMAE with its attention biases actually loaded.

MCG-NJU/videomae-* checkpoints store the self-attention biases as `q_bias`
and `v_bias` (the key projection has no bias). transformers 5.x builds
query/key/value Linear layers with their own biases and does not convert the
old names, so it reports `query/key/value.bias` as MISSING and initialises
them afresh. Part 1 trained this way (data_note.md, problem 8).

`load_videomae(name)` loads the model, copies q_bias -> query.bias and
v_bias -> value.bias, sets key.bias to zero (as in the original model), and
fails unless every encoder layer was patched.

Check on any node:
    python src/videomae_fix.py MCG-NJU/videomae-base
"""
from __future__ import annotations

import re
import sys

LAYER_RE = re.compile(r"encoder\.layer\.(\d+)\.attention\.attention\.([qv])_bias$")


def checkpoint_biases(name: str) -> dict[int, dict[str, "object"]]:
    """{layer: {'q': tensor, 'v': tensor}} read straight from the checkpoint file."""
    from huggingface_hub import hf_hub_download

    try:
        from safetensors.torch import load_file
        sd = load_file(hf_hub_download(name, "model.safetensors"))
    except Exception:
        import torch
        sd = torch.load(hf_hub_download(name, "pytorch_model.bin"), map_location="cpu",
                        weights_only=True)
    out: dict[int, dict] = {}
    for k, v in sd.items():
        m = LAYER_RE.search(k)
        if m and not k.startswith("decoder"):
            out.setdefault(int(m.group(1)), {})[m.group(2)] = v
    return out


def fix_biases(model, name: str) -> int:
    import torch

    enc = getattr(model, "encoder", None) or model.videomae.encoder
    layers = enc.layer
    ck = checkpoint_biases(name)
    if len(ck) != len(layers) or any(set(b) != {"q", "v"} for b in ck.values()):
        raise RuntimeError(f"VideoMAE bias fix: checkpoint {name} has q/v biases for layers "
                           f"{sorted(ck)}, model has {len(layers)} layers")
    with torch.no_grad():
        for i, layer in enumerate(layers):
            att = layer.attention.attention
            for proj in ("query", "key", "value"):
                if getattr(att, proj).bias is None:
                    raise RuntimeError(f"VideoMAE bias fix: layer {i} {proj} has no bias "
                                       "parameter; this transformers version needs a "
                                       "different fix")
            att.query.bias.copy_(ck[i]["q"])
            att.value.bias.copy_(ck[i]["v"])
            att.key.bias.zero_()
    return len(layers)


def load_videomae(name: str, cls=None, **kw):
    """Drop-in for VideoMAEModel.from_pretrained(name, **kw) with biases loaded."""
    if cls is None:
        from transformers import VideoMAEModel as cls
    model = cls.from_pretrained(name, **kw)
    n = fix_biases(model, name)
    print(f"[videomae_fix] {name}: q/v attention biases loaded into {n} encoder layers",
          flush=True)
    return model


class FixedVideoMAEModel:
    """Stands in for `transformers.VideoMAEModel` inside part-1 model.py, which
    only ever calls `VideoMAEModel.from_pretrained(cfg.video_model)`."""

    @staticmethod
    def from_pretrained(name, **kw):
        from transformers import VideoMAEModel
        return load_videomae(name, VideoMAEModel, **kw)


if __name__ == "__main__":
    import torch
    name = sys.argv[1] if len(sys.argv) > 1 else "MCG-NJU/videomae-base"
    m = load_videomae(name)
    ck = checkpoint_biases(name)
    att = m.encoder.layer[0].attention.attention
    ok = (torch.equal(att.query.bias, ck[0]["q"]) and torch.equal(att.value.bias, ck[0]["v"])
          and not att.key.bias.any())
    print(f"layer 0: query.bias == q_bias, value.bias == v_bias, key.bias == 0: {ok}")
    print(f"|q_bias| mean {ck[0]['q'].abs().mean():.4f}, |v_bias| mean {ck[0]['v'].abs().mean():.4f}")
    sys.exit(0 if ok else 1)
