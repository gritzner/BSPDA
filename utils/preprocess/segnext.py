import lzma
import torch

for fn in ("mscan_t.pth", "mscan_s.pth", "mscan_b.pth", "mscan_l.pth"):
    print(f"processing '{fn}'...")
    weights = torch.load(fn)["state_dict"]

    submapping = {f"norm1.{suffix}": f"norm.{suffix}" for suffix in (
        "bias", "num_batches_tracked", "running_mean", "running_var", "weight"
    )}
    submapping.update({
        f"norm2.{suffix}": f"mlp.0.{suffix}" for suffix in (
            "bias", "num_batches_tracked", "running_mean", "running_var", "weight"
        )
    })
    submapping.update({"layer_scale_1": "att_scaling", "layer_scale_2": "mlp_scaling"}) # requires unsqueezing
    submapping.update({f"mlp.fc1.{suffix}": f"mlp.1.{suffix}" for suffix in ("bias", "weight")})
    submapping.update({f"mlp.dwconv.dwconv.{suffix}": f"mlp.2.{suffix}" for suffix in ("bias", "weight")})
    submapping.update({f"mlp.fc2.{suffix}": f"mlp.4.{suffix}" for suffix in ("bias", "weight")})
    submapping.update({f"attn.proj_1.{suffix}": f"attention.0.{suffix}" for suffix in ("bias", "weight")})
    submapping.update({f"attn.proj_2.{suffix}": f"attention.3.{suffix}" for suffix in ("bias", "weight")})
    submapping.update({
        f"attn.spatial_gating_unit.conv0.{suffix}": f"attention.2.conv0.{suffix}" for suffix in ("bias", "weight")
    })
    submapping.update({
        f"attn.spatial_gating_unit.conv3.{suffix}": f"attention.2.conv1.{suffix}" for suffix in ("bias", "weight")
    })
    for i in range(3):
        for j in range(2):
            submapping.update({
                f"attn.spatial_gating_unit.conv{i}_{j+1}.{suffix}": f"attention.2.branch{i}.{j}.{suffix}" for suffix in (
                    "bias", "weight"
                )
            })

    mapping = {
        f"patch_embed1.proj.1.{suffix}": f"stage0.1.{suffix}" for suffix in (
            "bias", "num_batches_tracked", "running_mean", "running_var", "weight"
        )
    }
    mapping.update({
        f"patch_embed1.proj.3.{suffix}": f"stage1.embed_patch.0.{suffix}" for suffix in (
            "bias", "weight"
        )
    })
    mapping.update({
        f"patch_embed1.proj.4.{suffix}": f"stage1.embed_patch.1.{suffix}" for suffix in (
            "bias", "num_batches_tracked", "running_mean", "running_var", "weight"
        )
    })
    for i in range(2, 5):
        mapping.update({
            f"patch_embed{i}.proj.{suffix}": f"stage{i}.embed_patch.0.{suffix}" for suffix in (
                "bias", "weight"
            )
        })
        mapping.update({
            f"patch_embed{i}.norm.{suffix}": f"stage{i}.embed_patch.1.{suffix}" for suffix in (
                "bias", "num_batches_tracked", "running_mean", "running_var", "weight"
            )
        })
    for i in range(1, 5):
        j = 0
        while f"block{i}.{j}.norm1.bias" in weights:
            mapping.update({f"block{i}.{j}.{k}": f"stage{i}.block{j}.{v}" for k, v in submapping.items()})
            j += 1
        mapping.update({
            f"norm{i}.{suffix}": f"stage{i}.norm.{suffix}" for suffix in (
                "bias", "weight"
            )
        })

    with lzma.open(f"mscan_{fn[6]}.pt.xz", "wb", preset=9) as f:
        torch.save({
            v: (weights[k].reshape(1, -1, 1, 1) if "layer_scale_" in k else weights[k]) for k, v in mapping.items()
        }, f)
