"""
Variant configuration helper for Whipstr STT (ASR) model family.

Defines scaling table for whipstr-small, whipstr-base, whipstr-medium,
and whipstr-large, and provides helpers to create configs from variant names.

Usage:

    from whipstr.whipstr_variants import get_variant_config, list_variants

    config = get_variant_config("whipstr-base")
    for name in list_variants():
        cfg = get_variant_config(name)
        print(f"{name}: {cfg['d_model']}d, {cfg['num_encoder_layers']}enc/{cfg['num_decoder_layers']}dec")
"""

from .hf_integration import WhipstrConfig


VARIANT_CONFIGS = {
    "whipstr-small": {
        "encoder_embed_dim": 32,
        "d_model": 128,
        "nhead": 4,
        "num_encoder_layers": 2,
        "num_decoder_layers": 2,
        "dim_feedforward": 512,
        "dropout": 0.1,
        "stride": 1,
        "window_size": 11,
        "encoder_fc2": 32,
    },
    "whipstr-base": {
        "encoder_embed_dim": 64,
        "d_model": 256,
        "nhead": 8,
        "num_encoder_layers": 4,
        "num_decoder_layers": 4,
        "dim_feedforward": 1024,
        "dropout": 0.1,
        "stride": 1,
        "window_size": 11,
        "encoder_fc2": 64,
    },
    "whipstr-medium": {
        "encoder_embed_dim": 128,
        "d_model": 512,
        "nhead": 8,
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "dim_feedforward": 2048,
        "dropout": 0.1,
        "stride": 1,
        "window_size": 11,
        "encoder_fc2": 128,
    },
    "whipstr-large": {
        "encoder_embed_dim": 192,
        "d_model": 768,
        "nhead": 12,
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "dim_feedforward": 3072,
        "dropout": 0.15,
        "stride": 1,
        "window_size": 11,
        "encoder_fc2": 192,
    },
}


def list_variants():
    """Return list of available variant names."""
    return list(VARIANT_CONFIGS.keys())


def get_variant_config(variant_name, vocab_size=None):
    """Get the full config dict for a variant.

    Args:
        variant_name: One of "whipstr-small", "whipstr-base",
                      "whipstr-medium", "whipstr-large".
        vocab_size: Optional vocab_size override. If None, uses
                    the default (43).

    Returns:
        Dict with all parameters needed to instantiate a model.
    """
    if variant_name not in VARIANT_CONFIGS:
        raise ValueError(
            f"Unknown variant '{variant_name}'. "
            f"Available: {', '.join(list_variants())}"
        )

    cfg = dict(VARIANT_CONFIGS[variant_name])
    if vocab_size is not None:
        cfg["vocab_size"] = vocab_size
    else:
        cfg["vocab_size"] = 43

    return cfg


def get_hf_config(variant_name, vocab_size=None):
    """Create a WhipstrConfig for a variant, suitable for HF save/load.

    Args:
        variant_name: One of "whipstr-small", "whipstr-base",
                      "whipstr-medium", "whipstr-large".
        vocab_size: Optional vocab_size override.

    Returns:
        WhipstrConfig instance.
    """
    cfg = get_variant_config(variant_name, vocab_size=vocab_size)
    return WhipstrConfig(
        vocab_size=cfg["vocab_size"],
        stride=cfg["stride"],
        window_size=cfg["window_size"],
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_encoder_layers=cfg["num_encoder_layers"],
        num_decoder_layers=cfg["num_decoder_layers"],
        dim_feedforward=cfg["dim_feedforward"],
        dropout=cfg["dropout"],
        encoder_embed_dim=cfg["encoder_embed_dim"],
    )


def get_encoder_kwargs(variant_name):
    """Get encoder-specific kwargs (channels, fc sizes, output dim).

    Args:
        variant_name: One of "whipstr-small", "whipstr-base",
                      "whipstr-medium", "whipstr-large".

    Returns:
        Dict with encoder_embed_dim, encoder_channels, encoder_fc.
    """
    cfg = get_variant_config(variant_name)
    return {
        "encoder_embed_dim": cfg["encoder_embed_dim"],
        "encoder_channels": cfg["encoder_channels"],
        "encoder_fc": cfg["encoder_fc"],
    }


def print_variant_table():
    """Print a summary table of all variants."""
    header = (
        f"{'Variant':<18} {'emb':>4} {'d_model':>6} {'nhead':>5} "
        f"{'enc':>3} {'dec':>3} {'ff':>5} {'drop':>5}"
    )
    print(header)
    print("-" * len(header))
    for name in list_variants():
        cfg = VARIANT_CONFIGS[name]
        print(
            f"{name:<18} {cfg['encoder_embed_dim']:>4} {cfg['d_model']:>6} "
            f"{cfg['nhead']:>5} {cfg['num_encoder_layers']:>3} "
            f"{cfg['num_decoder_layers']:>3} {cfg['dim_feedforward']:>5} "
            f"{cfg['dropout']:>5}"
        )


if __name__ == "__main__":
    print_variant_table()
