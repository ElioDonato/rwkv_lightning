import torch

from infer.rwkv_batch.sampler import sample, sample_batch_per_row
from infer.rwkv_batch.utils import sampler_gumbel_batch


def get_torch():
    return torch


def get_sample():
    return sample


def get_sampler_gumbel_batch():
    return sampler_gumbel_batch


def get_sample_batch_per_row():
    """The per-row sampler (see rwkv_batch/sampler.sample_batch_per_row): takes
    whole-batch logits + per-row penalties/rand-states + per-row scalar lists
    and returns [B,1] sampled tokens, each row with its own config. Resolved
    lazily so a caller-side monkeypatch can substitute the impl."""
    return sample_batch_per_row
