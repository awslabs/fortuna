from fortuna.partitioner.base import Partitioner
from fortuna.typing import AxisDims


def get_default_partitioner(model_name_or_path: str, axes_dims: AxisDims):
    names = ["gptj", "roberta"]

    if model_name_or_path.lower() == "eleutherai/gpt-j-6b":
        return Partitioner(
                axes_dims=axes_dims,
                rules={
                    'transformer/wte/embedding': ('mp', 'fsdp'),
                    'attn/(k_proj|q_proj|v_proj)/kernel': ('fsdp', 'mp'),
                    'attn/out_proj/kernel': ('mp', 'fsdp'),
                    'mlp/fc_in/kernel': ('fsdp', 'mp'),
                    'mlp/fc_in/bias': ('mp',),
                    'mlp/fc_out/kernel': ('mp', 'fsdp'),
                    'lm_head/kernel': ('fsdp', 'mp'),
                    'lm_head/bias': ('mp',),
                }
            )

    if model_name_or_path == "roberta-base":
        return Partitioner(
                axes_dims=axes_dims,
                rules={
                    'attention/self/(key|query|value)/kernel': ('fsdp', 'mp'),
                    'attention/output/dense/kernel': ('mp', 'fsdp'),
                    'intermediate/dense/kernel': ('fsdp', 'mp'),
                    'intermediate/dense/bias': ('mp',),
                    'output/dense/kernel': ('mp', 'fsdp'),
                    'lm_head/decoder/kernel': ('fsdp', 'mp'),
                    'lm_head/decoder/bias': ('mp',),
                    'lm_head/dense/kernel': ('fsdp', 'mp'),
                    'lm_head/dense/bias': ('mp',),
                }
        )

    raise ValueError("`model_name_or_path` not recognized."
                     f"Please choose one among the following options: {names}.")
