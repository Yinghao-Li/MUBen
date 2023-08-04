import re
import torch
import warnings

from typing import Optional, Tuple
from torch import nn
from torch_scatter import scatter

from . import output_modules


def create_model(args):
    shared_args = dict(
        hidden_channels=args["embedding_dimension"],
        num_layers=args["num_layers"],
        num_rbf=args["num_rbf"],
        rbf_type=args["rbf_type"],
        trainable_rbf=args["trainable_rbf"],
        activation=args["activation"],
        neighbor_embedding=args["neighbor_embedding"],
        cutoff_lower=args["cutoff_lower"],
        cutoff_upper=args["cutoff_upper"],
        max_z=args["max_z"],
        max_num_neighbors=args["max_num_neighbors"],
    )

    # representation network
    if args["model"] == "equivariant-transformer":
        from .torchmd_et import TorchMD_ET

        is_equivariant = True
        representation_model = TorchMD_ET(
            attn_activation=args["attn_activation"],
            num_heads=args["num_heads"],
            distance_influence=args["distance_influence"],
            layernorm_on_vec=args["layernorm_on_vec"],
            **shared_args,
        )
    else:
        raise ValueError(f'Unknown architecture: {args["model"]}')

    # create output network
    output_prefix = "Equivariant" if is_equivariant else ""
    output_model = getattr(output_modules, output_prefix + args["output_model"])(
        args["embedding_dimension"], args["activation"]
    )

    # create the denoising output network
    output_model_noise = None
    if args['output_model_noise'] is not None:
        output_model_noise = getattr(output_modules, output_prefix + args["output_model_noise"])(
            args["embedding_dimension"], args["activation"],
        )

    # combine representation and output network
    model = TorchMDNet(
        representation_model,
        output_model,
        output_model_noise=output_model_noise,
        position_noise_scale=args['position_noise_scale'],
    )
    return model


def load_model(config, **kwargs):
    ckpt = torch.load(config.checkpoint_path, map_location="cpu")

    if config is None:
        config = ckpt["hyper_parameters"]

    for key, value in kwargs.items():
        if key not in config:
            warnings.warn(f'Unknown hyperparameter: {key}={value}')
        config[key] = value

    model = create_model(config)

    state_dict = {re.sub(r"^model\.", "", k): v for k, v in ckpt["state_dict"].items()}
    loading_return = model.load_state_dict(state_dict, strict=False)

    if len(loading_return.unexpected_keys) > 0:
        # Should only happen if not applying denoising during fine-tuning.
        # we also removed mean and std from the model
        assert all(("output_model_noise" in k or "pos_normalizer" in k or k in ['mean', 'std'])
                   for k in loading_return.unexpected_keys)
    assert len(loading_return.missing_keys) == 0, f"Missing keys: {loading_return.missing_keys}"

    return model


class TorchMDNet(nn.Module):
    def __init__(
            self,
            representation_model,
            output_model,
            prior_model=None,
            reduce_op="add",
            derivative=False,
            output_model_noise=None,
            position_noise_scale=0.,
    ):
        super(TorchMDNet, self).__init__()
        self.representation_model = representation_model
        self.output_model = output_model

        self.prior_model = prior_model

        self.reduce_op = reduce_op
        self.derivative = derivative
        self.output_model_noise = output_model_noise
        self.position_noise_scale = position_noise_scale

        if self.position_noise_scale > 0:
            self.pos_normalizer = AccumulatedNormalization(accumulator_shape=(3,))

        self.reset_parameters()

    def reset_parameters(self):
        self.representation_model.reset_parameters()
        self.output_model.reset_parameters()
        if self.prior_model is not None:
            self.prior_model.reset_parameters()

    def forward(self, z, pos, batch: Optional[torch.Tensor] = None):
        assert z.dim() == 1 and z.dtype == torch.long
        batch = torch.zeros_like(z) if batch is None else batch

        # run the potentially wrapped representation model
        x, v, z, pos, batch = self.representation_model(z, pos, batch=batch)

        # predict noise
        noise_pred = None
        if self.output_model_noise is not None:
            noise_pred = self.output_model_noise.pre_reduce(x, v, z, pos, batch)

        # apply the output network
        x = self.output_model.pre_reduce(x, v, z, pos, batch)

        # apply prior model
        if self.prior_model is not None:
            x = self.prior_model(x, z, pos, batch)

        # aggregate atoms
        out = scatter(x, batch, dim=0, reduce=self.reduce_op)

        # apply output model after reduction
        out = self.output_model.post_reduce(out)

        return out, noise_pred, None


class AccumulatedNormalization(nn.Module):
    """Running normalization of a tensor."""

    def __init__(self, accumulator_shape: Tuple[int, ...], epsilon: float = 1e-8):
        super(AccumulatedNormalization, self).__init__()

        self._epsilon = epsilon
        self.register_buffer("acc_sum", torch.zeros(accumulator_shape))
        self.register_buffer("acc_squared_sum", torch.zeros(accumulator_shape))
        self.register_buffer("acc_count", torch.zeros((1,)))
        self.register_buffer("num_accumulations", torch.zeros((1,)))

    def update_statistics(self, batch: torch.Tensor):
        batch_size = batch.shape[0]
        self.acc_sum += batch.sum(dim=0)
        self.acc_squared_sum += batch.pow(2).sum(dim=0)
        self.acc_count += batch_size
        self.num_accumulations += 1

    @property
    def acc_count_safe(self):
        return self.acc_count.clamp(min=1)

    @property
    def mean(self):
        return self.acc_sum / self.acc_count_safe

    @property
    def std(self):
        return torch.sqrt(
            (self.acc_squared_sum / self.acc_count_safe) - self.mean.pow(2)
        ).clamp(min=self._epsilon)

    def forward(self, batch: torch.Tensor):
        if self.training:
            self.update_statistics(batch)
        return (batch - self.mean) / self.std
