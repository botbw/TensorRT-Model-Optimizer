import torch
import torch.nn as nn
from argparse import ArgumentParser
from torch.nn import Parameter
from typing import Optional, List
from sgl_kernel import cutlass_scaled_fp4_mm, scaled_fp4_quant
from diffusers import FluxPipeline, WanPipeline
from diffusers.utils import export_to_video
from safetensors.torch import load_file

class Fp4Linear(nn.Module):
    """Drop-in replacement for torch.nn.Linear using NVFP4 quantized weights.

    Args:
        in_features (int): Input feature dimension.
        out_features (int): Output feature dimension.
        bias (bool): Whether to include bias.
        is_checkpoint_nvfp4_serialized (bool): If True, expect FP4 checkpoint structure.
        group_size (int): Block size for quantization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int,
        bias: bool = True,
        is_checkpoint_nvfp4_serialized: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.is_checkpoint_nvfp4_serialized = is_checkpoint_nvfp4_serialized
        self.group_size = group_size

        if not self.is_checkpoint_nvfp4_serialized:
            raise ValueError(
                "NVFP4 quantization selected, dynamic quantization not supported."
            )
        if in_features % 16 != 0:
            raise ValueError("Input feature size must be multiple of 16")

        weight_dtype = (
            torch.float8_e4m3fn
            if self.is_checkpoint_nvfp4_serialized
            else torch.float32
        )

        # weight: uint8 [out_features, in_features/2]
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features // 2, dtype=torch.uint8), requires_grad=False
        )

        # per-output scale params
        self.input_scale = nn.Parameter(
            torch.empty((), dtype=torch.float32), requires_grad=False
        )
        self.weight_scale_2 = nn.Parameter(
            torch.empty((), dtype=torch.float32), requires_grad=False
        )

        # blockwise scale: [out_features, in_features/group_size]
        self.weight_scale = nn.Parameter(
            torch.empty(
                out_features, in_features // group_size, dtype=weight_dtype
            ),
            requires_grad=False,
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Will be computed later
        self.alpha = None
        self.weight_scale_interleaved = None

    @torch.no_grad()
    def process_weights_after_loading(self):
        input_scale_2 = self.input_scale.max().to(torch.float32)
        weight_scale_2 = self.weight_scale_2.max().to(torch.float32)
        self.input_scale = Parameter(input_scale_2, requires_grad=False)
        self.weight_scale_2 = Parameter(weight_scale_2, requires_grad=False)
        self.alpha = Parameter(self.input_scale * self.weight_scale_2, requires_grad=False)
        self.input_scale_inv = Parameter(
            (1 / input_scale_2).to(torch.float32), requires_grad=False
        )

        scales = self.weight_scale
        scale_ndim = scales.ndim
        if scale_ndim == 2:
            scales = scales.unsqueeze(0)
        assert scales.ndim == 3
        B, M, K = scales.shape
        round_up_multiple = lambda x, m: (x + m - 1) // m * m
        M_padded = round_up_multiple(M, 128)
        K_padded = round_up_multiple(K, 4)
        padded_scales = torch.zeros((B, M_padded, K_padded), dtype=scales.dtype)
        padded_scales[:B, :M, :K] = scales
        batches, rows, cols = padded_scales.shape
        assert rows % 128 == 0
        assert cols % 4 == 0
        padded_scales = padded_scales.reshape(batches, rows // 128, 4, 32, cols // 4, 4)
        padded_scales = padded_scales.permute((0, 1, 4, 3, 2, 5))
        padded_scales = padded_scales.contiguous().cuda()
        padded_scales = (
            padded_scales.reshape(M_padded, K_padded)
            if scale_ndim == 2
            else padded_scales.reshape(B, M_padded, K_padded)
        )
        self.weight_scale_interleaved = Parameter(padded_scales, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.half()
        assert x.dim() in [1, 2, 3], f"{x.shape=}"
        original_dim = 2
        if x.dim() == 1:
            original_dim = 1
            x = x.unsqueeze(0)
        elif x.dim() == 3:
            assert x.shape[0] == 1
            original_dim = 3
            x = x.squeeze(0)
        output_dtype = x.dtype
        x_m, _ = x.shape
        w_n, _ = self.weight.shape
        output_shape = [x_m, w_n]

        # Quantize BF16/FP16 -> FP4
        x_fp4, x_scale_interleaved = scaled_fp4_quant(x, self.input_scale_inv)

        assert x_fp4.dtype == torch.uint8
        assert x_scale_interleaved.dtype == torch.float8_e4m3fn
        assert self.weight.dtype == torch.uint8
        assert self.weight_scale_interleaved.dtype == torch.float8_e4m3fn
        assert self.alpha.dtype == torch.float32

        out = cutlass_scaled_fp4_mm(
            x_fp4,
            self.weight,
            x_scale_interleaved,
            self.weight_scale_interleaved,
            self.alpha,
            output_dtype,
        )
        if self.bias is not None:
            out = out + self.bias
        out = out.view(*output_shape)
        if original_dim == 1:
            out = out.squeeze(0)
        elif original_dim == 3:
            out = out.unsqueeze(0)
        return out


def replace_linear_with_fp4(
    model: nn.Module,
    group_size: int,
    is_checkpoint_nvfp4_serialized: bool = True,
) -> nn.Module:
    """
    Recursively replace all torch.nn.Linear layers in a model with Fp4Linear.
    """
    for name, module in model.named_children():
        if name in ["time_text_embed", "context_embedder", "x_embedder", "norm_out"]:
            continue
        if isinstance(module, nn.Linear):
            new_layer = Fp4Linear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
                is_checkpoint_nvfp4_serialized=is_checkpoint_nvfp4_serialized,
                group_size=group_size,
            ).to('cuda')
            setattr(model, name, new_layer)
        else:
            replace_linear_with_fp4(model=module, group_size=group_size, is_checkpoint_nvfp4_serialized=is_checkpoint_nvfp4_serialized)
    return model

def process_model_fp4_weights(model: nn.Module):
    """
    Process all Fp4Linear layers in the model after loading weights.
    """
    for module in model.modules():
        if isinstance(module, Fp4Linear):
            module.process_weights_after_loading()

def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, choices=["wan", "flux"], default="flux")
    parser.add_argument("--group-size", type=int, default=16, help="Group size for FP4 quantization.")
    args = parser.parse_args()
    if args.model == "flux":
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev")
        pipe = pipe.to("cuda")
        replace_linear_with_fp4(pipe.transformer, args.group_size)
        pipe.transformer.load_state_dict(load_file("fp4/flux-fp4-max-1-sample-28-step/transformer/diffusion_pytorch_model.safetensors"), strict=False)
        process_model_fp4_weights(pipe.transformer)
        prompt = "A beautiful anime girl with flowers around her."
        image = pipe(prompt=prompt).images[0]
        image.save("example.png")
    elif args.model == "wan":
        pipe = WanPipeline.from_pretrained("Wan-AI/Wan2.2-T2V-A14B-Diffusers")
        pipe = pipe.to("cuda")
        replace_linear_with_fp4(pipe.transformer, args.group_size)
        pipe.transformer.load_state_dict(load_file("fp4/wan2.2-fp4-32-sample-50-step/transformer/diffusion_pytorch_model.safetensors"), strict=False)
        process_model_fp4_weights(pipe.transformer)
        replace_linear_with_fp4(pipe.transformer_2, args.group_size)
        pipe.transformer_2.load_state_dict(load_file("fp4/wan2.2-fp4-32-sample-50-step/transformer_2/diffusion_pytorch_model.safetensors"), strict=False)
        process_model_fp4_weights(pipe.transformer_2)
        prompt = "A beautiful anime girl with flowers around her."
        output = pipe(prompt).frames[0]
        export_to_video(output, "example.mp4", fps=24)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

if __name__ == "__main__":
    main()