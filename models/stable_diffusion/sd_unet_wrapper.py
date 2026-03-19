import torch
import torch.nn as nn


class SDUNetWrapper(nn.Module):
    """
    Wraps SD 2.1 VAE + U-Net into a single nn.Module whose forward()
    accepts raw image tensors and runs the full encode -> noise -> denoise
    pipeline.

    Exposes submodules via standard naming so FeatureExtractor hooks work:
      - "vae.encoder.*"       -> VAE encoder layers
      - "unet.down_blocks.*"  -> U-Net encoder (downsampling) blocks
      - "unet.mid_block.*"    -> U-Net bottleneck
      - "unet.up_blocks.*"    -> U-Net decoder (upsampling) blocks
    """

    def __init__(self, timestep=200):
        super().__init__()
        from diffusers import StableDiffusionPipeline

        pipe = StableDiffusionPipeline.from_pretrained(
            "sd2-community/stable-diffusion-2-1",
            torch_dtype=torch.float16,
            safety_checker=None,
        )

        # Expose as named submodules (critical for hook registration)
        self.vae = pipe.vae
        self.unet = pipe.unet

        # Precompute null text embedding (77 tokens x 1024 dim for SD 2.1)
        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder

        with torch.no_grad():
            null_tokens = tokenizer(
                [""],
                padding="max_length",
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids
            self.register_buffer(
                "null_text_emb",
                text_encoder(null_tokens.to(text_encoder.device))[0],
            )

        # Free text encoder memory
        del pipe.tokenizer, pipe.text_encoder
        del tokenizer, text_encoder

        # Noise scheduler for alpha values
        self.noise_scheduler = pipe.scheduler
        del pipe

        # Configurable timestep
        self.timestep = timestep

        # Freeze everything
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.eval()

    def forward(self, images):
        """
        Args:
            images: (B, 3, 512, 512) float tensor, normalized to [-1, 1]
        Returns:
            noise_pred: U-Net output (actual features are captured by hooks)
        """
        # 1. VAE encode -> latent z_0
        with torch.no_grad():
            z_0 = self.vae.encode(images).latent_dist.mean
            z_0 = z_0 * self.vae.config.scaling_factor

        # 2. Add noise at timestep t (fixed seed for reproducibility)
        torch.manual_seed(42)
        noise = torch.randn_like(z_0)
        timesteps = torch.full(
            (z_0.shape[0],), self.timestep,
            device=z_0.device, dtype=torch.long,
        )
        z_t = self.noise_scheduler.add_noise(z_0, noise, timesteps)

        # 3. U-Net forward with null conditioning
        null_emb = self.null_text_emb.expand(z_0.shape[0], -1, -1)
        noise_pred = self.unet(
            z_t, timesteps, encoder_hidden_states=null_emb,
        ).sample

        return noise_pred
