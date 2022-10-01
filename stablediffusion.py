from ast import iter_child_nodes
from pytorch_lightning import seed_everything
import torch
from torch import autocast
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers import LMSDiscreteScheduler
from tqdm.auto import tqdm
from PIL import Image
import torch_tensorrt

class StableDiffusion:
    def __init__(self, huggingface_auth_token):
        self.vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", use_auth_token=huggingface_auth_token)
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
        self.unet = torch.jit.load('unet_v1_4_fp16_pytorch_sim.ts')
        self.unet.eval()
        self.scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
        self.vae.to("cuda")
        self.text_encoder.to("cuda")
        self.unet.to("cuda")
    
    def create(self, prompt = "a photograph of an astronaut riding a horse", width = 512, height = 512, sample_steps = 50, iter_count = 2, scale = 10, seed = -1, batch_size = 1, filename="output.png"):
        seed_everything(int(seed))
        width = int(width)
        height = int(height)
        sample_steps = int(sample_steps)
        batch_size = int(batch_size)
        scale = float(scale)

        text_input = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        text_embeddings = self.text_encoder(text_input.input_ids.to("cuda"))[0]
        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to("cuda"))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings]).half().cuda()

        for iter in range(iter_count):
            latents = torch.randn((batch_size, 4, height // 8, width // 8))
            latents = latents.to("cuda")

            self.scheduler.set_timesteps(sample_steps)

            latents = latents * self.scheduler.sigmas[0]

            self.scheduler.set_timesteps(sample_steps)
            # Denoising Loop
            with torch.inference_mode(), autocast("cuda"):
                for i, t in tqdm(enumerate(self.scheduler.timesteps)):
                    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                    latent_model_input = torch.cat([latents] * 2)
                    sigma = self.scheduler.sigmas[i]
                    latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

                    # predict the noise residual
                    noise_pred = noise_pred = self.unet(latent_model_input.half().cuda(), torch.Tensor(t).half().cuda(), text_embeddings)

                    # perform guidance
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, i, latents)["prev_sample"]

                # scale and decode the image latents with vae
                latents = 1 / 0.18215 * latents
                image = self.vae.decode(latents)

                #Convert the image with PIL and save it
                image = (image.sample / 2.0 + 0.5).clamp(0, 1)
                image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
                images = (image * 255).round().astype("uint8")
                pil_images = [Image.fromarray(image) for image in images]
                pil_images[0].save(f"{iter}-{filename}")
