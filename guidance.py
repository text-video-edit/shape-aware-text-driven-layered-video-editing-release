from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, EulerDiscreteScheduler, DPMSolverMultistepScheduler

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F

import time, os
import cv2
import numpy as np
import imageio

class StableDiffusion(nn.Module):
    def __init__(self, device, v2, min_step, max_step):
        super().__init__()

        try:
            with open('./TOKEN', 'r') as f:
                self.token = f.read().replace('\n', '') # remove the last \n!
                print(f'[INFO] loaded hugging face access token from ./TOKEN!')
        except FileNotFoundError as e:
            self.token = True
            print(f'[INFO] try to load hugging face access token from the default place, make sure you have run `huggingface-cli login`.')
        
        self.device = device
        self.num_train_timesteps = 1000
        self.min_step = int(min_step)
        self.max_step = int(max_step)
        self.clip_gradient = False
        self.v2 = v2

        print(f'[INFO] loading stable diffusion...')
        local_pretrained_dir = f'pretrained-guidance/{"v1" if not v2 else "v2"}'
        hf_paths = {
            'v1': 'runwayml/stable-diffusion-v1-5',
            'v2': 'stabilityai/stable-diffusion-2-1'
        }
        # local_paths = {k: local_pretrained_dir for k in hf_paths[f'{"1.5" if not v2 else "2"}'].keys()}
        if not os.path.isdir(local_pretrained_dir):
            save_pretrained = True
            load_paths = hf_paths[f'{"v1" if not v2 else "v2"}']
            os.makedirs(local_pretrained_dir, exist_ok=True)
        else:
            save_pretrained = False
            load_paths = local_pretrained_dir

        # 1. Load the autoencoder model which will be used to decode the latents into image space. 
        self.vae = AutoencoderKL.from_pretrained(load_paths, subfolder="vae", use_auth_token=self.token).to(self.device)

        # 2. Load the tokenizer and text encoder to tokenize and encode the text. 
        self.tokenizer = CLIPTokenizer.from_pretrained(load_paths, subfolder='tokenizer', use_auth_token=self.token)
        self.text_encoder = CLIPTextModel.from_pretrained(load_paths, subfolder='text_encoder', use_auth_token=self.token).to(self.device)
        
        # 3. The UNet model for generating the latents.
        self.unet = UNet2DConditionModel.from_pretrained(load_paths, subfolder="unet", use_auth_token=self.token).to(self.device)
        
        if save_pretrained:
            self.vae.save_pretrained(os.path.join(local_pretrained_dir, 'vae'))
            self.tokenizer.save_pretrained(os.path.join(local_pretrained_dir, 'tokenizer'))
            self.text_encoder.save_pretrained(os.path.join(local_pretrained_dir, 'text_encoder'))
            self.unet.save_pretrained(os.path.join(local_pretrained_dir, 'unet'))

        self.scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=self.num_train_timesteps)
        if v2:
            self.scheduler = DPMSolverMultistepScheduler.from_config('stabilityai/stable-diffusion-2', subfolder="scheduler")
        self.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device) 

        print(f'[INFO] loaded stable diffusion!')

    def get_text_embeds(self, prompt, negative_prompt):
        # prompt, negative_prompt: [str]
        if self.v2:
            print("[INFO] using Stable Diffusion v2, loading new text prompt...")
            import requests
            response = requests.post("https://fffiloni-prompt-converter.hf.space/run/predict", json={
            "data": [
                prompt,
                "best", # use the best text converter
            ]}).json()
            prompt = response["data"]
            print(f'[INFO] loaded new text prompt: {prompt}!')

        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')

        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings


    def train_step(self, text_embeddings, pred_rgb, guidance_scale=100, gradient_weight=1, shape=(512, 512), mask=None):
        pred_rgb_512 = F.interpolate(pred_rgb, shape, mode='bilinear', align_corners=False)
        
        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)

        latents, posterior = self.encode_imgs(pred_rgb_512)
        #kl_loss = posterior.kl() * 1e-3
        #kl_loss.backward(retain_graph=True)
        
        latents = latents.clamp(-1, 1)
        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)


        w = (1 - self.alphas_cumprod[t])
        grad = w * (noise_pred - noise)
        grad = grad * gradient_weight
        latents.backward(gradient=grad, retain_graph=True)

        return 0 # dummy loss value

    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8), device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        with torch.autocast('cuda'):
            for i, t in enumerate(self.scheduler.timesteps):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']
        
        return latents

    def decode_latents(self, latents):

        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            imgs = self.vae.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        
        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215

        return latents, posterior

    def prompt_to_img(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if isinstance(prompts, str):
            prompts = [prompts]
        
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(prompts, negative_prompts) # [2, 77, 768]

        # Text embeds -> img latents
        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale) # [1, 4, 64, 64]
        
        # Img latents -> imgs
        imgs = self.decode_latents(latents) # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        return imgs





