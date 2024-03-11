"""
Finetuning a diffusion model using StylEx counterfactual.
"""
################ general import ################
import torch
import torchvision.utils as vutils

############# diffusion import #################
import argparse

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from .fp16_util import MixedPrecisionTrainer

############## stylex counterfactual ##############





"""
main code
"""
def main():
  
    model, diffusion = create_model_and_diffusion(
      **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to("cuda")
    
    self.mp_trainer = MixedPrecisionTrainer(model=model, use_fp16=False,fp16_scale_growth=1e-3)
    opt = AdamW(self.mp_trainer.master_params, lr=3e-4, weight_decay=0.0)
    opt.to("cuda")
    
    logger.log("creating data loader...")
    original_data = load_data(
      data_dir="../ref/counterfactual_dataset/original_images,
      batch_size=80,
      image_size=256,
      class_cond=False,
    )

    counterfactual_data = load_data(
          data_dir="../ref/counterfactual_dataset/counterfactual_images,
          batch_size=80,
          image_size=256,
          class_cond=False,
        )
    
    
    ############### load checkpoint #############
    checkpoint = "../models/ffhq_p2.pt"
    logger.log(f"loading model from checkpoint: {checkpoint}...")
    model.load_state_dict(
      dist_util.load_state_dict(
          checkpoint, map_location="cuda"
      )
    )
    
    opt.load_state_dict(
      dist_util.load_state_dict(
          checkpoint, map_location="cuda"
      )
    )
    #############################################

    schedule_sampler = create_named_schedule_sampler("uniform", diffusion)

    ################ precompute latents #####################
    latent = q_sample(self, original_data, 1000, noise=None)

    # Salva il batch di immagini in un file per iterarle nel training
    vutils.save_image(tensor_img_batch, '../latents/batch_images.png', nrow=80, normalize=True)

    #################### training #########################
  





def create_argparser():
  defaults = dict(
      data_dir="",
      schedule_sampler="uniform",
      lr=1e-4,
      weight_decay=0.0,
      lr_anneal_steps=0,
      batch_size=1,
      microbatch=-1,  # -1 disables microbatches
      ema_rate="0.9999",  # comma-separated list of EMA values
      log_interval=10,
      save_interval=10000,
      resume_checkpoint="",
      use_fp16=False,
      fp16_scale_growth=1e-3,
  )
  defaults.update(model_and_diffusion_defaults())
  parser = argparse.ArgumentParser()
  add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
