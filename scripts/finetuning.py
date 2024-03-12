"""
Finetuning a diffusion model using StylEx counterfactual.
"""
################ general import ################
import torch as th
import torchvision.utils as vutils

############# diffusion import #################
import argparse

from sdg import dist_util, logger
from sdg.image_datasets import load_data
from sdg.resample import create_named_schedule_sampler
from sdg.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from sdg.fp16_util import MixedPrecisionTrainer

############## stylex counterfactual ##############





"""
main code
"""
def main():

    """
    model, diffusion = create_model_and_diffusion(
    #**args_to_dict(args, model_and_diffusion_defaults().keys())
    image_size=256,class_cond=False,learn_sigma=True,num_channels=256, num_res_blocks=3,channel_mult="",num_heads=4,num_head_channels=-1,
    num_heads_upsample=-1,attention_resolutions="32,16,8",dropout=0.0,noise_schedule="linear",use_checkpoint=False,use_scale_shift_norm=True,
    resblock_updown=False,use_fp16=False,use_new_attention_order=False,'text_cond', diffusion_steps=4000, timestep_respacing, use_kl=False, 'predict_xstart',
    rescale_timesteps, rescale_learned_sigmas
    )
    """
    model, diffusion = model_and_diffusion_defaults()

    model.to("cuda")
    
    self.mp_trainer = MixedPrecisionTrainer(model=model, use_fp16=False,fp16_scale_growth=1e-3)
    opt = AdamW(self.mp_trainer.master_params, lr=3e-4, weight_decay=0.0)
    opt.to("cuda")
    
    logger.log("creating data loader...")
    original_data = load_data(
      data_dir="../ref/counterfactual_dataset/original_images",
      batch_size=80,
      image_size=256,
      class_cond=False,
    )

    counterfactual_data = load_data(
          data_dir="../ref/counterfactual_dataset/counterfactual_images",
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
  

if __name__ == "__main__":
    main()
