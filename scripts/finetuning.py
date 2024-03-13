"""
Finetuning a diffusion model using StylEx counterfactual.
"""
################ general import ################
import torch as th
import torchvision.utils as vutils
from torch.optim import AdamW
import blobfile as bf

############# diffusion import #################
import argparse

from sdg import dist_util, logger
from sdg.image_dataset import load_data
from sdg.resample import create_named_schedule_sampler
from sdg.script_util import (
    create_model,
    create_gaussian_diffusion,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from sdg.fp16_util import MixedPrecisionTrainer
from sdg.gaussian_diffusion import GaussianDiffusion

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
    model = create_model(
    image_size=256,
    num_channels=128,
    num_res_blocks=1,
    channel_mult="",
    learn_sigma=True,
    class_cond=False,
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=1,
    num_head_channels=64,
    num_heads_upsample=-1,
    use_scale_shift_norm=True,
    dropout=0.0,
    resblock_updown=True,
    use_fp16=False,
    use_new_attention_order=False,
    )

    diffusion = create_gaussian_diffusion(
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
    )

    model.to("cuda")
    
    mp_trainer = MixedPrecisionTrainer(model=model, use_fp16=False,fp16_scale_growth=1e-3)
    opt = AdamW(mp_trainer.master_params, lr=3e-4, weight_decay=0.0)
    
    
    logger.log("creating data loader...")
    original_data = load_data(
      data_dir="../ref/counterfactual_dataset/original_images",
      #data_dir="../ref/ref_ffhq",
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
    checkpoint = "../../drive/MyDrive/ffhq_p2.pt"

    logger.log(f"loading model from checkpoint: {checkpoint}...")
    model.load_state_dict(
      dist_util.load_state_dict(
          checkpoint, map_location="cuda"
      )
    )
    """
    opt_checkpoint = bf.join(
            bf.dirname(checkpoint), "opt.pt"
        )
    if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location="cuda"
            )
            opt.load_state_dict(state_dict)
    """
    #############################################

    schedule_sampler = create_named_schedule_sampler("uniform", diffusion)

    img_lat_pairs = [] # to save x_original, x_reversed, x_latent

    for b in original_data:
        #print(b[0].shape)
        #print(b[1])
        image = b[0].to("cuda")
        ################ precompute latents #####################
        latent = diffusion.q_sample(image, th.tensor(999).to("cuda"), noise=None)

        #x_reversed = diffusion.ddim_sample_loop(model,latent,shape = (80, 3, 256, 256),noise=None,device="cuda",progress=False,t=th.tensor(999),clip_denoised=False,denoised_fn=None,cond_fn=None,model_kwargs=None,eta=0.0)
        
        #img_lat_pairs.append([b[0], x_reversed.detach(), latent.detach()])
        
        # Salva il batch di immagini latenti tutte insieme in un file per vederle
        #vutils.save_image(latent, '../latents/batch_images.png', nrow=80, normalize=True)
        break

    t=th.tensor(999)
    shape = (80,3,256,256)
    x_reversed = diffusion.ddim_sample_loop(model,shape=shape,noise=latent,clip_denoised=False,denoised_fn=None,cond_fn=None,model_kwargs=None,device="cuda",progress=False,eta=0.0)
    #save latent in the right format for the traning
    

    #################### training #########################
  

if __name__ == "__main__":
    main()
