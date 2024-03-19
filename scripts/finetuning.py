"""
Finetuning a diffusion model using StylEx counterfactual.
"""
################ general import ################
import torch as th
import torchvision.utils as vutils
from torch.optim import AdamW
import blobfile as bf
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image

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
from sdg.misc import set_random_seed

############## stylex counterfactual ##############
from StylEx256.change_style import change_image
from StylEx256.change_style import create_latent


#####################


"""
main code
"""
def main():

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
    learn_sigma=True,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
    )
    
    model.to("cuda")
    model.eval()
    
    mp_trainer = MixedPrecisionTrainer(model=model, use_fp16=False,fp16_scale_growth=1e-3)
    opt = AdamW(mp_trainer.master_params, lr=3e-4, weight_decay=0.0)
    
    
    logger.log("creating data loader...")
    original_data = load_data(
        data_dir="../ref/counterfactual_dataset/original_images",
        #data_dir="../ref/ref_ffhq",
        batch_size=1,
        image_size=256,
        class_cond=False,
        deterministic=True,
        random_crop=False,
        random_flip=False
    )
    
    counterfactual_data = load_data(
        data_dir="../ref/counterfactual_dataset/counterfactual_images",
        batch_size=80,
        image_size=256,
        class_cond=False,
        deterministic=True,
        random_crop=False,
        random_flip=False
    )
    
    
    ############### load checkpoint #############
    checkpoint = "../../drive/MyDrive/ffhq_p2.pt"
    
    logger.log(f"loading model from checkpoint: {checkpoint}...")
    model.load_state_dict(
    th.load(checkpoint, map_location="cuda")
    )
    """model.load_state_dict(
        dist_util.load_state_dict(
          checkpoint, map_location="cuda"
        )
    )
    
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

    shape = (1,3,256,256)

    with th.cuda.amp.autocast(True): 
      for b in original_data:
        #print(b[0].shape)
        #print(b[1])
        image = b[0].to("cuda")
        vutils.save_image(image, '../latents/batch_original_images.png', nrow=80, normalize=True)
        ################ precompute latents #####################
        #t = th.tensor([0] * 8, device="cuda")

        t = th.tensor([999]*1, device="cuda")
        latent = diffusion.ddim_reverse_sample_loop(model,shape=shape,noise=image, t, clip_denoised=False,denoised_fn=None,model_kwargs=None,device="cuda",progress=True,eta=0.0)
        #latent = diffusion.q_sample(image, th.tensor(999).to("cuda"), noise=None)
        
        #img_lat_pairs.append([b[0], x_reversed.detach(), latent.detach()])
        
        # Salva il batch di immagini latenti tutte insieme in un file per vederle
        vutils.save_image(latent["sample"], '../latents/batch_noise.png', nrow=80, normalize=True)
        break


    #t=th.tensor(1000)
    logger.log("start the sampling")
    with th.cuda.amp.autocast(True):
        x_reversed = diffusion.ddim_sample_loop(model,shape=shape,noise=latent["sample"],clip_denoised=False,denoised_fn=None,cond_fn=None,model_kwargs=None,device="cuda",progress=True,eta=0.0)
        #x_reversed = diffusion.p_sample_loop(
                      model,
                      (1, 3, 256,256),
                      noise=None,
                      clip_denoised=False,
                      model_kwargs={},
                      cond_fn=None,
                      device='cuda',
                      progress = True
                  )
    logger.log("sampling done")
    #save latent in the right format for the traning
    vutils.save_image(x_reversed, '../latents/batch_images_reversed.png', nrow=80, normalize=True)
    
    #################### create counterfactual #########################
    dlatents = create_latent(x_reversed)
    generate_counterfactual(dlatents)


if __name__ == "__main__":
    main()
