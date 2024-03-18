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

def generate_counterfactual(our_dlat):
  expanded_dlatent_tmp = torch.tile(our_dlat.unsqueeze(1),[1, num_layers, 1])
  svbg, _, _ = generator.synthesis.style_vector_calculator(expanded_dlatent_tmp)
  #print(len(svbg))
  result_image = np.zeros((resolution, 2 * resolution, 3), np.uint8)
  images_out = generator.synthesis.image_given_dlatent(expanded_dlatent_tmp, svbg)
  images_out = torch.maximum(torch.minimum(images_out, torch.Tensor([1])), torch.Tensor([-1]))
  result = classifier(images_out)
  base_image = images_out.permute(0, 2, 3, 1)

  #plt.imshow(base_image[0].detach().numpy())

  #####
  class_index = 0
  #sindex = 5300         #Lentiggini
  sindex = 3301         #Occhiali
  #sindex = 3199          #Capelli bianchi
  #sindex = 3921
  shift_sign = "1"
  wsign_index = int(shift_sign)
  shift_size =  3
  #####


  change_image, change_prob, svbg_new = (
      generate_change_image_given_dlatent(our_dlat.detach(), generator, classifier,
                                          class_index, sindex,
                                          style_min[sindex], style_max[sindex],
                                          wsign_index, shift_size,
                                          label_size))

  #print(change_image.shape)

  base_image = torch.from_numpy((base_image[0].numpy() * 127.5 + 127.5).astype(np.uint8)).unsqueeze(0)
  change_image = torch.from_numpy((change_image[0].numpy() * 127.5 + 127.5).astype(np.uint8)).unsqueeze(0)

  """
  #for i in range(7):
  #  print(svbg[i][0] - svbg_new[i][0])
  fig, axes = plt.subplots(1, 2)
  image_np = base_image[0].detach().numpy()
  axes[0].imshow(image_np)
  axes[0].axis('off')
  image_np2 = change_image[0].detach().numpy()
  axes[1].imshow(image_np2)
  axes[1].axis('off')
  plt.show()
  """


  base_image = base_image[0].permute(2, 0, 1).float() / 255.0
  change_image = change_image[0].permute(2, 0, 1).float() / 255.0

  save_image(base_image, f"../gdrive/MyDrive/dataset3/original/original.png")
  save_image(change_image, f"../gdrive/MyDrive/dataset3/counterfactual/counterfactual.png")




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
    steps=3000,
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

    """with th.cuda.amp.autocast(True): 
      for b in original_data:
        #print(b[0].shape)
        #print(b[1])
        image = b[0].to("cuda")
        vutils.save_image(image, '../latents/batch_original_images.png', nrow=80, normalize=True)
        ################ precompute latents #####################
        #t = th.tensor([0] * 8, device="cuda")

        t = th.tensor([2999]*1, device="cuda")
        #latent = diffusion.ddim_reverse_sample(model, image, t, clip_denoised=False,denoised_fn=None,model_kwargs=None)
        latent = diffusion.q_sample(image, th.tensor(999).to("cuda"), noise=None)
        
        #img_lat_pairs.append([b[0], x_reversed.detach(), latent.detach()])
        
        # Salva il batch di immagini latenti tutte insieme in un file per vederle
        #vutils.save_image(latent, '../latents/batch_images.png', nrow=80, normalize=True)
        break"""

    set_random_seed(2, by_rank=True)


    #t=th.tensor(1000)
    shape = (1,3,256,256)
    logger.log("start the sampling")
    with th.cuda.amp.autocast(True):
        #x_reversed = diffusion.ddim_sample_loop(model,shape=shape,noise=latent["sample"],clip_denoised=False,denoised_fn=None,cond_fn=None,model_kwargs=None,device="cuda",progress=True,eta=0.0)
        x_reversed = diffusion.p_sample_loop(
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
