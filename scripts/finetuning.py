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
from tqdm.auto import tqdm 
import os 
from torch import nn
 
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
from sdg.misc import requires_grad, set_random_seed 
from sdg.clip_guidance import CLIP_gd 
 
############## stylex counterfactual ############## 
#from StylEx256.change_style import change_image 
#from StylEx256.change_style import create_latent 
 
 
##################### CLIP parameters ############ 
clip_path = "../../drive/MyDrive/clip_ffhq.pt" 
args = argparse.Namespace(image_size=256, finetune_clip_layer='all') 
clip_ft = CLIP_gd(args) 
clip_ft.load_state_dict(th.load(clip_path, map_location='cpu')) 
clip_ft.eval() 
clip_ft = clip_ft.cuda() 
############################################### 
 
#compute cos distance 
def cos_distance(source, target): 
  source[-1] = source[-1] / source[-1].norm(dim=-1, keepdim=True) 
  target[-1] = target[-1] / target[-1].norm(dim=-1, keepdim=True) 
  loss = (source[-1] * target[-1]).sum(1) 
  return loss 
 
#compute the image loss between countefactual and reversed using cos distance 
def direction_loss(x_reversed, x_counterfactual, t): 
  with th.enable_grad(): 
    with th.cuda.amp.autocast(True):  
      #embedding image reversed 
      x_reversed = x_reversed.half()  
      embedding_reversed = clip_ft.encode_image_list(x_reversed, t) 
 
      #embedding counterfactual 
      x_counterfactual = x_counterfactual.half() 
      embedding_counterfactual = clip_ft.encode_image_list(x_counterfactual.to("cuda"), t) 
 
      cos_similarity = cos_distance(embedding_reversed, embedding_counterfactual) 
 
      #return th.autograd.grad(cos_similarity, x_reversed)[0] 
      return cos_similarity 

"""
class IDLoss(nn.Module):
    def __init__(self, use_mobile_id=False):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(MODEL_PATHS['ir_se50']))

        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()

    def extract_feats(self, x):
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, x, x_hat):
        n_samples = x.shape[0]
        x_feats = self.extract_feats(x)
        x_feats = x_feats.detach()

        x_hat_feats = self.extract_feats(x_hat)
        losses = []
        for i in range(n_samples):
            loss_sample = 1 - x_hat_feats[i].dot(x_feats[i])
            losses.append(loss_sample.unsqueeze(0))

        losses = torch.cat(losses, dim=0)
        return losses
"""


 
 
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
    opt = AdamW(mp_trainer.master_params, lr=4e-6, weight_decay=0.0) 
     
     
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
        batch_size=1, 
        image_size=256, 
        class_cond=False, 
        deterministic=True, 
        random_crop=False, 
        random_flip=False,
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
    counterfactual_array = [] 
 
    shape = (1,3,256,256) 
 
    file_path = '../latents/precomputed_latents' 
 
    if os.path.exists(file_path): 
      logger.log("loading the precomputed latents") 
      # Carica il tensore dal file 
      img_lat_pairs = th.load(file_path) 
    else: 
        logger.log("file not exist, compute the latents") 
        with th.cuda.amp.autocast(True):  
          for b in original_data: 
 
            image = b[0].to("cuda") 
            vutils.save_image(image, '../latents/batch_original_images.png', nrow=80, normalize=True) 
            ################ precompute latents ##################### 
            #t = th.tensor([0] * 8, device="cuda") 
 
            latent = diffusion.ddim_reverse_sample_loop(model,shape=shape,noise=image, clip_denoised=False,denoised_fn=None,model_kwargs=None,device="cuda",progress=True,eta=0.0) 
            #latent = diffusion.q_sample(image, th.tensor(999).to("cuda"), noise=None) 
 
            x_reversed = diffusion.ddim_sample_loop(model,shape=shape,noise=latent,clip_denoised=False,denoised_fn=None,cond_fn=None,model_kwargs=None,device="cuda",progress=True,eta=0.0) 
             
            img_lat_pairs.append([b[0], x_reversed.detach(), latent.detach()]) 
             
            # Salva il batch di immagini latenti tutte insieme in un file per vederle 
            vutils.save_image(latent, '../latents/batch_noise.png', nrow=80, normalize=True) 
            break 
     
          # Salva il tensore nel file 
          th.save( img_lat_pairs, file_path) 
 
 
    logger.log("saving counterfactuals") 
    #save counterfactual images in array of tensor 
    with th.cuda.amp.autocast(True):  
      for a in counterfactual_data: 
        image = a[0].to("cuda") 
        counterfactual_array.append(a[0])  
        break       
 
     
    """logger.log("start the sampling") 
    with th.cuda.amp.autocast(True): 
        x_reversed = diffusion.ddim_sample_loop(model,shape=shape,noise=latent,clip_denoised=False,denoised_fn=None,cond_fn=None,model_kwargs=None,device="cuda",progress=True,eta=0.0) 
        x_reversed = diffusion.p_sample_loop( 
                      model, 
                      (3, 3, 256,256), 
                      noise=latent, 
                      clip_denoised=False, 
                      model_kwargs={}, 
                      cond_fn=None, 
                      device='cuda', 
                      progress = True 
                  )""" 
    #save latent in the right format for the traning 
    #vutils.save_image(x_reversed, '../latents/batch_images_reversed.png', nrow=80, normalize=True) 
     
    #################### finetuning ######################### 
    logger.log("start finetuning") 
    n_iter = 10 
    for epoch in range(n_iter):#epoch 
      for step in range(len(img_lat_pairs)): 
        print(step) 
        model.train() 
        opt.zero_grad() 
        with tqdm(total=1, desc=f"step iteration") as progress_bar: 
          with th.cuda.amp.autocast(True): 
            # Creazione di una matrice con 1000 righe e una colonna
            shapes = (1000, 1)
            #t = th.arange(1, 1001, device="cuda").reshape(shapes)
            indices = list(range(1000))[::-1]
            im = img_lat_pairs[step][2].half().to("cuda")
            #image = im.requires_grad_(True)
            image = im.clone()
            # Abilita il rilevamento delle anomalie
            th.autograd.set_detect_anomaly(True)
            for i in indices:           
                t = th.tensor([i] * 1, device='cuda')
                print("EZIO "+str(t))
                #opt.zero_grad() 
                #t = th.tensor([100] * shape[0], device="cuda")
                x = diffusion.ddim_sample(model,x=image,t=t,clip_denoised=False,denoised_fn=None,cond_fn=None,model_kwargs=None,eta=0.0) 

                #x_reversed = diffusion.ddim_sample_loop(model,shape=shape,noise=img_lat_pairs[step][2].half().to("cuda"),clip_denoised=False,denoised_fn=None,cond_fn=None,model_kwargs=None,device="cuda",progress=False,eta=0.0) 
                #x_reversed = x_reversed.requires_grad_(True) 
                y = x["sample"]
                #x_reversed = x_reversed
                #t = th.tensor([100] * shape[0], device="cuda")
                #model_output = model(img_lat_pairs[step][2].half().to("cuda"), t, {})

                #model_output, model_var_values = th.split(model_output, 3, dim=1)
                #print(model_output.shape)
              
                counterfactual_array[step] = counterfactual_array[step].to('cuda')
                #count = counterfactual_array[step].requires_grad_(True) 
                progress_bar.update(1) 
        
                #save image 
                vutils.save_image(y,'../latents/batch_images_reversed.png', nrow=80, normalize=True) 
                vutils.save_image(counterfactual_array[step],'../latents/counterfactual.png', nrow=80, normalize=True) 
                """
                loss = x_reversed.mean()
                grads = th.autograd.grad(loss, model.parameters(), retain_graph=True,allow_unused=True)
                opt.zero_grad()
                for param, grad in zip(model.parameters(), grads):
                    param.grad = grad
                opt.step()
                """

                #compute cos distance 
                #loss = direction_loss(y, counterfactual_array[step], t)
                loss_clip = -th.log((2 - direction_loss(y, counterfactual_array[step], t)) / 2)
                loss_l1 = nn.L1Loss()(y, counterfactual_array[step])
                #loss_id = th.mean(id_loss(y, counterfactual_array[step]))
                loss = 3 * loss_clip + 1 * loss_l1# + self.args.id_loss_w * loss_id

                loss.backward() 

                print("MATTO")
                print(loss)

                opt.step()

                for p in model.parameters():
                    p.grad = None

                image = y.detach().clone()
                #image = y

            """
            # Verify gradients 
            for name, param in model.named_parameters(): 
                if param.grad is not None: 
                    print(f'Parameter {name}: Gradients exist') 
                else: 
                    print(f'Parameter {name}: No gradients') 
            """

         
      save_name = "../latents/nuovi.pt" 
      th.save(model.state_dict(), save_name) 
      logger.log(f'Model {save_name} is saved.') 
 
 
 
if __name__ == "__main__":
    main()
