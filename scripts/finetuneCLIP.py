import time
from glob import glob
from tqdm import tqdm
import os
import numpy as np
import cv2
from PIL import Image
import torch
from torch import nn
import torchvision.utils as tvu

from models.ddpm.diffusion import DDPM
from models.improved_ddpm.script_util import i_DDPM
from utils.text_dic import SRC_TRG_TXT_DIC
from utils.diffusion_utils import get_beta_schedule, denoising_step
from losses import id_loss
from losses.clip_loss import CLIPLoss
from datasets.data_utils import get_dataset, get_dataloader
from configs.paths_config import DATASET_PATHS, MODEL_PATHS, HYBRID_MODEL_PATHS, HYBRID_CONFIG
from datasets.imagenet_dic import IMAGENET_DIC
from utils.align_utils import run_alignment

###########################################nostre import #####################
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
from sdg.clip_guidance import CLIP_gd 
#############################################################################

class DiffusionCLIP(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = torch.device(
                "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps
        )
        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        posterior_variance = betas * \
                             (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.model_var_type == "fixedlarge"
        self.logvar = np.log(np.append(posterior_variance[1], betas[1:]))

    def clip_finetune(self):

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
        learn_sigma = True
        print("Improved diffusion Model loaded.")
        
        model.load_state_dict( 
          th.load(checkpoint, map_location="cuda") 
        ) 
        model.to("cuda") 
        #model = torch.nn.DataParallel(model)

        # ----------- Optimizer and Scheduler -----------#
        print("loading optimizer and scheduler")
        mp_trainer = MixedPrecisionTrainer(model=model, use_fp16=False,fp16_scale_growth=1e-3) 
        optim_ft = AdamW(mp_trainer.master_params, lr=4e-6, weight_decay=0.0) 
        schedule_ft = create_named_schedule_sampler("uniform", diffusion)
        # ----------- Loss -----------#
        print("Loading losses")

        # ----------- Precompute Latents -----------#
        print("Prepare identity latent")
        seq_inv = np.linspace(0, 1, self.args.n_inv_step) * self.args.t_0
        seq_inv = [int(s) for s in list(seq_inv)]
        seq_inv_next = [-1] + list(seq_inv[:-1])

        n = self.args.bs_train
        data_category ="FFHQ"
        img_lat_pairs_dic = {}
        for mode in ['train', 'test']:
            img_lat_pairs = []
            pairs_path = os.path.join('precomputed/',
                                      f'{data_category}_{mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth')
            print(pairs_path)
            if os.path.exists(pairs_path):
                print(f'{mode} pairs exists')
                img_lat_pairs_dic[mode] = torch.load(pairs_path)
                for step, (x0, x_id, x_lat) in enumerate(img_lat_pairs_dic[mode]):
                    tvu.save_image((x0 + 1) * 0.5, os.path.join(self.args.image_folder, f'{mode}_{step}_0_orig.png'))
                    tvu.save_image((x_id + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                  f'{mode}_{step}_1_rec_ninv{self.args.n_inv_step}.png'))
                    if step == self.args.n_precomp_img - 1:
                        break
                continue
            else:
                train_dataset, test_dataset = get_dataset(self.config.data.dataset, DATASET_PATHS, self.config)
                loader_dic = get_dataloader(train_dataset, test_dataset, bs_train=self.args.bs_train,
                                            num_workers=self.config.data.num_workers)
                loader = loader_dic[mode]

            for step, img in enumerate(loader):
                x0 = img.to(self.config.device)
                tvu.save_image((x0 + 1) * 0.5, os.path.join(self.args.image_folder, f'{mode}_{step}_0_orig.png'))

                x = x0.clone()
                model.eval()
                with torch.no_grad():
                    with tqdm(total=len(seq_inv), desc=f"Inversion process {mode} {step}") as progress_bar:
                        for it, (i, j) in enumerate(zip((seq_inv_next[1:]), (seq_inv[1:]))):
                            t = (torch.ones(n) * i).to(self.device)
                            t_prev = (torch.ones(n) * j).to(self.device)

                            x = denoising_step(x, t=t, t_next=t_prev, models=model,
                                               logvars=self.logvar,
                                               sampling_type='ddim',
                                               b=self.betas,
                                               eta=0,
                                               learn_sigma=learn_sigma)

                            progress_bar.update(1)
                    x_lat = x.clone()
                    tvu.save_image((x_lat + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                   f'{mode}_{step}_1_lat_ninv{self.args.n_inv_step}.png'))

                    with tqdm(total=len(seq_inv), desc=f"Generative process {mode} {step}") as progress_bar:
                        for it, (i, j) in enumerate(zip(reversed((seq_inv)), reversed((seq_inv_next)))):
                            t = (torch.ones(n) * i).to(self.device)
                            t_next = (torch.ones(n) * j).to(self.device)

                            x = denoising_step(x, t=t, t_next=t_next, models=model,
                                               logvars=self.logvar,
                                               sampling_type=self.args.sample_type,
                                               b=self.betas,
                                               learn_sigma=learn_sigma)
                            progress_bar.update(1)

                    img_lat_pairs.append([x0, x.detach().clone(), x_lat.detach().clone()])
                tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                           f'{mode}_{step}_1_rec_ninv{self.args.n_inv_step}.png'))
                if step == self.args.n_precomp_img - 1:
                    break

            img_lat_pairs_dic[mode] = img_lat_pairs
            pairs_path = os.path.join('precomputed/',
                                      f'{self.config.data.category}_{mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth')
            torch.save(img_lat_pairs, pairs_path)

        # ----------- Finetune Diffusion Models -----------#
        print("Start finetuning")
        print(f"Sampling type: {self.args.sample_type.upper()} with eta {self.args.eta}")
        if self.args.n_train_step != 0:
            seq_train = np.linspace(0, 1, self.args.n_train_step) * self.args.t_0
            seq_train = [int(s) for s in list(seq_train)]
            print('Uniform skip type')
        else:
            seq_train = list(range(self.args.t_0))
            print('No skip')
        seq_train_next = [-1] + list(seq_train[:-1])

        seq_test = np.linspace(0, 1, self.args.n_test_step) * self.args.t_0
        seq_test = [int(s) for s in list(seq_test)]
        seq_test_next = [-1] + list(seq_test[:-1])

        for src_txt, trg_txt in zip(self.src_txts, self.trg_txts):
            print(f"CHANGE {src_txt} TO {trg_txt}")
            model.module.load_state_dict(init_ckpt)
            optim_ft.load_state_dict(init_opt_ckpt)
            scheduler_ft.load_state_dict(init_sch_ckpt)
            clip_loss_func.target_direction = None

            # ----------- Train -----------#
            for it_out in range(self.args.n_iter):
                exp_id = os.path.split(self.args.exp)[-1]
                save_name = f'checkpoint/{exp_id}_{trg_txt.replace(" ", "_")}-{it_out}.pth'
                if self.args.do_train:
                    if os.path.exists(save_name):
                        print(f'{save_name} already exists.')
                        model.module.load_state_dict(torch.load(save_name))
                        continue
                    else:
                        for step, (x0, x_id, x_lat) in enumerate(img_lat_pairs_dic['train']):
                            model.train()
                            time_in_start = time.time()

                            optim_ft.zero_grad()
                            x = x_lat.clone()

                            with tqdm(total=len(seq_train), desc=f"CLIP iteration") as progress_bar:
                                for t_it, (i, j) in enumerate(zip(reversed(seq_train), reversed(seq_train_next))):
                                    t = (torch.ones(n) * i).to(self.device)
                                    t_next = (torch.ones(n) * j).to(self.device)

                                    x = denoising_step(x, t=t, t_next=t_next, models=model,
                                                       logvars=self.logvar,
                                                       sampling_type=self.args.sample_type,
                                                       b=self.betas,
                                                       eta=self.args.eta,
                                                       learn_sigma=learn_sigma)

                                    progress_bar.update(1)

                            loss_clip = (2 - clip_loss_func(x0, src_txt, x, trg_txt)) / 2
                            loss_clip = -torch.log(loss_clip)
                            loss_id = torch.mean(id_loss_func(x0, x))
                            loss_l1 = nn.L1Loss()(x0, x)
                            loss = self.args.clip_loss_w * loss_clip + self.args.id_loss_w * loss_id + self.args.l1_loss_w * loss_l1
                            loss.backward()

                            optim_ft.step()
                            print(f"CLIP {step}-{it_out}: loss_id: {loss_id:.3f}, loss_clip: {loss_clip:.3f}")

                            if self.args.save_train_image:
                                tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                           f'train_{step}_2_clip_{trg_txt.replace(" ", "_")}_{it_out}_ngen{self.args.n_train_step}.png'))
                            time_in_end = time.time()
                            print(f"Training for 1 image takes {time_in_end - time_in_start:.4f}s")
                            if step == self.args.n_train_img - 1:
                                break
                              
                          torch.save(model.state_dict(), save_name)
                        print(f'Model {save_name} is saved.')
                        scheduler_ft.step()
