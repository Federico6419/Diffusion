############################
import torch
import numpy as np
from matplotlib import pyplot as plt
import os
import IPython.display
import pickle
import requests
import collections

from tfrecord.torch.dataset import TFRecordDataset

from StylEx256.mobilenet_pytorch import MobileNetV1
from StylEx256.stylegan2_pytorch.training import networks

from StylEx256.utils import *

CLASSIFIER_PATH = '../StylEx256/models/classifier.pth'
DISCRIMINATOR_PATH = '../StylEx256/models/discriminator/discriminator.pth'
ENCODER_PATH = '../StylEx256/models/encoder/encoder.pth'
GENERATOR_PATH = '../StylEx256/models/generator/generator.pth'
##################################

##################################
torch.manual_seed(0)

classifier = MobileNetV1()
classifier.load_state_dict(torch.load(CLASSIFIER_PATH))
classifier.eval()
classifier.to('cuda')

generator = load_torch_generator(pth_file=GENERATOR_PATH)
generator.eval()
#generator.to('cuda')

encoder = load_torch_encoder(pth_file=ENCODER_PATH)
encoder.eval()
#encoder.to('cuda')

discriminator = load_torch_discriminator(pth_file=DISCRIMINATOR_PATH)
discriminator.eval()
discriminator.to('cuda')
print("")
####################################

#####################################
#Global variables

num_layers = 14
label_size = 2
resolution = 256
#####################################

#####################################
#@title Load the precomputed dlatents (already concatenated to the labels)
latents_file = open("../StylEx256/data/saved_dlantents.pkl",'rb')
dlatents = pickle.load(latents_file)


#@title Load effect data from the tfrecord {form-width: '20%'}
data_path = 'StylEx256/data/examples_1.tfrecord'
num_classes = 2
print(f'Loaded dataset: {data_path}')
index_path = None
description = {"dlatent": "float", "result": "float", "base_prob": "float"}
dataset = TFRecordDataset(data_path, index_path, description)
loader = torch.utils.data.DataLoader(dataset, batch_size=1)

style_change_effect = []
dlatents = []
base_probs = []
for raw_record in iter(loader):
  dlatents.append(
      np.array(raw_record['dlatent']))
  seffect = np.array(
      raw_record['result']).reshape(
          (-1, 2, num_classes))
  style_change_effect.append(seffect.transpose([1, 0, 2]))
  base_probs.append(
      np.array(raw_record['base_prob']))

base_probs = np.array(base_probs)
style_change_effect = np.array(style_change_effect)
dlatents = torch.from_numpy(np.array(dlatents))
expanded_dlatent_tmp = torch.tile(dlatents, [1, num_layers, 1])
W_values, style_change_effect, base_probs = dlatents.squeeze(), style_change_effect.squeeze(), base_probs.squeeze()

style_change_effect = filter_unstable_images(style_change_effect, effect_threshold=2)
all_style_vectors = torch.cat(generator.synthesis.style_vector_calculator(expanded_dlatent_tmp)[1], dim=1).numpy()
style_min = np.min(all_style_vectors, axis=0)
style_max = np.max(all_style_vectors, axis=0)

all_style_vectors_distances = np.zeros((all_style_vectors.shape[0], all_style_vectors.shape[1], 2))
all_style_vectors_distances[:,:, 0] = all_style_vectors - np.tile(style_min, (all_style_vectors.shape[0], 1))
all_style_vectors_distances[:,:, 1] = np.tile(style_max, (all_style_vectors.shape[0], 1)) - all_style_vectors
############################à

#####################################
def generate_change_image_given_dlatent(
    dlatent: np.ndarray,
    generator: networks.Generator,
    classifier: Optional[MobileNetV1],
    class_index: int,
    sindex: int,
    s_style_min: float,
    s_style_max: float,
    style_direction_index: int,
    shift_size: float,
    label_size: int = 2,
    num_layers: int = 14
) -> Tuple[np.ndarray, float, float]:
  """Modifies an image given the dlatent on a specific S-index.

  Args:
    dlatent: The image dlatent, with sape [dlatent_size].
    generator: The generator model. Either StyleGAN or GLO.
    classifier: The classifier to visualize.
    class_index: The index of the class to visualize.
    sindex: The specific style index to visualize.
    s_style_min: The minimal value of the style index.
    s_style_max: The maximal value of the style index.
    style_direction_index: If 0 move s to it's min value otherwise to it's max
      value.
    shift_size: Factor of the shift of the style vector.
    label_size: The size of the label.

  Returns:
    The image after the style index modification, and the output of
    the classifier on this image.
  """
  expanded_dlatent_tmp = torch.tile(dlatent.unsqueeze(1),[1, num_layers, 1])
  network_inputs = generator.synthesis.style_vector_calculator(expanded_dlatent_tmp)

  style_vector = torch.cat(generator.synthesis.style_vector_calculator(expanded_dlatent_tmp)[1], dim=1).numpy()
  orig_value = style_vector[0, sindex]
  target_value = (s_style_min if style_direction_index == 0 else s_style_max)

  if target_value == orig_value:
    weight_shift = shift_size
  else:
    weight_shift = shift_size * (target_value - orig_value)

  layer_idx, in_idx = sindex_to_layer_idx_and_index(network_inputs[1], sindex)

  layer_one_hot = torch.nn.functional.one_hot(torch.Tensor([in_idx]).to(int), network_inputs[1][layer_idx].shape[1])

  network_inputs[1][layer_idx] += (weight_shift * layer_one_hot)
  svbg_new = group_new_style_vec_block(network_inputs[1])

  images_out = generator.synthesis.image_given_dlatent(expanded_dlatent_tmp, svbg_new)
  images_out = torch.maximum(torch.minimum(images_out, torch.Tensor([1])), torch.Tensor([-1]))

  change_image = torch.tensor(images_out.numpy())
  result = classifier(change_image)
  change_prob = nn.Softmax(dim=1)(result).detach().numpy()[0, class_index]
  change_image = change_image.permute(0, 2, 3, 1)

  return change_image, change_prob, svbg_new
#####################################

#####################################
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, image_list, transform=None):
        self.image_list = image_list
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = self.image_list[idx]
        if self.transform:
            image = self.transform(image)
        return image
#####################################

#####################################
def create_latent(image):
  dataset = CustomDataset(image)

  batch_size = 8
  from torch.utils.data import DataLoader
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

  for image in dataloader:
    image=image.to('cuda')
    #image = plt.imread('./data/00019.jpg').transpose(2,0,1)
    #image = image.astype(np.float32) / 255.0
    
    #logits = classifier(torch.from_numpy(image).unsqueeze(0))
    #logits = classifier(image.unsqueeze(0))
    logits = classifier(image)
    #our_dlat = create_dlat_from_img_and_logits(encoder, logits, image[0].detach().cpu().numpy())
    #image = torch.from_numpy(image.detach().cpu().numpy()).unsqueeze(0)
    enc_out = encoder(image.cpu(), 2)
    our_dlat = torch.cat([enc_out.to('cuda'), logits], dim=1)

  return our_dlat
#####################################

#####################################
def change_image(attribute_number, lat_images):  
  print("!!!")
  """
  expanded_dlatent_tmp = torch.tile(lat_images.unsqueeze(1),[1, num_layers, 1]).cpu()
  svbg, _, _ = generator.synthesis.style_vector_calculator(expanded_dlatent_tmp.squeeze(0))
  result_image = np.zeros((resolution, 2 * resolution, 3), np.uint8)
  images_out = generator.synthesis.image_given_dlatent(expanded_dlatent_tmp, svbg)
  images_out = torch.maximum(torch.minimum(images_out, torch.Tensor([1])), torch.Tensor([-1]))
  result = classifier(images_out.to('cuda'))
  base_image = images_out.permute(0, 2, 3, 1)
  """
  
  class_index = 0
  #sindex = attribute_number
  #sindex = 5300         #Lentiggini
  sindex = 3301         #Occhiali
  #sindex = 3199          #Capelli bianchi
  #sindex = 3921
  shift_sign = "1"
  wsign_index = int(shift_sign)
  shift_size =  1

  new_images = []

  for lat in lat_images:
    change_image = generate_change_image_given_dlatent(lat.unsqueeze(0).detach().cpu(), generator, classifier,
                                            class_index, sindex,
                                            style_min[sindex], style_max[sindex],
                                            wsign_index, shift_size,
                                            label_size)

    new_images.append(change_image)

    """
    fig, axes = plt.subplots(1, 2)
    image_np = base_image[0].detach().numpy()
    axes[0].imshow(image_np)
    axes[0].axis('off')
    image_np2 = change_image[0].detach().numpy()
    axes[1].imshow(image_np2)
    axes[1].axis('off')
    plt.show()
    """
    
  new_images = torch.stack(new_images, dim=0).squeeze(1).permute(0, 3, 1, 2)

  embeddings = create_latent(new_images)

  return embeddings
    
#####################################
