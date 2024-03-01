import os
import sys
import h5py
import numpy as np
import shutil
import pandas as pd

from torch.utils.data import DataLoader
import math
import tqdm
import random
import imageio

import multiprocessing
from torchvision.utils import make_grid
from PIL import Image
import ast
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import resize
from torchvision.utils import save_image

import requests
from PIL import ImageDraw
from PIL import ImageFont
from io import BytesIO
import IPython.display
from IPython.display import HTML
import matplotlib.pyplot as plt
from shutil import copyfile
import IPython.display as IPython_display

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

from resnet_classifier import ResNet



model_to_use = "faces_old"



# Threshold should be 501 for the new faces model
stylex_path = None
classifier_name = None
data = None
USE_OLD_ARCHITECTURE = False
hf = None # hdf5 file.
shift_size = 1
threshold_index = 101

if model_to_use == "plant":
    stylex_path = "../pretrained_stylex/models/old_plant_mobilenet/model_260.pt"
    data = "../data/plant_village/all" # Plant dataset
    classifier_name = "../pretrained_stylex/trained_classifiers/mobilenet-64px-plant.pt" # Only use mobilenet for plants
    hf = h5py.File("../pretrained_stylex/precomputed_attfind_files/style_change_records_old_plants.hdf5", 'r')
    threshold_index = 101
    shift_size = 2
    USE_OLD_ARCHITECTURE = True

elif model_to_use == "faces_old":
    stylex_path = "../pretrained_stylex/models/old_faces_gender_mobilenet/model_134.pt"
    data = "../data/Kaggle_FFHQ_Resized_256px/flickrfaceshq-dataset-nvidia-resized-256px/resized" # FFHQ faces dataset
    classifier_name = "../pretrained_stylex/trained_classifiers/resnet-18-64px-gender.pt"  # Use ResNet for all the gender related ones, even the one trained on mobilenet
    hf = h5py.File("Stylex/pretrained_stylex/precomputed_attfind_files/style_change_records_old_faces.hdf5", 'r')
    threshold_index = 101
    shift_size = 2
    USE_OLD_ARCHITECTURE = True

elif model_to_use == "faces_new":
    stylex_path = "../pretrained_stylex/models/new_faces_gender_resnet/model_300.pt"
    data = "../data/Kaggle_FFHQ_Resized_256px/flickrfaceshq-dataset-nvidia-resized-256px/resized" # FFHQ faces dataset
    classifier_name = "../pretrained_stylex/trained_classifiers/resnet-18-64px-gender.pt"  # Use ResNet for all the gender related ones, even the one trained on mobilenet
    hf = h5py.File("../pretrained_stylex/precomputed_attfind_files/style_change_records_new_faces.hdf5", 'r')
    threshold_index = 501
    shift_size = 1
    USE_OLD_ARCHITECTURE = False


if USE_OLD_ARCHITECTURE:
    from stylex_train import StylEx, Dataset, DistributedSampler, MNIST_1vA
    from stylex_train import image_noise, styles_def_to_tensor, make_weights_for_balanced_classes, cycle, default
else:
    from stylex_train_new import StylEx, Dataset, DistributedSampler, MNIST_1vA
    from stylex_train_new import image_noise, styles_def_to_tensor, make_weights_for_balanced_classes, cycle, default



def load_hdf5_results(data_file, name, threshold):
    return np.array(data_file[name])[0:threshold]

def model_loader(stylex_path,
                   classifier_name,
                   image_size,
                   cuda_rank):

    init_stylex = StylEx(image_size=image_size)
    init_stylex.load_state_dict(torch.load(stylex_path)["StylEx"])

    init_classifier = None

    if "mobilenet" in classifier_name.lower():
        init_classifier = MobileNet(classifier_name, cuda_rank=cuda_rank, output_size=2, image_size=image_size)
    elif "resnet" in classifier_name.lower():
        init_classifier = ResNet(classifier_name, cuda_rank=cuda_rank, output_size=2, image_size=image_size)
    else:
        raise NotImplementedError("This classifier is not supported yet, please add support or change the filename to contain MobileNet or ResNet.")
    return init_stylex, init_classifier

def sindex_to_block_idx_and_index(generator, sindex):
    tmp_idx = sindex

    block_idx = None
    idx = None

    for idx, block in enumerate(generator.blocks):
        if tmp_idx < block.num_style_coords:
            block_idx = idx
            idx = tmp_idx
            break
        else:
            tmp_idx = tmp_idx - block.num_style_coords

    return block_idx, idx

def plot_image(tensor, upscale_res=None):
    if upscale_res is not None:
        tensor = resize(tensor, upscale_res)
    grid = make_grid(tensor,nrow=5)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    display(im)



font_path = "./Roboto-Bold.ttf"

# Download font if it doesn't exist
if not os.path.exists(font_path):
  r = requests.get('https://github.com/openmaptiles/fonts/raw/master/roboto/Roboto-Bold.ttf')
  open(font_path, 'wb').write(r.content)

results_folder = './'
threshold_folder = './'
dataset_name = None # for any dataset that is not MNIST
cuda_rank = 0



style_change_effect = load_hdf5_results(hf, "style_change", threshold_index)
W_values = load_hdf5_results(hf, "latents", threshold_index)
base_probs = load_hdf5_results(hf, "base_prob", threshold_index)
all_style_vectors = load_hdf5_results(hf, "style_coordinates", threshold_index)
original_images = load_hdf5_results(hf, "original_images", threshold_index)
discriminator_results = load_hdf5_results(hf, "discriminator", threshold_index)

print(all_style_vectors.shape)

saved_noise = torch.Tensor(np.array(hf["noise"])).cuda(cuda_rank)
style_min = torch.Tensor(np.squeeze(np.array(hf["minima"])))
style_max = torch.Tensor(np.squeeze(np.array(hf["maxima"])))

all_style_vectors_distances = np.zeros((all_style_vectors.shape[0], all_style_vectors.shape[1], 2))
all_style_vectors_distances[:,:, 0] = all_style_vectors - np.tile(style_min, (all_style_vectors.shape[0], 1))
all_style_vectors_distances[:,:, 1] = np.tile(style_max, (all_style_vectors.shape[0], 1)) - all_style_vectors

style_min = style_min.cuda(cuda_rank)
style_max = style_max.cuda(cuda_rank)



num_style_coords = len(style_min)
image_size = original_images.shape[-1]
batch_size = 1



stylex, classifier = model_loader(stylex_path = stylex_path,
                                  classifier_name = classifier_name,
                                  image_size = image_size,
                                  cuda_rank = cuda_rank)



all_labels = np.argmax(base_probs, axis=1)
style_effect_classes = {}
W_classes = {}
style_vectors_distances_classes = {}
all_style_vectors_classes = {}

for img_ind in range(2):

    img_inx = np.array([i for i in range(all_labels.shape[0]) if all_labels[i] == img_ind])
    curr_style_effect = np.zeros((len(img_inx), style_change_effect.shape[1], style_change_effect.shape[2], style_change_effect.shape[3]))
    curr_w = np.zeros((len(img_inx), W_values.shape[1]))
    curr_style_vector_distances = np.zeros((len(img_inx), style_change_effect.shape[2], 2))

    for k, i in enumerate(img_inx):
        curr_style_effect[k, :, :] = style_change_effect[i, :, :, :]
        curr_w[k, :] = W_values[i, :]
        curr_style_vector_distances[k, :, :] = all_style_vectors_distances[i, :, :]

    style_effect_classes[img_ind] = curr_style_effect
    W_classes[img_ind] = curr_w
    style_vectors_distances_classes[img_ind] = curr_style_vector_distances
    all_style_vectors_classes[img_ind] = all_style_vectors[img_inx]
    print(f'Class {img_ind}, {len(img_inx)} images.')




def find_significant_styles(style_change_effect,
                            num_indices,
                            class_index,
                            generator,
                            classifier,
                            all_dlatents,
                            style_min,
                            style_max,
                            max_image_effect = 0.2,
                            label_size = 2,
                            sindex_offset = 0):

    num_images = style_change_effect.shape[0]
    style_effect_direction = np.maximum(0, style_change_effect[:, :, :, class_index].reshape((num_images, -1)))

    images_effect = np.zeros(num_images)
    all_sindices = []
    discriminator_removed = []

    while len(all_sindices) < num_indices:
        next_s = np.argmax(np.mean(style_effect_direction[images_effect < max_image_effect], axis=0))

        all_sindices.append(next_s)
        images_effect += style_effect_direction[:, next_s]
        style_effect_direction[:, next_s] = 0

    return [(x // style_change_effect.shape[2], (x % style_change_effect.shape[2]) + sindex_offset) for x in all_sindices]




label_size_clasifier = 2
num_indices =  20
effect_threshold = 0.5
s_indices_and_signs_dict = {}

for class_index in [0, 1]:
    split_ind =  class_index #1 - class_index
    all_s = style_effect_classes[split_ind]
    all_w = W_classes[split_ind]

    # Find s indicies
    s_indices_and_signs = find_significant_styles(style_change_effect=all_s,
                                                  num_indices=num_indices,
                                                  class_index=class_index,
                                                  generator=stylex.G,
                                                  classifier=classifier,
                                                  all_dlatents=all_w,
                                                  style_min=style_min,
                                                  style_max=style_max,
                                                  max_image_effect=effect_threshold*5,
                                                  label_size=label_size_clasifier,
                                                  sindex_offset=0)

    s_indices_and_signs_dict[class_index] = s_indices_and_signs

sindex_class_0 = [sindex for _, sindex in s_indices_and_signs_dict[0]]
all_sindex_joined_class_0 = [(1 - direction, sindex) for direction, sindex in s_indices_and_signs_dict[1] if sindex not in sindex_class_0]
all_sindex_joined_class_0 += s_indices_and_signs_dict[0]
scores = []

for direction, sindex in all_sindex_joined_class_0:
    other_direction = 1 if direction == 0 else 0
    curr_score = np.mean(style_change_effect[:, direction, sindex, 0]) + np.mean(style_change_effect[:, other_direction, sindex, 1])
    scores.append(curr_score)


s_indices_and_signs = [all_sindex_joined_class_0[i] for i in np.argsort(scores)[::-1]]
#s_indices_and_signs = [i for i in s_indices_and_signs if i[0] != 0 and i[1] != 0]

print('Directions and style indices for moving from class 1 to class 0 = ', s_indices_and_signs[:num_indices])
print('Use the other direction to move from class 0 to 1.')




def generate_user_study_img(tensor, upscale_res=None, nrow=2) -> None:
    changed_tensor = tensor.clone()
    if upscale_res is not None:
        changed_tensor = resize(changed_tensor, upscale_res)
    grid = make_grid(changed_tensor, nrow=nrow)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)

    return im

def get_images(dlatent,
                generator,
                classifier,
                sindex,
                s_style_min,
                s_style_max,
                style_direction_index,
                shift_size,
                label_size,
                noise,
                cuda_rank):

    with torch.no_grad():
        dlatent = [(torch.unsqueeze(torch.Tensor(dlatent).cuda(cuda_rank), 0), 5)]
        w_latent_tensor = styles_def_to_tensor(dlatent)
        generated_image, style_coords = generator(w_latent_tensor, noise, get_style_coords=True)
        base_prob = torch.softmax(classifier.classify_images(generated_image), dim=1)[0]

        block_idx, weight_idx = sindex_to_block_idx_and_index(generator, sindex)
        block = generator.blocks[block_idx]

        current_style_layer = None
        one_hot = None

        if weight_idx < block.input_channels:
            current_style_layer = block.to_style1
            one_hot = torch.zeros((1, block.input_channels)).cuda(cuda_rank)
        else:
            weight_idx -= block.input_channels
            current_style_layer = block.to_style2
            one_hot = torch.zeros((1, block.filters)).cuda(cuda_rank)

        one_hot[:, weight_idx] = 1


        if style_direction_index == 0:
            shift = one_hot * ((s_style_min - style_coords[:, sindex]) * shift_size).unsqueeze(1)
        else:
            shift = one_hot * ((s_style_max - style_coords[:, sindex]) * shift_size).unsqueeze(1)

        shift = shift.squeeze(0)
        current_style_layer.bias += shift
        changed_image, style_coords2 = generator(w_latent_tensor, noise, get_style_coords=True)
        changed_prob = torch.softmax(classifier.classify_images(changed_image), dim=1)[0]
        current_style_layer.bias -= shift

    return generated_image, changed_image, style_coords, style_coords2


def create_latent(image):
  dataset = torch.utils.data.TensorDataset(image)
  dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
  
  StylEx = StylEx(image_size = 64)
  StylEx.load_state_dict(torch.load("/content/drive/MyDrive/StylEx/models/old_faces_gender_mobilenet/model_134.pt")["StylEx"])#load the weight for the generator

  for image in data_loader:
      image=image.to('cuda')
      encoder_output = StylEx.encoder(image)
      real_classified_logits = classifier.classify_images(image)
      style = [(torch.cat((encoder_output, real_classified_logits), dim=1),
                StylEx.G.num_layers)]  # Has to be bracketed because expects a noise mix
      noise = image_noise(1, 1, device=0)
    
      w_styles = styles_def_to_tensor(style)

  return w_styles





#GENERATORE CON UN SOLO ATTRIBUTO

def change_image(attribute_number, image):
  
    
  indices_and_signs = np.array([s_indices_and_signs[0]] + [s_indices_and_signs[1]] + [s_indices_and_signs[2], s_indices_and_signs[3]])

  direction_index, style_index = indices_and_signs[attribute_number]
  gender = "male" if attribute_number % 2 == 0 else "female"
  if gender == "male" or gender == "female":
        gender_index = 0 if gender == "male" else 1
  if gender_index == 0:
              style_direction = 1 if direction_index == 0 else 0
  else:
      style_direction = direction_index       #Direction 0 diventa più maschio, Direction 1 diventa più donna


  a, b, c, d = get_images(dlatent=W_values[image_number], generator=stylex.G, classifier=classifier, sindex=style_index, s_style_min=style_min[style_index], s_style_max=style_max[style_index],
                  style_direction_index=style_direction, shift_size=shift_size, label_size=2, noise=saved_noise, cuda_rank=cuda_rank)
  torch.set_printoptions(threshold=10_000)
  #print(d-c)

  """
  fig, axes = plt.subplots(1, 2)
  image_np = a.squeeze().permute(1, 2, 0).cpu().numpy()
  axes[0].imshow(image_np)
  axes[0].axis('off')
  image_np2 = b.squeeze().permute(1, 2, 0).cpu().numpy()
  axes[1].imshow(image_np2)
  axes[1].axis('off')
  plt.show()
  """
