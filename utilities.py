import pyunsplash
import requests
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.io import imread, imshow
import numpy as np
import argparse
import cv2 
from google.colab.patches import cv2_imshow
import math
import random
import os
import pandas as pd
from tqdm import tqdm
import colorsys

def convert_rgb_to_hls(r, g, b):
    h, l, s = colorsys.rgb_to_hls(r/255, g/255, b/255)
    return (int(round(h * 360)), int(round(l * 100)), int(round(s * 100)))

def convert_hls_to_rgb(h, l, s):
    r, g, b = colorsys.hls_to_rgb(h/360, l/100, s/100)
    return (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))

class Bucket():
  def __init__(self, colours):
    # Initialising Bucket attributes
    self.red = []; self.blue= []; self.green= []
    for channel in colours:
      self.red.append(channel[0])
      self.green.append(channel[1])
      self.blue.append(channel[2])
    self.ranges = (max(self.red) - min(self.red),
                   max(self.green) - min(self.green),
                   max(self.blue) - min(self.blue))
    self.max_range = max(self.ranges)
    self.max_channel_index = self.ranges.index(self.max_range)
    self.colours = colours

  
  def median_split(self):
    # Calculating median and splitting buckets
    median = len(self.colours) // 2
    colours = sorted(self.colours, key= lambda a: a[self.max_channel_index])
    return Bucket(colours[0:median]), Bucket(colours[median:-1])

  def average(self):
    # Averaging
    r = int(np.mean(self.red))
    g = int(np.mean(self.green))
    b = int(np.mean(self.blue))
    return r, g, b

  def __lt__(self, other):
    # Defining the behaviour for the less than operator. Used while sorting
    return self.max_range < other.max_range

def median_cut(image, num_colours):
  colours = []; averages = []
  # Generating Colour List
  colour_list = image.getcolors(image.width * image.height)
  for count, colour in colour_list:
    colours += [colour] * count
  
  # Initialising Buckets
  buckets = [Bucket(colours)]

  # Sorting and averaging
  while ((len(buckets) <= num_colours) or (len(buckets) <= 16)): 
    buckets.sort()
    buckets += buckets.pop().median_split()
  for bucket in buckets:
    averages.append(bucket.average())
  return averages
  
def create_palette(buckets):
    w = 100; h = 100
    bucks=[]
    background = Image.new('RGB', (w * len(buckets), h))
    for index in range(len(buckets)):
      buckets[index] = [int(x) for x in buckets[index]]
      colour = Image.new('RGB', (w, h), tuple(buckets[index]))
      background.paste(colour, (w * index, 0))
      bucks.append(buckets[index])
    return background, bucks

def apply_blur(path, blur):
  img_copy = cv2.imread(path)
  img_copy = cv2.GaussianBlur(img_copy, blur, 0)
  img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
  img_copy = Image.fromarray(img_copy)
  return img_copy

def calculate_cost(col, colours_wanted, brightness = 1):
  costs = []; indexes = []; sorted_costs = []; final_palettes = []; final_colours = []; count=0; sorted_costs_final = []
  for colour in col:
    colour_hls = [convert_rgb_to_hls(c[0], c[1], c[2]) for c in colour]
    cost = np.zeros(len(col))
    sorted_cost = np.zeros_like(cost)
    # Find the channel in RGB with the highest value
    cost = np.square(colour_hls[0][0] - np.array(colour_hls)[:,0])
    sorted_cost = sorted(cost, reverse= False)
    # Find indexes of the costs to sort the colour list with
    index = np.argsort(brightness*cost)
    colour = np.array(colour)[index]
    cost = cost/np.max(cost)
    costs.append(cost)
    sorted_costs.append(sorted_cost)
    indexes.append(index)
    max = math.ceil(math.log10(sorted_cost[-1]))
    sc_final = []
    for i in range(0, max):
      sc_temp = np.array(sorted_cost)[tuple([(np.log10(sorted_cost) >= i) & (np.log10(sorted_cost) < i+1)])]
      if len(sc_temp) == 0:
        continue
      a = np.array(colour)[[sorted_cost.index(x) for x in sc_temp if x in set(sorted_cost)]]
      sc_final.append(np.round((np.average(a, axis= 0)), 0))
    sorted_costs_final.append(sc_final)
    palette_new,buckets = create_palette(sc_final)
    final_palettes.append(palette_new)
    final_colours.append(buckets)
  return final_palettes, final_colours

def execute(paths, blur_dict, colours_wanted):
  palettes = []; colours_list = []  
  for path in tqdm(paths):
    blur, im = calc_blur(path, blur_dict)
    image_blur = apply_blur(path, blur)
    median = median_cut(image_blur, colours_wanted)
    palette, colour = create_palette(median)
    palettes.append(palette)
    colours_list.append(colour)
  return palettes, paths, colours_list

def image_resize(temp_filepath = os.getcwd() + "/local/"):
  os.chdir(f"/content/")
  for image_filepath in os.listdir(temp_filepath):
    im = Image.open(os.path.join(temp_filepath, image_filepath))
    im = im.resize((1200, 1200), resample= Image.ANTIALIAS)
    im.save(os.path.join(temp_filepath, image_filepath))

def final_palette(buckets):
  w = 100; h = 100; dw =100//len(buckets[0][0]); dh = 100//len(buckets[0][0])
  middle = math.floor(len(buckets[0][0])/2)
  bucks=[]; backgrounds = []
  for index in range(len(buckets)):
    background = Image.new('RGB', (w * len(buckets[index]), h))
    for i in range(len(buckets[index])):
      colour = Image.new('RGB', (w, h), buckets[index][i][middle])
      background.paste(colour, (w*i, 0))
      for j in range(len(buckets[index][i])):
        side = Image.new('RGB', (dw, dh), tuple(buckets[index][i][j]))
        background.paste(side, (w*i + (dw * j), h-dh))
    bucks.append(buckets[index])
    backgrounds.append(background)
  return backgrounds, bucks

def fetch(queries, colours_wanted, blur_dict, dataset):
  tally_list = []
  # Cleaning up queries
  queries = queries.split(sep = ",")
  queries = [query.strip() for query in queries]
  queries = [query.lower() for query in queries]
  paths = []
  # Compare query with Keywords from IR Database
  if queries[0] == "all":
    paths = "." + dataset["File Path"]
  else:
    dataset["Tally"] = pd.Series(np.zeros((len(dataset["Image Name"]))), index= dataset.index)
    for query in queries: 
      count = 0
      for key in dataset["Keywords"]:
        if key.find(query) != -1:
          #dataset.iloc["Tally"][count] += 1
          dataset.loc[count, "Tally"] += 1
        count+=1
    dataset.sort_values(by='Tally', ascending=False, inplace= True)
    item_counts = dataset["Tally"].value_counts(normalize=False)
    paths = "." + dataset[:len(dataset)-item_counts[0]]["File Path"]
    tally_list = [int(tally) for tally in dataset["Tally"] if int(tally) != 0]
    tally_list.sort(reverse = True)
  # Perform Median cut on each of these images
  return execute(paths, blur_dict, colours_wanted), tally_list

def plot(final_palettes, paths, colours_wanted, tallies = []):
  plt.figure(figsize= (5*len(paths), 2.5*len(paths)))
  for count in range(len(final_palettes)): 
    # Plot colour palette
    plt.subplot(len(final_palettes)*2, 1, 2*count+1)
    if len(tallies) != 0:
      plt.title(f"Keyword Matches: {tallies[count]}")
    plt.imshow(np.asarray(final_palettes[count]))
    plt.axis("off")
    # Plot corresponding image
    plt.subplot(len(final_palettes)*2, 1, 2*count+2)
    image = Image.open(np.array(paths)[count])
    plt.imshow(image)
    plt.axis('off')
    count+=1

def generate_side_colours(final_colours):
  fin_pal = []
  for palettes in final_colours:
    final = []
    for colours in palettes:
      hls_colours = []; rgb_colours = []
      hls = convert_rgb_to_hls(colours[0], colours[1], colours[2])
      inc = (100 - hls[1]) // 4
      dec = math.ceil((0 - hls[1]) / 4)
      for j in range(4):
        hls_colours.append((hls[0], hls[1] + (4-j)*dec, hls[2]))
        rgb_colours.append(convert_hls_to_rgb(hls_colours[j][0], hls_colours[j][1], hls_colours[j][2]))
      for i in range(5):
        hls_colours.append((hls[0], hls[1] + i*inc, hls[2]))
        rgb_colours.append(convert_hls_to_rgb(hls_colours[4+i][0], hls_colours[4+i][1], hls_colours[4+i][2]))
      rgb_colours = rgb_colours[1:-1]
      final.append(rgb_colours)
    fin_pal.append(final)
  return fin_pal

def fetch_from_unsplash(query, per_page_results, api_key, blur_dict, colours_wanted):
  os.chdir(f"/content/")
  palettes = []; colours_list = []
  # Preprocess queries to remove whitespaces and make it case insensitive
  pu = pyunsplash.PyUnsplash(api_key)
  queries = query.lower()
  search = pu.search(type_='photos', query= queries, per_page= per_page_results)
  count = 1
  # Make folder temp to store images locally
  temp_filepath = os.getcwd() + "/local/"
  if not os.path.exists(temp_filepath):
    os.mkdir(temp_filepath)
  # Download images and resize
  for photo in search.entries:
    response = requests.get(photo.link_download)
    # For standard naming format
    if count <= 9:
      file = open(f"{temp_filepath}sample_image0{count}.jpg", "wb")
    else:
      file = open(f"{temp_filepath}sample_image{count}.jpg", "wb")
    file.write(response.content)
    file.close()
    count+=1
  # Removing all extra images from last searches
  [os.remove(temp_filepath + path) for path in os.listdir(temp_filepath) if int(path[-6:-4]) > int(per_page_results)]
  
  paths = []
  for root, dirs, files in os.walk(os.path.abspath(temp_filepath)):
    for file in files:
      paths.append(temp_filepath + file)
  
  image_resize()
  
  return execute(paths, blur_dict, colours_wanted)
  
def calc_blur(path, blur_dict):
  im = cv2.imread(path)
  im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
  hsv_im = cv2.cvtColor(im, cv2.COLOR_BGR2HLS)
  h, l, s = cv2.split(hsv_im)
  h = h.flatten()
  h.sort()
  sensitivity = 20; count = 0
  steps = 200 // sensitivity
  categories = []; temp = []
  sum = len(h)
  for a, b in zip(range(steps, 220, steps), range(0, 200, steps)):
    newArr = h[(h < a) & (h >= b)]
    categories.append(newArr)
  for cat in categories:
    if (len(cat) / sum) > (1/sensitivity):
      count+=1
  blur = blur_dict[count]
  return blur, im