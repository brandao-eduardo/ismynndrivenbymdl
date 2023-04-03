"""
cifar-10 dataset, with support for random labels
"""
import numpy as np
import torch.optim

import torch
import torchvision.datasets as datasets
import pandas as pd
import sklearn.decomposition
import sklearn.metrics

from pathlib import Path
import torch
import torchvision

import torchvision.datasets as datasets


class CIFAR10RandomLabels(datasets.CIFAR10):
  """CIFAR10 dataset, with support for random labels and pixels

  Params
  ------
  corrupt_prob: float
    Default 0.0. The probability of a label being replaced with
    random label.
  num_classes: int
    Default 10. The number of classes in the dataset.
  """
  def __init__(self, corrupt_prob=0.0, num_classes=10, shfld_pxls=False , rnd_pxls=False, gaussian=False, **kwargs):
    super(CIFAR10RandomLabels, self).__init__(**kwargs)
    self.n_classes = num_classes
    if corrupt_prob > 0:
      print("Corrupting labels ...")
      self.corrupt_labels(corrupt_prob)
    elif shfld_pxls:
      print("Shuffling pixels")
      self.shuffled_pixels()
    elif rnd_pxls:
      print("Randomizing pixels")
      self.random_pixels()
    elif gaussian:
      print("Generating pixels from a Gaussian distribution")
      self.gaussian()

  def corrupt_labels(self, corrupt_prob):
    labels = np.array(self.targets)
    np.random.seed(12345)
    mask = np.random.rand(len(labels)) <= corrupt_prob/100.
    rnd_labels = np.random.choice(self.n_classes, mask.sum())
    labels[mask] = rnd_labels
    # we need to explicitly cast the labels from npy.int64 to
    # builtin int type, otherwise pytorch will fail...
    labels = [int(x) for x in labels]
    self.targets = labels

  def shuffled_pixels(self):
    """
    All images are permuted using the same permutation
    :return:
    """
    self.permutation = 12345 #same seed for each permutation
    images = np.array(self.data)
    data = np.swapaxes(np.reshape(images, (images.shape[0], -1, images.shape[-1])), 0, 1)
    np.random.shuffle(data)
    self.data = np.reshape(np.swapaxes(data, 0, 1), images.shape)


  def random_pixels(self):
    """
    Each image is permuted using a different permutation matrix
    :return:
    """
    np.random.seed(12345) # sets the seed for the ensemble of permutations
    images = np.array(self.data)
    for i, img in enumerate(images):
      rndpxls = np.reshape(img, (-1, img.shape[2])).copy()
      np.random.shuffle(rndpxls)
      images[i] = np.reshape(rndpxls, img.shape)
    self.data = images

  def gaussian(self):
    """
    Each pixel in the data is ind. sampled from a Gaussian dist. with mean and variance matching the original dataset's
    :return:
    """
    np.random.seed(12345) # sets the seed for the ensemble of generated pixels
    images = self.data
    print(images.shape)
    mean = np.mean(images, axis=(0,1,2))
    std = np.std(images, axis=(0,1,2))
    data = np.clip(np.random.multivariate_normal(mean=mean, cov=np.diag(std), size=images.shape[:-1]), 0, 255).astype(np.uint8)     # we need to explicitly cast the data as int otherwise pytorch will fail
    self.data = data


#This is all for the dataset noise thing 

def transform_truncated(pca, X, n_components):
    X = pca._validate_data(X, dtype=[np.float64, np.float32], reset=False)
    if pca.mean_ is not None:
        X = X - pca.mean_
    X_transformed = np.dot(X, pca.components_[:n_components, :].T)
    if pca.whiten:
        X_transformed /= np.sqrt(pca.explained_variance_)
    return X_transformed


def inv_transform(pca, X, n_components):
    return np.dot(X, pca.components_[:n_components, :]) + pca.mean_


def inv_forward_transform(pca, X, n_components):
    return inv_transform(
        pca, transform_truncated(pca, X, n_components), n_components
    )

def get_pca_transformed_cifar_100(n_components, train=True):

  transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
  data = torchvision.datasets.CIFAR100(Path() / "data", train=train, download=True, transform=transform)
  dataloader = torch.utils.data.DataLoader(dataset=data, batch_size=len(data))
  images_all, _ = next(iter(dataloader))
  # convert to 1D
  images_flat = images_all.reshape(images_all.shape[0], -1)

  pca = sklearn.decomposition.PCA(n_components=1500)
  images_flat_hat = pca.inverse_transform(pca.fit_transform(images_flat))
  images_hat = inv_forward_transform(pca, X=images_flat, n_components=n_components).reshape(-1, 3, 32,32)

  return images_hat

def normalize_img(x):
    """
    Over all three channels; there should be numpy way
    :param x:
    :return:
    """
    for i in range(3):
        x[...,i] = (x[...,i] - np.amin(x[...,i])) / (np.amax(x[...,i]) - np.amin(x[...,i]))
    return (255*x).astype(np.uint8)

def normalize_dataset(data):
    data_norm = np.array([normalize_img(data[i]) for i in range(len(data))])
    return data_norm


class CIFAR10DatasetNoise(datasets.CIFAR10):
  """CIFAR10 dataset, with support for random labels and pixels

  Params
  ------
  corrupt_prob: float
    Default 0.0. The probability of a label being replaced with
    random label.
  num_classes: int
    Default 10. The number of classes in the dataset.
  """
  def __init__(self, n_components=0, num_classes=10, **kwargs):
    super(CIFAR10DatasetNoise, self).__init__(**kwargs)
    self.n_classes = num_classes
    self.n_components = n_components
    self.train = kwargs['train']
    if n_components > 0:
      print("Adding dataset noise ...")
      self.add_cifar_100(n_components)


  def add_cifar_100(self, n_components):
    cif_100 = get_pca_transformed_cifar_100(self.n_components, self.train)
    print(self.data.shape, np.max(self.data), np.min(self.data))
    norm_cif100 = normalize_dataset(np.moveaxis(cif_100, 1, -1))
    d = np.mean([norm_cif100,self.data], axis=0).astype(np.uint8)#norm_cif100 + self.data #np.moveaxis(cif_100, 1, -1)
    print(norm_cif100.shape, np.max(norm_cif100), np.min(norm_cif100))
    self.data = d#TODO: need to
    print(self.data.shape, np.max(self.data), np.min(self.data))
