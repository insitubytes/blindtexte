#!/usr/bin/env python
# coding: utf-8
# version: 0.4 2020.03.05 23:54
import gc
import serial
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from matplotlib import use

use("Qt5Agg")
import pylab as pl

pl.ion()
import numpy as np
import random
import time
from time import sleep
import os
from collections import deque
from PIL import ImageFilter, Image, ImageOps


class NewGenerator:
    def __init__(self):
        # CONNECTION
        self._initial_pause = 0.0  # s
        self._min_pause = 0.000  # pause zwischen den pixeldatenblöcken
        # MODEL AND IMAGES
        self._model = None
        self._model_name = None
        # self._model_path = "models/gen/"
        self._model_path = "/home/jokiel/gdrive_insitubytes/link/gen/"
        self._noise_dim = 8
        self._thres = 0.3
        self._images = deque([], 2)  # für veränderung der bilder
        self._show_diffs = False
        self._image_size = (128, 128)
        self._diff = None
        self._shuffle = True  # pixelreihenfolge zufällig
        self._n_images = 0
        self._n_generated_images = 0
        self._plot_images = False
        self._horizontal_flip_saved_images = True
        self._image_path = "images/"
        self._image_format = ".jpg"
        self._axes = []
        self._plot_panels = True
        self._filter_images = False
        self._filter_function = ImageFilter.EDGE_ENHANCE_MORE
        self._panel_fig = None

    def load_most_recent_model(self):
        print("loading most recent model...")
        all_models = next(os.walk(self._model_path))[1]
        if len(all_models) == 0:
            print("\r no model available")
            return
        if self._model_name is None or self._model_name is not all_models[-1]:
            try:
                self._model = load_model(self._model_path + all_models[-1])
            except:
                print("\r could not load model", all_models[-1])
            else:
                print("\r successfully loaded model", all_models[-1])
                self._model_name = all_models[-1]

    def generate_new_image(self):
        if self._model is None:
            print("no model loaded")
            return None
        print("creating image", self._n_images)
        self._generate_new_image()
        print("saving image", self._n_images)
        self._n_images += 1
        self._n_generated_images += 1
        gc.collect()

    def _generate_new_image(self):
        # zufälliger Code!
        random_code = np.random.randn(1, self._noise_dim)
        # hochskalieren
        tf_image = tf.image.resize(self._model.predict(random_code), self._image_size)
        image = tf_image.numpy().squeeze()
        if self._filter_images:
            image = self._filter(image)
        self._images.append((image > self._thres).astype("int"))
        self._determine_diff()
        if self._plot_images:
            self._plot_output()

    def _filter(self, image):
        img = Image.fromarray((image * 255).astype(np.uint8))
        f_img = np.array(img.filter(self._filter_function)) / 255.0
        return f_img

    def _determine_diff(self):
        if not self._show_diffs or len(self._images) == 1:
            self._diff = []
        elif len(self._images) == 2:
            img1, img2 = (self._images[0], self._images[1])
            inds = np.where(img1 != img2)
            self._diff = []
            for row, col in zip(inds[0], inds[1]):
                self._diff.append((row, col))
        else:
            raise Exception("should never visit this place!")

    def _plot_output(self):  # anzeige portraits auf notebookbildschirm
        fig = pl.figure(0)
        fig.clear()
        ax = fig.add_subplot(111)
        ax.imshow(self._images[-1], interpolation=None, cmap="gray", origin="lower")
        diff = np.array(self._diff)
        img = np.zeros((*self._image_size, 4))
        # img[diff[:, 0], diff[:, 1]] = (128, 255, 0, 250)  # zeigt die unterschiede
        ax.imshow(img, alpha=0.7, interpolation=None, origin="lower")
        pl.pause(0.1)
        pl.show()

    def save_image(self, filename):
        if self._model is None:
            print("no model loaded")
            return None
        # determine filename
        # timestr = time.strftime("%Y%m%d-%H%M%S")
        # filename = timestr + "-" + str(self._n_generated_images)
        # filename = "always_the_same"
        img = Image.fromarray((255 * self._images[-1]).astype("uint8")).convert("RGB")
        print("saving image to", self._image_path + filename + self._image_format)
        if not os.path.isdir(self._image_path):
            if os.path.isfile(self._image_path):
                raise Exception("Warning: image path is a file")
            os.mkdir(self._image_path)
        if self._horizontal_flip_saved_images:
            img = ImageOps.flip(img)
        img.save(self._image_path + filename + self._image_format)
