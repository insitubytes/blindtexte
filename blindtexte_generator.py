#!/usr/bin/env python
# coding: utf-8
# version: 0.4 2020.03.05 23:54
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


class BlindTexteGenerator:
    def __init__(self):
        # CONNECTION
        self.status = 0  # 0: no connection
        self.devices = []
        self.port = None
        self._baud = 115200  # schnittstelle arduino
        self._initial_pause = 0.0  # s
        self._min_pause = 0.000  # pause zwischen den pixeldatenblöcken
        # SIGNALING
        self._control_marker = 200
        self._clear_all = 201
        self._set_all = 202
        self._max_col = 128
        self._max_row = 128
        self._panel_size = 16
        self._n_panels = 8
        self._verbose = False  # ausgabe der übertragen daten
        # MODEL AND IMAGES
        self.model = load_model("models/gen/gan10000")
        self._noise_dim = 8
        self._thres = 0.3
        self._images = deque([], 2)  # für veränderung der bilder
        self._send_diffs = False
        self._image_size = (128, 128)
        self._output = None
        self._diff = None
        self._shuffle = True  # pixelreihenfolge zufällig
        self._n_images = 0
        self._n_generated_images = 0
        self._plot_images = True
        self._save_images = True
        self._horizontal_flip_saved_images = True
        self._image_path = "images/"
        self._image_format = ".jpg"
        self._axes = []
        self._plot_panels = True
        self._filter_images = False
        self._filter_function = ImageFilter.EDGE_ENHANCE_MORE
        self._panel_fig = None

    def connect(self):
        device_dir = "/dev/"
        self.devices = []
        for file in os.listdir(device_dir):
            if file.startswith("ttyUSB"):
                self.devices.append(os.path.join(device_dir, file))
        if len(self.devices) == 0:
            raise Exception("no device found!")
        if len(self.devices) > 1:
            print(self.devices)
            raise Exception("multiple ttyusb devices found!")
        self.port = serial.Serial(self.devices[0], self._baud, timeout=None)
        print("connection successful - connected to port", self.devices[0])
        sleep(self._initial_pause)

    def clear_all(self):
        self._send([self._clear_all])

    def set_all(self):
        self._send([self._set_all])

    def _send(self, values):
        if self._verbose:
            out_string = "sending "
            for value, label in zip(values, ["signal", "\trow", "\tcol"]):
                out_string += label + ": " + str(value)
            print(out_string)
        # if not self.port.is_open:
        #     self.port.open()
        self.port.write(bytearray(values))
        if self._plot_panels:
            self._plot_on_panels(values)
        # self.port.close()
        sleep(self._min_pause)

    def _get_signal(self, ix_panel, pixel_value):
        # (1-pixel_value) invertiert die bilder
        return 210 + 20 * pixel_value + ix_panel

    def load_and_send_image(self, filename):
        img = Image.open(filename)
        img = img.convert("L")
        img = img.resize((self._max_row, self._max_col))
        self._images.append(np.array(img))
        self._determine_output()
        if self._plot_images:
            self._plot_output()
        self._send_image()

    def _get_panel_index_and_col(self, col):
        ix_panel = col // self._panel_size + 1
        col_panel = col % self._panel_size
        return ix_panel, col_panel

    def send_random(self, options=[10, 30]):
        row = np.random.randint(0, self._max_row) + 1
        col = np.random.randint(0, self._max_col) + 1
        ix_panel, col_panel = self._get_panel_index_and_col(col)
        status = np.random.choice(options)
        signal = self._control_marker + ix_panel + status
        self._send([signal, row, col_panel])

    def generate_and_send_images(self, n_images=None):
        self._n_images = 0  # need to reset incase this is called multiple times
        while n_images is None or self._n_images < n_images:
            print("creating image", self._n_images)
            self._generate_new_image()
            print("sending image", self._n_images)
            self._send_image()
            self._n_images += 1
            self._n_generated_images += 1

    def generate_and_save_images_as_text(self, filename, n_images=None):
        self._n_images = 0  # need to reset incase this is called multiple times
        with open(filename, "wt") as file:
            while n_images is None or self._n_images < n_images:
                # print("creating image", self._n_images)
                self._generate_new_image()
                # print("sending image", self._n_images)
                self._save_image_as_txt(file)
                self._n_images += 1

    def _send_image(self):
        if self._output is None:
            return
        t = len(self._output)
        for i, value in enumerate(self._output):
            self._send(value)
            n_dash = int((i / t * 100) // 10)  # ausgabe
            n_none = 9 - n_dash
            print(
                "\rsending", t, "pixels [" + "-" * n_dash + " " * n_none + "]", end=""
            )
        print("\n")

    def _save_image_as_txt(self, file):
        if self._output is None:
            return
        t = len(self._output)
        for i, value in enumerate(self._output):
            for v in value:
                file.write(str(v) + " ")
            n_dash = int((i / t * 100) // 10)
            n_none = 9 - n_dash
            print(
                "\rsaving image",
                self._n_images,
                "with",
                t,
                "pixels \t [" + "-" * n_dash + " " * n_none + "]",
                end="",
            )
        file.write("\n")
        sleep(self._min_pause)

    def _save_image(self):
        # determine filename
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filename = timestr + "-" + str(self._n_generated_images)
        img = Image.fromarray((255 * self._images[-1]).astype("float")).convert("RGB")
        print("saving image to", self._image_path + filename + self._image_format)
        if not os.path.isdir(self._image_path):
            if os.path.isfile(self._image_path):
                raise Exception("Warning: image path is a file")
            os.mkdir(self._image_path)
        if self._horizontal_flip_saved_images:
            img = ImageOps.flip(img)
        img.save(self._image_path + filename + self._image_format)

    def _generate_new_image(self):
        # zufälliger Code!
        random_code = np.random.randn(1, self._noise_dim)
        # hochskalieren
        tf_image = tf.image.resize(self.model.predict(random_code), self._image_size)
        image = tf_image.numpy().squeeze()
        if self._filter_images:
            image = self._filter(image)
        self._images.append((image > self._thres).astype("int"))
        self._determine_output()
        if self._plot_images:
            self._plot_output()
        if self._save_images:
            self._save_image()

    def _filter(self, image):
        img = Image.fromarray((image * 255).astype(np.uint8))
        f_img = np.array(img.filter(self._filter_function)) / 255.0
        return f_img

    def _determine_output(self):
        if not self._send_diffs or len(self._images) == 1:
            self._output = []
            self._diff = []
            # the complete first image needs to be plotted
            for row in range(self._max_row):
                for col in range(self._max_col):
                    ix_panel, col_panel = self._get_panel_index_and_col(col)
                    signal = self._get_signal(ix_panel, self._images[-1][row, col])
                    self._output.append((signal, row, col_panel))
        elif len(self._images) == 2:
            img1, img2 = (self._images[0], self._images[1])
            inds = np.where(img1 != img2)
            self._output = []
            self._diff = []
            for row, col in zip(inds[0], inds[1]):
                if row < self._max_row and col < self._max_col:
                    ix_panel, col_panel = self._get_panel_index_and_col(col)
                    signal = self._get_signal(ix_panel, img2[row, col])
                    self._output.append((signal, self._max_row - row, col_panel))
                    self._diff.append((row, col))
        else:
            raise Exception("should never visit this place!")
        if self._shuffle:
            random.shuffle(self._output)

    def _plot_output(self):  # anzeige portraits auf notebookbildschirm
        fig = pl.figure(0)
        fig.clear()
        ax = fig.add_subplot(111)
        ax.imshow(self._images[-1], interpolation=None, cmap="gray", origin="lower")
        diff = np.array(self._diff)
        img = np.zeros((self._max_row, self._max_col, 4))
        # img[diff[:, 0], diff[:, 1]] = (128, 255, 0, 250)  # zeigt die unterschiede
        ax.imshow(img, alpha=0.7, interpolation=None, origin="lower")
        pl.pause(0.1)
        pl.show()

    def _plot_on_panels(self, values):
        if self._panel_fig is None:
            self._panel_fig = pl.figure(1)
        if values[0] == self._clear_all:
            self._panel_fig.clear()
            self._axes = []
            for p in range(self._n_panels):
                new_ax = self._panel_fig.add_subplot(1, self._n_panels, p + 1)
                self._axes.append(new_ax)
                # self._axes[-1].set_xticklabels([])
                # self._axes[-1].set_yticklabels([])
                self._axes[-1].set_xlim([0, self._max_col // self._n_panels])
                self._axes[-1].set_ylim([0, self._max_row])
                self._axes[-1].set_title("panel " + str(p + 1))
            return
        if values[0] == self._set_all:
            return
        if len(values) == 3:
            signal, row, col = values
            ix_panel = int(str(signal)[-1])  # läuft nur bis 9
            pixel_on = (signal - 200 - ix_panel) == 10
            if pixel_on:
                self._axes[ix_panel - 1].plot(col, row, ".", color=(0, 0, 0))
            else:
                self._axes[ix_panel - 1].plot(col, row, ".", color=(1, 0.4, 0.7))
            # self._axes[-1].set_xlim([0, self._max_col // self._n_panels])
            # self._axes[-1].set_ylim([0, self._max_row])
