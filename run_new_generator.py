#!/usr/bin/env python
# coding: utf-8

from time import sleep, time
from importlib import reload
import new_generator as new_gen

reload(new_gen)


ng = new_gen.NewGenerator()

for i in range(3):
    ng.generate_new_image()
    ng.save_image()
    sleep(2)
