#!/usr/bin/env python
# coding: utf-8

from time import sleep, time
from importlib import reload
import blindtexte_generator as btg

reload(btg)


bt = btg.BlindTexteGenerator()
bt.connect()
# bt.set_all()
bt.clear_all()
sleep(1)

# for i in range(10):
#   bt.send_random()
#  print(i)

# for i in range(5):
# print(i)
# bt.generate_and_send_images(n_images=1)
# bt.generate_and_save_images("test.txt", n_images=100)
# bt.send_random

start = time()
end = start + 1 * 60  # *60:min
while time() < end:
    bt.generate_and_send_images(n_images=1)
