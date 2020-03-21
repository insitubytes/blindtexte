#!/usr/bin/env python
# coding: utf-8

from time import sleep, time
from importlib import reload
import blindtexte_generator as btg

reload(btg)


bt = btg.BlindTexteGenerator()
# bt.connect()
# bt.clear_all()

# for i in range(1000):
#     bt.send_random()
#     print(i)

# bt.generate_and_send_images(n_images=9)
# for i in range(10):
#     bt.generate_and_save_images("test.txt", n_images=1)


# ein_01 = "15"
# aus_01 = "16"
# now = time.strftime("%H:%M")
