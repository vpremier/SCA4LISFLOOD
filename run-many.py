#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 12:55:23 2023

@author: vpremier
"""

import subprocess
import time

basins = ['Alpenrhein','Adige', 'Arve','Laborec','Gallego', 'Umealven', 'Morrumsan', 'Salzach', 
          'Guadalfeo']
seasons = ['1819','1920','2021','2122','2223']
script_to_run = r'./main.py'


for basin in basins:
    with open(script_to_run) as f:
        content = f.readlines()
        
    with open(script_to_run, 'w') as f:
        for line in content:
            if line.startswith('basin'):
                f.write("basin = '%s' \n" % basin)
            else:
                f.write(line)



    # Run the other script
    subprocess.run(["python", script_to_run])
        
# for season in seasons:
#     hy_xxxx = 'hy' + season
#     print(hy_xxxx)
#     for basin in basins:
#         with open(script_to_run) as f:
#             content = f.readlines()
            
#         with open(script_to_run, 'w') as f:
#             for line in content:
#                 if line.startswith('hy_xxxx'):
#                     f.write("hy_xxxx = 'hy%s'\n" % season)
#                 elif line.startswith('basin'):
#                     f.write("basin = '%s' \n" % basin)
#                 else:
#                   f.write(line)


    
#         # Run the other script
#         subprocess.run(["python", script_to_run])