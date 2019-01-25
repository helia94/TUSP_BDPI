#!/usr/bin/python3
# This file is part of Bootstrapped Dual Policy Iteration
# 
# Copyright 2018, Vrije Universiteit Brussel (http://vub.ac.be)
#     authored by Denis Steckelmacher <dsteckel@ai.vub.ac.be>
#
# BDPI is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# BDPI is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with BDPI.  If not, see <http://www.gnu.org/licenses/>.

import sys
import math

##if len(sys.argv) < 3:
##    print("Usage: %s <column> <prefix> [file...]" % sys.argv[0])
##    sys.exit(0)
##
### Open all files
##col = int(sys.argv[1])
##prefix = sys.argv[2]
##files = [open(f, 'r') for f in sys.argv[3:]]
files = [open(f, 'r') for f in ['out-']]
# Read and average each files
N = float(len(files))
i = 0
running_mean = None
running_err = None
elems = [0.0] * len(files)
timesteps = [0] * len(files)
elements = [None] * len(files)

running_mean = None
running_err = 0.0
running_coeff = 0.95

while True:
    # Read a line from every file
    ok = False

    for j, f in enumerate(files):
        while True:
            elements[j] = f.readline().strip().replace(',', ' ')

##            if elements[j].startswith(prefix) or len(elements[j]) == 0:
            if  len(elements[j]) == 0:
                break

        if len(elements[j]) > 0:
            ok = True

    if not ok:
        # No more file
        break

    try:
        # Plot lines
        N = 0

        for j in range(len(files)):
            if len(elements[j]) > 0:
                parts = elements[j].split()

                elems[j] = float(parts[col])
                try:
                    timesteps[j] = int(parts[2])
                except:
                    timesteps[j] = 0

                N += 1
            else:
                elems[j] = 0.0
                timesteps[j] = 0

        mean = sum(elems) / N
        var = sum([(e - mean)**2 for e in elems if e != 0.0])
        std = math.sqrt(var)
        err = 1.96 * std / math.sqrt(N)     # 95% confidence interval, known standard deviation

        if running_mean is None:
            running_mean = mean
            running_err = err
        else:
            running_mean = running_coeff * running_mean + (1.0 - running_coeff) * mean
            running_err = running_coeff * running_err + (1.0 - running_coeff) * err

        if i % 16 == 0:
            print(i / 1000., running_mean, running_mean + running_err, running_mean - running_err, sum(timesteps) / N)

        i += 1
    except Exception:
        pass
