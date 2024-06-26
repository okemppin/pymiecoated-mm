# Description

A Python code for computing the scattering properties of single- and dual-layered spheres with an easy-to-use object oriented interface.

The code is based on the original [pymiecoated](https://github.com/jleinonen/pymiecoated/) by Jussi Leinonen and has been optimized for fast repeated Mie simulations (multiple sizes and refractive indices). The main use is with NASA's [GEOSmie](https://github.com/GEOS-ESM/GEOSmie/) tool for calculating single-scattering and bulk optical properties for the [GEOS Earth System Model](https://github.com/GEOS-ESM) but the Mie code is fully standalone and can be used as a standard Mie tool where high performance over repeated simulations is desired.

The performance improvement compared to the original pymiecoated is roughly three orders of magnitude (see [Benchmarks](#benchmarks)).

# Usage

Basic usage is as follow:
````python
from pymiecoated.mie_coated import MultipleMie
import numpy as np

# array of size parameters
xarr = np.linspace(0.1, 1, 10)

# high efficiency calculations for coated spheres are not implemented yet
# this is a placeholder to maintain the eventual constructor signature
yarr = None

# array of cosines at which the scattering matrix values should be calculated
costarr = np.cos(np.radians(np.linspace(0,181,1)))

# initialize the MultipleMie object for given size parameters and scattering angles
mm = MultipleMie(xarr, None, costarr)

# pre-calculates various internal Mie parameters that are re-used in later stages
mm.preCalculate()

# list of real refractive indices
mrarr = np.linspace(1.3, 1.5, 10)

# list of imaginary refractive indices
miarr = np.linspace(0.01, 0.05, 10)

for mr in mrarr:
  for mi in miarr:
    # high efficiency Mie simulations over the previously prescribed size and scattering angle ranges
    vals = mm.calculateS12SizeRange(mr, mi)

    # access values from the returned dictionary
    qsca = vals['qsca']
````

Regular Mie simulations (using `from pymiecoated import Mie`) may work but have not been tested. For this usage the user should consult [the instructions of the original pymiecoated repository](https://github.com/jleinonen/pymiecoated/wiki/Instructions) or use the code in the original repository.

Requires NumPy, SciPy and Numba.

# Benchmarks

Simple benchmarks can be found below. Wiscombe's MIEV code is provided for reference, where the code has been modified such that the binary is called only once and all loops are done inside MIEV. The ~2x speedup compared to MIEV is expected given that roughly half of the floating-point operations in repeated simulations are redundant and can be pre-calculated.

Single-core performance for 100 spheres of size x=1000, time per one simulation:
````
                   MIEV: 7.79e-04s
   pymiecoated-original: 3.56e-01s
pymiecoated-multiplemie: 5.70e-04s
````

Single-core performance for 100 spheres of size x=1, time per one simulation:
````
                   MIEV: 9.00e-05s
   pymiecoated-original: 1.53e-02s
pymiecoated-multiplemie: 2.98e-05s
````

Single-core performance for 100 spheres of size x=0.001, time per one simulation:
````

                   MIEV: 6.64e-05s
   pymiecoated-original: 1.34e-02s
pymiecoated-multiplemie: 3.02e-05s
````
