# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# This file is part of the TIGRE Toolbox
#
# Copyright (c) 2015, University of Bath and
#                     CERN-European Organization for Nuclear Research
#                     All rights reserved.
#
# License:            Open Source under BSD.
#                     See the full license at
#                     https://github.com/CERN/TIGRE/blob/master/LICENSE
#
# Contact:            tigre.toolbox@gmail.com
# Codes:              https://github.com/CERN/TIGRE/
# Coded by:           Alexey Smirnov
# --------------------------------------------------------------------------

#%%Initialize
import tigre
import numpy as np
import skimage
from skimage.data import shepp_logan_phantom
from skimage.transform import rescale
from tigre.utilities import sample_loader
from tigre.utilities import CTnoise
import tigre.algorithms as algs
from tigre.utilities.Measure_Quality import Measure_Quality
import tigre.utilities.gpu as gpu
import matplotlib.pyplot as plt
from tigre.utilities import sl3d
from tigre.utilities.im3Dnorm import im3DNORM

#%% Geometry
## FAN BEAM 2D

geo = tigre.geometry(default=True)
# VARIABLE                                   DESCRIPTION                    UNITS
# -------------------------------------------------------------------------------------
# Image parameters
geo.nVoxel = np.array([1, 256, 256])  # number of voxels              (vx)
geo.sVoxel = np.array([1, 256, 256])  # total size of the image       (mm)
geo.dVoxel = geo.sVoxel / geo.nVoxel  # size of each voxel            (mm)

geo.mode = "cone"

#%% Load data and generate projections
# define angles
angles = np.linspace(0, 2/3 * np.pi, 100)

# shepp3d = sl3d.shepp_logan_3d([256, 256, 256], phantom_type=phantom_type)
# shepp = shepp3d[128,:,:].reshape(1, 256, 256) 
shepp = np.float32(shepp_logan_phantom())
shepp = rescale(shepp, scale=256/400, mode='reflect')
shepp = shepp.reshape(1, 256, 256) 

# generate projections
projections = tigre.Ax(shepp, geo, angles)
# add noise
# noise_projections = CTnoise.add(projections, Poisson=1e5, Gaussian=np.array([0, 10]))
noise_projections = projections

#%% recosntruct

#%% Total Variation algorithms
#
#  ASD-POCS: Adaptative Steeppest Descent-Projection On Convex Subsets
# Often called POCS
# ==========================================================================
# ==========================================================================
#  ASD-POCS minimizes At-B and the TV norm separately in each iteration,
#  i.e. reconstructs the image first and then reduces the TV norm, every
#  iteration. As the other algorithms the mandatory inputs are projections,
#  geometry, angles and maximum iterations.
#
# ASD-POCS has a veriety of optional arguments, and some of them are crucial
# to determine the behaviour of the algorithm. The advantage of ASD-POCS is
# the power to create good images from bad data, but it needs a lot of
# tunning.
#
#  Optional parameters that are very relevant:
# ----------------------------------------------
#    'maxL2err'    Maximum L2 error to accept an image as valid. This
#                  parameter is crucial for the algorithm, determines at
#                  what point an image should not be updated further.
#                  Default is 20% of the FDK L2 norm.
#
# its called epsilon in the paper
epsilon = (
    im3DNORM(tigre.Ax(algs.fdk(noise_projections, geo, angles), geo, angles) - noise_projections, 2)
    * 0.15
)
#   'alpha':       Defines the TV hyperparameter. default is 0.002.
#                  However the paper mentions 0.2 as good choice
alpha = 0.002

#   'tviter':      Defines the amount of TV iterations performed per SART
#                  iteration. Default is 20

ng = 25

# Other optional parameters
# ----------------------------------------------
#   'lambda':      Sets the value of the hyperparameter for the SART iterations.
#                  Default is 1
#
#   'lambdared':   Reduction of lambda Every iteration
#                  lambda=lambdared*lambda. Default is 0.99
#
lmbda = 1
lambdared = 0.9999  # you generally want 1


#   'alpha_red':   Defines the reduction rate of the TV hyperparameter
alpha_red = 0.95

#   'Ratio':       The maximum allowed image/TV update ration. If the TV
#                  update changes the image more than this, the parameter
#                  will be reduced.default is 0.95
ratio = 0.94

#   'Verbose'      1 or 0. Default is 1. Gives information about the
#                  progress of the algorithm.

verb = True

niter = 10

imgOSSART = algs.ossart(projections, geo, angles, niter)
imgASDPOCS = algs.asd_pocs(
    projections,
    geo,
    angles,
    10,  # these are very important
    tviter=ng,
    maxl2err=epsilon,
    alpha=alpha,  # less important.
    lmbda=lmbda,
    lmbda_red=lambdared,
    rmax=ratio,
    verbose=verb,
)

#  AwASD_POCS: adaptative weighted ASD_POCS
# ==========================================================================
# ==========================================================================
#
# This is a more edge preserving algorithms than ASD_POCS, in theory.
# delta is the cuttof vlaue of anromalized edge exponential weight....
# not super clear, but it cotnrols at which point you accept something as real vs noise edge.

imgAWASDPOCS = algs.awasd_pocs(
    projections,
    geo,
    angles,
    10,  # these are very important
    tviter=ng,
    maxl2err=epsilon,
    alpha=alpha,  # less important.
    lmbda=lmbda,
    lmbda_red=lambdared,
    rmax=ratio,
    verbose=verb,  # AwASD_POCS params
    delta=np.array([-0.005]),
)

# Measure Quality
# 'RMSE', 'MSSIM', 'SSD', 'UQI'
print("RMSE OSSART:")
print(Measure_Quality(imgOSSART, shepp, ["nRMSE"]))
print("RMSE ASDPOCS:")
print(Measure_Quality(imgASDPOCS, shepp, ["nRMSE"]))
print("RMSE AWASDPOCS:")
print(Measure_Quality(imgAWASDPOCS, shepp, ["nRMSE"]))
print("RMSE DTVASDPOCS:")
print(Measure_Quality(imgDTVASDPOCS, shepp, ["nRMSE"]))

#%% Plot
plt.imsave('shepp-logan.png', shepp[0])
plt.imsave('os-sart.png', imgOSSART[0])
plt.imsave('asd-pocs.png', imgASDPOCS[0])
plt.imsave('aw-asd-pocs.png', imgAWASDPOCS[0])
plt.imsave('dtv-asd-pocs.png', imgDTVASDPOCS[0])
plt.imsave('os-sart-err.png', imgOSSART[0]-shepp[0])
plt.imsave('asd-pocs-err.png', imgASDPOCS[0]-shepp[0])
plt.imsave('aw-asd-pocs-err.png', imgAWASDPOCS[0]-shepp[0])
plt.imsave('dtv-asd-pocs-err.png', imgDTVASDPOCS[0]-shepp[0])

# tigre.plotProj(proj)
# tigre.plotImg(fdkout)
