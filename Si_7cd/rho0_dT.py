# +
import numpy as np
from matplotlib import pyplot as plt

from dftpy.ions import Ions
from dftpy.functional import Functional
from dftpy.formats import io

from ase.build import bulk
from ase.io import read

from scipy.optimize import minimize
# -

path_pp='/home/valeria/Desktop/programs/dftpy/examples/ofpp/EAC/upf/blps/'
file1='si.lda.upf'
PP_list = {'Si': path_pp+file1}

ks_ke = np.load("/home/valeria/Documents/DFTPY/Fitting_densities/Si7-CD/DATA2/ks_ke.npy")


def min_energy(x0, *args):
        KS_KE, rho = args
        KE = Functional(type='KEDF',name='WT', rho0=x0)
        OF_KE = KE(rho).energy
        diff = np.abs(KS_KE*1/2-OF_KE)
        return diff


l = np.linspace(0.8, 1.4, 30)
delta_E = []
R0 = []
vol = []
for n in np.arange(0,30,1):
    i = int(n)
    rho = io.read_density('/home/valeria/Documents/DFTPY/Fitting_densities/Si7-CD/DATA2/rho0'+str(i)+'.xsf')
    si = bulk('Si', 'diamond', a=5.43, cubic=True)
    si.pop(i=1)
    ions = Ions.from_ase(si)
    cell = ions.get_cell()
    ions.set_cell(cell * l[i], scale_atoms=True) 
    KS_KE = ks_ke[i]
    minn = minimize(min_energy, 0.001, args = (KS_KE, rho), method='Powell', bounds=[[0.001,0.034]], options={'ftol' : 1e-9})
    VOLUME = ions.get_volume()
    vol.append(VOLUME)
    delta_E.append(minn.fun)
    R0.append(minn.x)

fig, axs = plt.subplots(1,2, figsize=(10,3))
im0 = axs[0].plot(np.asarray(vol)*0.529177**3/4,np.asarray(delta_E)*1/4,'*--')
im1 = axs[1].plot(np.asarray(vol)*0.529177**3/4,np.asarray(R0),'*--')
axs[0].set_title('Si-7cd')
axs[1].set_title('Si-7cd')
axs[0].set_xlabel('Volume ($\AA^{3}$)')
axs[0].set_ylabel('$\Delta T_{s}$ (eV/atom)')
axs[1].set_xlabel('Volume ($\AA^{3}$)')
axs[1].set_ylabel('$\u03C1_{0}$')

np.save("/home/valeria/Documents/DFTPY/Fitting_densities/Si7-CD/DATA2/rho0.npy", np.asarray(R0))
