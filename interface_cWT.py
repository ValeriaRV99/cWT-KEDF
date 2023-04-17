#!/usr/bin/env python
# coding: utf-8

import numpy as np
from matplotlib import pyplot as plt

from dftpy.ions import Ions
from dftpy.field import DirectField
from dftpy.grid import DirectGrid
from dftpy.functional import LocalPseudo, Functional, TotalFunctional, ExternalPotential
from dftpy.formats import io
from dftpy.math_utils import ecut2nr
from dftpy.optimization import Optimization


from ase.build import bulk
from ase.io.trajectory import Trajectory
from ase.io import read
from ase.lattice.spacegroup import Spacegroup
from ase.lattice.spacegroup import crystal

from scipy.optimize import minimize
import interface_cWT





# In[3]:


from sklearn.model_selection import train_test_split 
from sklearn.kernel_ridge import KernelRidge #as modelKR
from sklearn.linear_model import Ridge #as modelR
from sklearn.linear_model import Lasso
from dscribe.descriptors import CoulombMatrix, SineMatrix, EwaldSumMatrix, SOAP
from sklearn.tree import DecisionTreeRegressor


# In[26]:


# def main(Phase, model, descriptor):
#     real_rho0, pred_rho0, vol = ML_regression(Phase, model, descriptor) 
#     ks_energy, cwt_energy, pred_energy, volume_30 = get_energy(Phase, real_rho0, pred_rho0)
#     return ks_energy, cwt_energy, pred_energy, vol


# In[27]:


def ML_regression(Phase, model, descriptor):
    dictionary = {
        'Si_Btin': {'structure': 'diamond', 'pp': 'si.lda.recpot'}, 
        'Si_fcc': {'structure': 'fcc', 'lattice': '3.405'},
        'Si_bcc': {'structure': 'bcc', 'lattice': '3.09'},
        'Si_8cd': {'structure': 'diamond', 'lattice': '5.43'},
        'Si_7cd': {'structure': 'diamond', 'lattice': '5.43'},
    }
    if model == 'KernelRidge':
        model = KernelRidge(alpha=0.001, kernel='rbf', kernel_params={'gamma': 0.1})
    if model == 'DecisionTree':
        model = DecisionTreeRegressor(max_depth=25)
    
    rho0_Btin = np.load("/home/vr371/cWT-KEDF/Phases/DATA/Si_Btin/DATA2/rho0.npy") #
    rho0_fcc = np.load("/home/vr371/cWT-KEDF/Phases/DATA/Si_fcc/DATA2/rho0.npy") #
    rho0_bcc = np.load("/home/vr371/cWT-KEDF/Phases/DATA/Si_bcc/DATA2/rho0.npy") #
    rho0_8cd = np.load("/home/vr371/cWT-KEDF/Phases/DATA/Si_8cd/DATA2/rho0.npy")
    rho0_7cd = np.load("/home/vr371/cWT-KEDF/Phases/DATA/Si_7cd/DATA2/rho0.npy")
    rho0 = np.hstack((rho0_Btin, rho0_fcc, rho0_bcc, rho0_8cd.reshape(30), rho0_7cd.reshape(30)))
    
    if descriptor=='Ewal_matrix':
        Ewal_matrices = np.load('/home/vr371/cWT-KEDF/Phases/DATA/Ewal_matrices.npy')  
        esm_Btin = np.asarray(Ewal_matrices)[:30,0]
        esm_fcc = np.asarray(Ewal_matrices)[:30,1]
        esm_bcc = np.asarray(Ewal_matrices)[:30,2]
        esm_8cd = np.asarray(Ewal_matrices)[:30,3]
        esm_7cd = np.asarray(Ewal_matrices)[:30,4]
        Matrix = np.vstack((esm_Btin, esm_fcc, esm_bcc, esm_8cd, esm_7cd))
    if descriptor=='Sine_matrix':
        Sine_matrices = np.load('/home/vr371/cWT-KEDF/Phases/DATA/Sine_matrices.npy')
        sine_Btin = np.asarray(Sine_matrices)[:30,0]
        sine_fcc = np.asarray(Sine_matrices)[:30,1]
        sine_bcc = np.asarray(Sine_matrices)[:30,2]
        sine_8cd = np.asarray(Sine_matrices)[:30,3]
        sine_7cd = np.asarray(Sine_matrices)[:30,4]
        Matrix = np.vstack((sine_Btin, sine_fcc, sine_bcc, sine_8cd, sine_7cd))

    X = Matrix.reshape(150,8,8)
    y = np.asarray(rho0)
    
    rh0 = []
    for j, r0 in enumerate(y):
            rho0_matrix = np.zeros((len(X[1])*len(X[1])))
            rho0_matrix[0] = r0
            rh0.append(rho0_matrix)
    X = X.reshape(-1, len(X[1])*len(X[1]))
    y = np.asarray(rh0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model.fit(X_train, y_train)
    model.score(X_test, y_test)
    
    r = np.linspace(0.8, 1.4, 30)
    VOLUME = []
    pred_rho0 =[]
    for i, d in enumerate(r):
        if Phase == 'Si_Btin':
            ions = Ions.from_ase(crystal('Si',[(0,0,0)],spacegroup=141,cellpar = [4.81,4.81,2.65, 90,90,90]))
        else:
            ions = Ions.from_ase(bulk('Si', dictionary[Phase]['structure'], a=float(dictionary[Phase]['lattice']), cubic=True))   
        if Phase == 'Si_7cd':
            ions.pop(i=1)
        cell = ions.get_cell()
        ions.set_cell(cell * d, scale_atoms=True)
        vol=ions.get_volume()
        if descriptor=='Ewal_matrix':
            esm = EwaldSumMatrix(n_atoms_max=8)
            X_pol = esm.create(ions)
        if descriptor=='Sine_matrix':
            sine = SineMatrix(n_atoms_max=8)
            X_pol = sine.create(ions)
        y_pred = model.predict(X_pol.reshape(-1, len(X_pol)))
        pred_rho0.append(y_pred[0,0])
        VOLUME.append(vol)
    
    if Phase == 'Si_Btin':
        real_rho0 = rho0[0:30]
    if Phase == 'Si_fcc':
        real_rho0 = rho0[30:60]
    if Phase == 'Si_bcc':
        real_rho0 = rho0[60:90]
    if Phase == 'Si_8cd':
        real_rho0 = rho0[90:120]
    if Phase == 'Si_7cd':
        real_rho0 = rho0[120:150]
    return np.asarray(real_rho0), np.asarray(pred_rho0), np.asarray(VOLUME)


# In[28]:


def get_energy(Phase, real_rho0, pred_rho0):
    dictionary = {
        'Si_Btin': {'structure': 'diamond', 'pp': 'si.lda.recpot'}, 
        'Si_fcc': {'structure': 'fcc', 'lattice': '3.405'},
        'Si_bcc': {'structure': 'bcc', 'lattice': '3.09'},
        'Si_8cd': {'structure': 'diamond', 'lattice': '5.43'},
        'Si_7cd': {'structure': 'diamond', 'lattice': '5.43'},
    }
    
    path_pp='/home/vr371/PP/ofpp/EAC/upf/blps/'
    file1='si.lda.upf'
    PP_list = {'Si': path_pp+file1}
    
    KS_TE = []
    for i in np.arange(0,30,1):
        with open('/home/vr371/cWT-KEDF/Phases/DATA/'+ str(Phase)+ '/DATA2/Si'+str(i)+'.out') as D:
            k = [match for match in D if "Total energy" in match]
        KS_te = str(k).split()[6]
        KS_TE.append(float(KS_te))

    VOLUME_30 = []
    l = np.linspace(0.8, 1.4, 30)
    for i, d in enumerate(l):
        if Phase == 'Si_Btin':
            ions = Ions.from_ase(crystal('Si',[(0,0,0)],spacegroup=141,cellpar = [4.81,4.81,2.65, 90,90,90]))
        else:
            ions = Ions.from_ase(bulk('Si', dictionary[Phase]['structure'], a=float(dictionary[Phase]['lattice']), cubic=True))
        if Phase == 'Si_7cd':
            ions.pop(i=1)
        cell = ions.get_cell()
        ions.set_cell(cell * d, scale_atoms=True)
        vol=ions.get_volume()
        VOLUME_30.append(vol)
    
    r = np.linspace(0.8, 1.4, 30)
    real_rho0 = np.asarray(real_rho0)
    real_energy = []
    XC = Functional(type='XC',name='LDA')
    HARTREE = Functional(type='HARTREE')
    for i, d in enumerate(r):
        real_KE = Functional(type='KEDF',name='WT', rho0=real_rho0[i])
        if Phase == 'Si_Btin':
            ions = Ions.from_ase(crystal('Si',[(0,0,0)],spacegroup=141,cellpar = [4.81,4.81,2.65, 90,90,90]))
        else:
            ions = Ions.from_ase(bulk('Si', dictionary[Phase]['structure'], a=float(dictionary[Phase]['lattice']), cubic=True))
        if Phase == 'Si_7cd':
            ions.pop(i=1)
        cell = ions.get_cell()
        ions.set_charges(4)
        ions.set_cell(cell * l[i], scale_atoms=True)
        nr = ecut2nr(ecut=25, lattice=ions.cell)
        grid = DirectGrid(lattice=ions.cell, nr=nr)
        rho_ini = DirectField(grid=grid)
        rho_ini[:] = ions.get_ncharges()/ions.cell.volume
        PSEUDO = LocalPseudo(grid = grid, ions=ions, PP_list=PP_list, rcut=10)
        realevaluator = TotalFunctional(KE=real_KE, XC=XC, HARTREE=HARTREE, PSEUDO=PSEUDO)
        optimization_options = {'econv' : 1e-5*ions.nat}
        optreal = Optimization(EnergyEvaluator=realevaluator, optimization_options = optimization_options, 
                           optimization_method = 'TN')
        realrho = optreal.optimize_rho(guess_rho=rho_ini)
        realenergy = realevaluator.Energy(rho=realrho, ions=ions)
        real_energy.append(realenergy)

    l = np.linspace(0.8, 1.4, 30)
    pred_rho0 = np.asarray(pred_rho0) 
    pred_energy = []
    VOLUME = []
    for i, d in enumerate(l):
        pred_KE = Functional(type='KEDF',name='WT', rho0=pred_rho0[i])
        if Phase == 'Si_Btin':
            ions = Ions.from_ase(crystal('Si',[(0,0,0)],spacegroup=141,cellpar = [4.81,4.81,2.65, 90,90,90]))
        else:
            ions = Ions.from_ase(bulk('Si', dictionary[Phase]['structure'], a=float(dictionary[Phase]['lattice']), cubic=True))
        if Phase == 'Si_7cd':
            ions.pop(i=1)
        cell = ions.get_cell()
        ions.set_charges(4)
        ions.set_cell(cell * l[i], scale_atoms=True)
        nr = ecut2nr(ecut=25, lattice=ions.cell)
        grid = DirectGrid(lattice=ions.cell, nr=nr)
        rho_ini = DirectField(grid=grid)
        rho_ini[:] = ions.get_ncharges()/ions.cell.volume
        PSEUDO = LocalPseudo(grid = grid, ions=ions, PP_list=PP_list, rcut=10)
        predevaluator = TotalFunctional(KE=pred_KE, XC=XC, HARTREE=HARTREE, PSEUDO=PSEUDO)
        optimization_options = {'econv' : 1e-5*ions.nat}
        optpred = Optimization(EnergyEvaluator=predevaluator, optimization_options = optimization_options, 
                           optimization_method = 'TN')
        predrho = optpred.optimize_rho(guess_rho=rho_ini)
        predenergy = predevaluator.Energy(rho=predrho, ions=ions)
        pred_energy.append(predenergy)
    return np.asarray(KS_TE), np.asarray(real_energy), np.asarray(pred_energy), np.asarray(VOLUME_30)

