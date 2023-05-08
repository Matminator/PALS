
from ase.io.trajectory import Trajectory
from ase.visualize import view

from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase import units

from ase.calculators.calculator import Calculator, all_changes
import nequip
from nequip.ase import NequIPCalculator

import os
import time

import numpy as np
import pandas as pd

import torch


class NequIP_ensemble(Calculator):
    implemented_properties = ['energy', 'energies', 'forces', 'stress', 'stresses']
    
    def __init__(self, model_paths, device = 'cuda', *args, **kwargs):
        """
        model_paths: paths to NequIP models.
        """
        
        super().__init__(*args, **kwargs) # For ASE calculator ... 
        
        # importing NequIP models:
        
        start_time = time.time()
        
        models = []
        for path in model_paths:
            model = NequIPCalculator.from_deployed_model(
                model_path = path,
                species_to_type_name = {"Pt": "Pt", "Ti": "Ti", "O": "O"},
                device = device
                )
            models.append(model)
            
                    
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Init time: ", elapsed_time, flush = True)
        
        self.models = models
        
        self.froce_variances = []
        self.energy_variance = []
    
    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        
        super().calculate(atoms, properties, system_changes) # For ASE calculator ...
        
        models_energies = []
        models_forces = []
        
        # Doing calculation with all models:
        for model in self.models:
            
            self.NequIP_calculation(self.atoms.copy(),
                                    model,
                                    models_energies,
                                    models_forces)
            
        # Computing resulting froce and variances
        forces_mean, froce_variances = self.force_calculations(models_forces)
        energy_mean = np.mean(models_energies)
        energy_variance = np.var(models_energies)
        
        self.froce_variances.append(froce_variances)
        self.energy_variance.append(energy_variance)
        
        self.results = {'energy': energy_mean,
                        'energies': np.zeros(len(self.atoms)), 
                        'forces': forces_mean,
                        'stress': np.zeros(6),
                        'stresses': np.zeros((len(self.atoms), 6))}            
            
            
    def NequIP_calculation(self, atoms, model, models_energies, models_forces):
        
        atoms.calc = model
        energy = atoms.get_potential_energy() # Getting energy
        forces = atoms.get_forces()
        
        models_energies.append(energy)
        models_forces.append(forces)
        

    def force_calculations(self, models_forces):

        M = len(models_forces) # Number of models

        # The mean of each force component (x, y and z):
        mean_components = np.mean(models_forces, axis = 0)

        # Computing all variances in a few lines:
        variances = np.sum((models_forces - mean_components)**2, axis = 2)
        variances = np.sum(variances, axis = 0)/M

        return mean_components, variances










config = Trajectory('data_set_GEN_1_for_models.traj')[0]


model_paths = [
    's=0.pth',
    's=1.pth',
    's=2.pth',
    's=3.pth',
    's=4.pth']

# i = 0
# mod_E_val = []
# mod_SD_val = []
# #for config in val_set:
# for config in configs:
#     config.calc = NequIP_ensemble(model_paths)
#     mod_E_val.append(config.get_potential_energy()/len(config))
#     i += 1
#     print(i, flush = True)

config.calc = NequIP_ensemble(model_paths, device = 'cuda')
    

# **********************************************************

    

#MaxwellBoltzmannDistribution(config, temperature_K=300)

dynamics = VelocityVerlet(config, 5 * units.fs)

traj = Trajectory('PALS_molDynTest.traj', 'w', config)
dynamics.attach(traj.write, interval=1)


# start_time = time.time()

dynamics.run(20)

# end_time = time.time()
# elapsed_time = end_time - start_time
# print("Elapsed time: ", elapsed_time)

pd.DataFrame(config.calc.energy_variance).to_csv("PALS_E_Variance_test.csv")
pd.DataFrame(config.calc.froce_variances).to_csv("PALS_force_Variance_test.csv")


