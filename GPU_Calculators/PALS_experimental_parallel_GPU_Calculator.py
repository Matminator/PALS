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
import torch.multiprocessing as mp


class NequIP_ensemble(Calculator):
    implemented_properties = ['energy', 'energies', 'forces', 'stress', 'stresses']
    
    def __init__(self, model_paths, device = 'cuda', *args, **kwargs):
        """
        model_paths: paths to NequIP models.
        """
        
        super().__init__(*args, **kwargs) # For ASE calculator ... 
        
        self.num_GPUs = torch.cuda.device_count()
        self.model_paths = model_paths
        self.froce_variances = []
        self.energy_variance = []
    
    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        
        super().calculate(atoms, properties, system_changes) # For ASE calculator ...
        
        models_energies = []
        models_forces = []
        
        # Doing calculation with all models:
        if __name__ == '__main__':
            
            mp.set_start_method('spawn')
            
            models_forces = []
            models_energues = []
            
            print('0', flush = True)
            
            with mp.Manager() as manager:
                
                print('1', flush = True)

                models_forces = manager.Array('f', (len(self.atoms), 3))
                models_energies = manager.Array('f', 1)

                print('2', flush = True)
            
                processes = []
                for path in self.model_paths:
                    p = mp.Process(
                                    target=self.NequIP_calculation,
                                    args = (self.atoms.copy(), path, models_energies, models_forces, len(processes))
                                   )
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()
                    
            print(models_forces, flush = True)
                               
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
            
            
    def NequIP_calculation(self, atoms, path, models_energies, models_forces, key):
                               
        model = NequIPCalculator.from_deployed_model(
            model_path = path,
            species_to_type_name = {"Pt": "Pt", "Ti": "Ti", "O": "O"},
            device = device
        )
        
        atoms.calc = model
        energy = atoms.get_potential_energy() # Getting energy
        forces = atoms.get_forces()
        
        models_energies[key] = energy
        models_forces[key] = forces
        

    def force_calculations(self, models_forces):

        M = len(models_forces) # Number of models

        # The mean of each force component (x, y and z):
        mean_components = np.mean(models_forces, axis = 0)

        # Computing all variances in a few lines:
        variances = np.sum((models_forces - mean_components)**2, axis = 2)
        variances = np.sum(variances, axis = 0)/M

        return mean_components, variances
