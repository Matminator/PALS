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


class PALS_basic_ensemble(Calculator):
    implemented_properties = ['energy', 'energies', 'forces', 'stress', 'stresses']
    
    def __init__(self, model_paths, species_names, device = 'cpu', store_all = False, dir_name = None, logo = True, *args, **kwargs):
        """
        model_paths: paths to NequIP models.
        device: cup or cuda (cuda for GPU)
        store_all: if True will store all model calculations as ASE traj files.
        dir_name: If store all is true, this is the needed folder to save to.
        """
        super().__init__(*args, **kwargs) # For ASE calculator ... 

        if logo:
            self.print_logo()
        
        # importing NequIP models:
        print('*** Initializing models ***********************', flush = True)
        start_time = time.time()
        
        models = []
        for path in model_paths:
            model = NequIPCalculator.from_deployed_model(
                model_path = path,
                species_to_type_name = species_names, # {"Pt": "Pt", "Ti": "Ti", "O": "O"},
                device = device
                )
            models.append(model)
        self.models = models
                    
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Initializing time: ", elapsed_time, flush = True)
        print('***********************************************', flush = True)
            
        # Making lists to save model variances:
        self.force_variance = []
        self.energy_variance = []
        
        # Setting up store all, if true:
        self.store_all = store_all                
        if store_all:
            self.historic_atoms = [None] * len(models) # For saving all forces and energies
    
            model_logs = []
            for i in range(len(models)):
                name = dir_name + '/model_' + str(i) +'.traj'
                out = Trajectory(name, 'w')
                model_logs.append(out)
                
            self.model_logs = model_logs
    
    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes) # For ASE calculator ...
        
        models_energies = []
        models_forces = []
        
        # Doing calculation with all models:
        for i in range(len(self.models)):
            model = self.models[i]
            
            self.NequIP_calculation(self.atoms.copy(),
                                    model,
                                    models_energies,
                                    models_forces,
                                    i)
        
        if self.store_all:
            for i in range(len(self.models)):
                out = self.model_logs[i]
                out.write(self.atoms,
                          forces = models_forces[i],
                          energy = models_energies[i]
                         )
            
        # Computing resulting froce and variances
        forces_mean, force_variance = self.force_calculations(models_forces)
        energy_mean = np.mean(models_energies)
        energy_variance = np.var(models_energies)
        
        self.force_variance.append(force_variance)
        self.energy_variance.append(energy_variance)
        
        self.results = {'energy': energy_mean,
                        'energies': np.zeros(len(self.atoms)), 
                        'forces': forces_mean,
                        'stress': np.zeros(6),
                        'stresses': np.zeros((len(self.atoms), 6))}            
            
            
    def NequIP_calculation(self, atoms, model, models_energies, models_forces, i):
        
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

    def read_out(self, dir_name):
        
#         if not self.store_all:
#             raise 'Store all set to False'
        
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name) # Making directory
                
        name = dir_name + "/energy_variance.csv"
        pd.DataFrame(self.energy_variance).to_csv(name)
        name = dir_name + "/force_variance.csv"
        pd.DataFrame(self.force_variance).to_csv(name)

        
        
    def print_logo(self):
        print('')
        print(r"__/\\\\\\\\\\\\\_______/\\\\\\\\\______/\\\__________________/\\\\\\\\\\\_________") 
        print(r"__\/\\\/////////\\\___/\\\\\\\\\\\\\___\/\\\________________/\\\/////////\\\______")   
        print(r"___\/\\\_______\/\\\__/\\\/////////\\\__\/\\\_______________\//\\\______\///______")
        print(r"____\/\\\\\\\\\\\\\/__\/\\\_______\/\\\__\/\\\________________\////\\\____________")     
        print(r"_____\/\\\/////////____\/\\\\\\\\\\\\\\\__\/\\\___________________\////\\\________")     
        print(r"______\/\\\_____________\/\\\/////////\\\__\/\\\______________________\////\\\____")    
        print(r"_______\/\\\_____________\/\\\_______\/\\\__\/\\\_______________/\\\______\//\\\__")   
        print(r"________\/\\\_____________\/\\\_______\/\\\__\/\\\\\\\\\\\\\\\__\///\\\\\\\\\\\/__")  
        print(r"_________\///______________\///________\///___\///////////////_____\///////////___") 
        print('PALS base ensamble version:', 1.2, '\n',flush = True)
