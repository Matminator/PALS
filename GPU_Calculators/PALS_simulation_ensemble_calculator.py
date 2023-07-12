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

class PALS_simulation_ensemble(Calculator):
    implemented_properties = ['energy', 'energies', 'forces', 'stress', 'stresses']
    
    def __init__(self, model_paths, save_sigmas_to,
                 device = 'cuda', save_feq = 10, 
                 element_dict =  {"Pt": "Pt", "Ti": "Ti", "O": "O"},
                 logo = True, *args, **kwargs):
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
                species_to_type_name = element_dict,
                device = device
                )
            models.append(model)
        self.models = models
                    
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Initializing time: ", elapsed_time, flush = True)
        print('***********************************************', flush = True)
            
        self.save_sigmas_to = save_sigmas_to
            
        self.F_sigmas_DF = pd.DataFrame(columns = ['max', '99.9', '99', '95', '90',
                                                   'median', 'mean'])
        self.E_sigma_DF = pd.DataFrame(columns = ['E'])
        
        self.F_sigmas_DF.to_csv(self.save_sigmas_to + '_F_sigmas.csv', index=False)
        self.E_sigma_DF.to_csv(self.save_sigmas_to + '_E_sigma.csv', index=False)
        
        self.save_feq = save_feq
        self.sss = 0 # Steps since save
        
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
            
        # Computing resulting froce and variances
        forces_mean, force_variance = self.force_calculations(models_forces)
        energy_mean = np.mean(models_energies)
        energy_variance = np.var(models_energies)
        
        Fs = np.sqrt(force_variance)
        Es = np.sqrt(energy_variance)
        self.F_sigmas_DF.loc[len(self.F_sigmas_DF)] = [np.max(Fs),
                                         np.percentile(Fs, 99.9),
                                         np.percentile(Fs, 99),
                                         np.percentile(Fs, 95),
                                         np.percentile(Fs, 90),
                                         np.median(Fs),
                                         np.mean(Fs)
                                        ]
        self.E_sigma_DF.loc[len(self.E_sigma_DF)]  = [Es]
        
        self.sss += 1
        if self.sss == self.save_feq:
            
            print('PALS: saving sigmas, at given frequenz:', self.save_feq, flush = True)
            
            F_sigma_saves = pd.read_csv(self.save_sigmas_to + '_F_sigmas.csv')
            E_sigma_saves = pd.read_csv(self.save_sigmas_to + '_E_sigma.csv')

            F_sigma_saves = F_sigma_saves.append(self.F_sigmas_DF)
            E_sigma_saves = E_sigma_saves.append(self.E_sigma_DF)

            F_sigma_saves.to_csv(self.save_sigmas_to + '_F_sigmas.csv', index=False)
            E_sigma_saves.to_csv(self.save_sigmas_to + '_E_sigma.csv', index=False)

            self.F_sigmas_DF = pd.DataFrame(columns = ['max', '99.9', '99', '95', '90',
                                                       'median', 'mean'])
            self.E_sigma_DF = pd.DataFrame(columns = ['E'])

            self.sss = 0
        
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
        print('PALS simulation ensamble version:', 1.6, '\n',flush = True)
