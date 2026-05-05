"""STOCHASTIC ROSS Module.

This module creates random rotor instances and run stochastic analysis.
"""
#from collections.abc import Iterable

import ross as rs
import numpy as np
#import scipy as sp


#from rossT.units import check_units

from st_distributions import ST_Distribution
from st_results import (st_Frequency, st_Time, st_Campbell,st_Forced)

class ST_Rotor2():
    #from ross.rotor_assembly import Rotor
 
    def __init__(
        self,
        list_rotor,
    ):
        self.shaft = list_rotor[0]
        self.disks = list_rotor[1]
        self.bearings = list_rotor[2]

        self.Bearings = []
        self.Disks= []
        self.Points = []

        if len(list_rotor)==4:
            self.points = list_rotor[3]
        else:
            self.points = []

        is_random =set()
        
        for i in self.bearings:
            try:
                if type(i.is_random[0]) == str:                
                    is_random.add('bearing')
            except:
                break

        for i in self.disks: ##adicionar material
            try:
                if type(i.is_random[0]) == str:                
                    is_random.add('disk')
            except:
                break

        for i in self.shaft: ##adicionar material
            try:
                if type(i.is_random[0]) == str:                
                    is_random.add('shaft')
            except:
                break
        if len(list_rotor)==4:
            for i in self.points: ##adicionar material
                try:
                    if type(i.is_random[0]) == str:                
                        is_random.add('point_mass')
                except:
                    break
                
        self.is_random = list(is_random)
       
    def build_rotor(self):
        if 'bearing' in self.is_random:
            list_bearings = []
            for i in range(len(self.bearings)):
                list_bearings.append(list(iter(self.bearings[i]))[0])
            self.Bearings = list_bearings
            
        else:
            self.Bearings = self.bearings
                
        if 'disk' in self.is_random:
            list_disks = []
            for i in range(len(self.disks)):
                list_disks.append(list(iter(self.disks[i]))[0])
            self.Disks = list_disks
        else:
            self.Disks = self.disks

        if 'point_mass' in self.is_random:
            list_points = []
            for i in range(len(self.points)):
                list_points.append(list(iter(self.points[i]))[0])
            self.Points = list_points
        
        else:
            self.Points = self.points
            
        #Rotor_built = Rotor(self.shaft, self.disks, list_bearings)
        try:
            Rotor_built = rs.Rotor(self.shaft, self.Disks, self.Bearings,self.Points)

        except:
            Rotor_built = rs.Rotor(self.shaft, self.Disks, self.Bearings)
        
        return Rotor_built

        #como armazenar os valores aleatórios?
        #valores de rigidez variando na frequência

        if 'bearing' in self.is_random:
            list_bearings = []
            for i in range(len(self.bearings)):
                list_bearings.append(list(iter(self.bearings[i]))[0])
            self.Bearings = list_bearings
            
        else:
            self.Bearings = self.bearings

        if 'shaft' in self.is_random:
            list_shafts = []
            for i in range(len(self.shaft)):
                if isinstance(i, ST_Distribution):
                    material = list(iter(material))[0]
                    list_shafts.append(list(iter(self.shaft[i]))[0]) 
                    #ver como iterar e ver qual é o incerto
            self.shaft = list_shafts
            
        else:
            self.Bearings = self.bearings
                
        if 'disk' in self.is_random:
            #somente um disco aleatório?
            list_disks = []
            for i in range(len(self.disks)):
                list_disks.append(list(iter(self.disks[i]))[0])
            self.Disks = list_disks
        else:
            self.Disks = self.disks

        if 'point_mass' in self.is_random:
            list_points = []
            for i in range(len(self.points)):
                list_points.append(list(iter(self.points[i]))[0])
            self.Points = list_points
        
        else:
            self.Points = self.points
            

        Rotor_built = rs.Rotor(self.shaft, self.Disks, self.Bearings,self.Points)
        return Rotor_built


    
    def plot_rotor(self):
        Rotor_built = self.build_rotor()
        return Rotor_built.plot_rotor()
    
    def number_dof(self):

        Rotor_built = self.build_rotor()
        return Rotor_built.number_dof

    def ndof(self):
        
        Rotor_built = self.build_rotor()
        return int(Rotor_built.ndof)

    def nodes_pos(self):
        
        Rotor_built = self.build_rotor() #pos pode variar
        return Rotor_built.nodes_pos
    
    def nodes(self):
        
        Rotor_built = self.build_rotor()
        return Rotor_built.nodes
    
    def link_nodes(self):
        
        Rotor_built = self.build_rotor()
        return Rotor_built.link_nodes

    def run_stFreq(
        self,
        inp,
        out,
        NMC,
        speed_range=None,
        modes=None,
        cluster_points=False,
        num_modes=12,
        num_points=10,
        rtol=0.005,
    ):

        FRF_size = len(speed_range)
        freq_resp = np.empty((FRF_size, NMC), dtype=complex)
        velc_resp = np.empty((FRF_size, NMC), dtype=complex)
        accl_resp = np.empty((FRF_size, NMC), dtype=complex)

        # Monte Carlo - results storage
        for i in range(NMC):
            rotor = self.build_rotor()
            results = rotor.run_freq_response(
                speed_range,
                modes,
                cluster_points,
                num_modes,
                num_points,
                rtol,
            )

            freq_resp[:, i] = results.freq_resp[inp, out, :]
            velc_resp[:, i] = results.velc_resp[inp, out, :]
            accl_resp[:, i] = results.accl_resp[inp, out, :]
        
        results = st_Frequency(
            speed_range, 
            freq_resp, 
            velc_resp, 
            accl_resp
        )

        return results    

    def run_stCampbell(self, 
                       speed_range,
                       NMC, 
                       frequencies=6, 
                       frequency_type="wd", 
                       ):
        self.speed_range = speed_range
        self.NMC = NMC
        CAMP_size = len(speed_range)
        
        wd = np.zeros((frequencies, CAMP_size, NMC))
        log_dec = np.zeros((frequencies, CAMP_size, NMC))
        for y in range(NMC):
            rotor_case = self.build_rotor()
            
            result = rotor_case.run_campbell(speed_range, frequencies, frequency_type)
            for j in range(frequencies):
                wd[j, :, y] = result.wd[:, j]
                log_dec[j, :, y] = result.log_dec[:, j]

        results = st_Campbell(
            speed_range, 
            wd, 
            log_dec)
        return results
    
    def run_stTime(
                   self, 
                   speed, 
                   force, 
                   time, 
                   NMC): #TIME RESPONSE
        self.time = time
        self.force = force
        self.speed = speed
        self.NMC = NMC

        #nodes = self.nodes()

        xout = np.zeros((NMC, len(time), 2 * self.ndof())) #PQ 2NDOF
        yout = np.zeros((NMC, len(time), self.ndof()))
        
        #resp_stochTime = np.zeros((NMC, len(time), ndof,2))
        resp_stochTime=[]

        # Monte Carlo - results storage
        for u in range(self.NMC):
            rotor_case = self.build_rotor()
            
            response = rotor_case.run_time_response(self.speed, self.force, self.time)
            xout[u] = response.xout
            yout[u] = response.yout

        
        resp_stochTime.append(xout)
        resp_stochTime.append(yout)
        self.resp_stochTime = resp_stochTime

        results = st_Time(
            self.time,
            self.resp_stochTime,
            self.number_dof(),
            self.nodes_pos(),
            )#link_nodes=self.link_nodes)#sem link nodes
        return results  
    
    def run_stUnbalance(
        self,
        node,
        unbalance_magnitude,
        unbalance_phase,
        NMC,
        frequency_range=None,
        modes=None,
        cluster_points=False,
        num_modes=12,
        num_points=10,
        rtol=0.005,
    ):
        
        freq_size = len(frequency_range)
        ndof = self.ndof

        forced_resp = np.zeros((NMC, ndof, freq_size), dtype=complex)
        velc_resp = np.zeros((NMC, ndof, freq_size), dtype=complex)
        accl_resp = np.zeros((NMC, ndof, freq_size), dtype=complex)
        
        if type(unbalance_magnitude.is_random[0]) == str:                
            self.is_random.add('unbalance_magnitude')
        
        if type(unbalance_phase.is_random[0]) == str:                
            self.is_random.add('unbalance_phase')


        # Monte Carlo - results storage
        for i in range(NMC):
            if 'unbalance_magnitude' in self.is_random:
                unmag = unbalance_magnitude.value(1)[0]
            else:
                unmag = unbalance_magnitude

            if 'unbalance_phase' in self.is_random:
                unphase = unbalance_phase.value(1)[0]

            else:
                unphase = unbalance_phase

            rotor = self.build_rotor()
            results = rotor.run_unbalance_response(
                    node, 
                    unmag, 
                    unphase, 
                    frequency_range,
                    modes=modes,
                    cluster_points=cluster_points,
                    num_modes=num_modes,
                    num_points=num_points,
                    rtol=rtol,
                )

            forced_resp[:, i] = results.forced_resp[inp, out, :]
            velc_resp[:, i] = results.velc_resp[inp, out, :]
            accl_resp[:, i] = results.accl_resp[inp, out, :]
        

        results = st_Forced(
            forced_resp=forced_resp,
            frequency_range=frequency_range,
            velc_resp=velc_resp,
            accl_resp=accl_resp,
            number_dof=self.number_dof,
            nodes=self.nodes,
            link_nodes=self.link_nodes,
        )

        return results
