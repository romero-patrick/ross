"""Make it stochastic module.

This module convert an deterministic rotor to stochastic rotor.
"""

import numpy as np
import copy as cp
import numbers
import random

from ross.rotor_assembly import Rotor
from ross.materials import Material
from ross.bearing_seal_element import BearingElement
from ross.shaft_element import ShaftElement
from ross.disk_element import DiskElement

from st_distributions import ST_Distribution
from st_results import (st_Frequency, st_Time, st_Campbell,st_Forced)


from ross.units import check_units

class ST_Make_it_Stochastic():
    """Convert an deterministic object to stochastic.

    Class used to turn an deterministic object of the rotor into stochastic.

    Parameters
    ----------
    rotor : rotor in ROSS
        Rotor model in ROSS type.
    elements : str, list
        The random element or a a list of random elements.
    param : str, list
        The random parameter or a list of random parameters.
    

    Examples
    --------
    
    """

    @check_units
    def __init__(
        self, rotor, elements, params, distribution = 'Normal',erro = 5/100, **kwargs
    ):
        self.rotor = rotor
        self.elements = elements
        if " " in elements:
            raise ValueError("Spaces are not allowed in Element name")
        self.params = params
        self.distribution = distribution
        self.erro = erro
        
        attribute_dict = dict(
            rotor=rotor,
            elements=elements,
            params=params,
            distribution = distribution,
            erro=erro,
        )
        self.attribute_dict = attribute_dict


    def pickvalues(self):
        """Evaluate an array with values of parameters.

        """

        valueslist =[]
        if len(self.elements) == 1:
            attr = getattr(self.rotor, self.elements[0])
            values = np.zeros((len(attr),len(self.params[0])))
            for i in range(len(attr)):
                for j in range(len(self.params[0])):
                    attr2 = getattr(attr[i], self.params[0][j])
                    values[i][j] = attr2[0]

            valueslist.append(values)

        else:
            for z in range(len(self.elements)):
                if self.elements[z] == 'shaft_materials':
                    attr = getattr(self.rotor, 'shaft_elements')
                    values = np.zeros((len(attr),len(self.params[z])))
                    for i in range(len(attr)):
                        for j in range(len(self.params[z])):
                            attr2 = getattr(attr[i], 'material')
                            attr3 = getattr(attr2, self.params[z][j])
                            values[i][j] = attr3

                else:
                    attr = getattr(self.rotor, self.elements[z])
                    values = np.zeros((len(attr),len(self.params[z])))
                    for i in range(len(attr)):
                        for j in range(len(self.params[z])):
                            attr2 = getattr(attr[i], self.params[z][j])
                            if isinstance(attr2, list):
                                values[i][j] = attr2[0]   
                            if isinstance(attr2, numbers.Number):
                                values[i][j] = attr2

                valueslist.append(values) 

            '''
            for z in range(len(self.elements)):
                attr = getattr(self.rotor, self.elements[z])
                values = np.zeros((len(attr),len(self.params[z])))
                for i in range(len(attr)):
                    for j in range(len(self.params[z])):
                        attr2 = getattr(attr[i], self.params[z][j])
                        values[i][j] = attr2[0]   

                valueslist.append(values)   
            '''
        return valueslist   
   

    def limits(self):
        """Build the distributions.

        """

        if self.distribution == "Normal":
            distributions =[]
            values = self.pickvalues()
            for z in range(len(self.elements)):
                if self.elements[z] != 'shaft_materials':
                    listdist=[]            
                    attr = getattr(self.rotor, self.elements[z])
                    for i in range(len(attr)):
                        for j in range(len(self.params[z])):
                            std = values[z][i][j] * self.erro/2

                            listdist.append(ST_Distribution(name = self.distribution, 
                                                                info =[values[z][i][j],std],
                                                                param = self.params[z][j]))
                    distributions.append(listdist)

                else:
                    listdist=[]
                    attr = getattr(self.rotor, 'shaft_elements')
                    for i in range(len(attr)):
                        for j in range(len(self.params[z])):
                            std = values[z][i][j] * self.erro/2

                            listdist.append(ST_Distribution(name = self.distribution, 
                                                                info =[values[z][i][j],std],
                                                                param = self.params[z][j]))
                    distributions.append(listdist)
                
        elif self.distribution == "Uniform":
            distributions =[]
            values = self.pickvalues()
            for z in range(len(self.elements)):
                if self.elements[z] != 'shaft_materials':
                    listdist =[]
                    attr = getattr(self.rotor, self.elements[z])
                    for i in range(len(attr)):
                        for j in range(len(self.params[z])):
                            limsup = values[z][i][j] *(1 + self.erro)
                            liminf = values[z][i][j] *(1 - self.erro)

                            listdist.append(ST_Distribution(name = self.distribution, 
                                                                info =[liminf,limsup-liminf],
                                                                param = self.params[z][j]))
                    distributions.append(listdist)
                else:
                    listdist=[]
                    attr = getattr(self.rotor, 'shaft_elements') 
                    for i in range(len(attr)):
                        for j in range(len(self.params[z])):
                            limsup = values[z][i][j] *(1 + self.erro)
                            liminf = values[z][i][j] *(1 - self.erro)

                            listdist.append(ST_Distribution(name = self.distribution, 
                                                                info =[liminf,limsup-liminf],
                                                                param = self.params[z][j]))
                    distributions.append(listdist)
                
        else:
            raise KeyError("Wrong Name: "+self.distribution+".")

        return distributions
    
    def limits_values(self):
        """Extreme values from each parameter.

        """

        ext_values = {}

        values = self.pickvalues()
        for z in range(len(self.elements)):
            if self.elements[z] != 'shaft_materials':
                attr = getattr(self.rotor, self.elements[z])
                for i in range(len(attr)):
                    for j in range(len(self.params[z])):
                        limsup = values[z][i][j] *(1 + self.erro)
                        liminf = values[z][i][j] *(1 - self.erro)

                        if i == 0:
                            ext_values[self.params[z][j]] = [[liminf,limsup]]

                        else:
                            ext_values[self.params[z][j]].extend([[0,0]])
                            ext_values[self.params[z][j]][i] = [liminf,limsup]



            else:
                attr = getattr(self.rotor, 'shaft_elements') 
                for i in range(len(attr)):
                    for j in range(len(self.params[z])):
                        limsup = values[z][0][j] *(1 + self.erro)
                        liminf = values[z][0][j] *(1 - self.erro)

                        if i == 0:
                            ext_values[self.params[z][j]] = [[liminf,limsup]]

                        else:
                            ext_values[self.params[z][j]].extend([[0,0]])
                            ext_values[self.params[z][j]][i] = [liminf,limsup]
                
        return ext_values

    
    def switch_rotor_values(self):
        ''' Modifing the chosen values of the rotor.
         
        '''
        
        distributions = self.limits()

        shaft = cp.deepcopy(self.rotor.shaft_elements)
        disks = cp.deepcopy(self.rotor.disk_elements)
        bearings = cp.deepcopy(self.rotor.bearing_elements)

        modified_rotor = Rotor(shaft, disks, bearings)
        
        for idk,k in enumerate(self.elements):
            if k=='bearing_elements':
                for i in range(len(modified_rotor.bearing_elements)):
                    for idj,j in enumerate(self.params[idk]):
                        try:
                            setattr(modified_rotor.bearing_elements[i], j, distributions[idk][idj].value(1))
                        except:
                            raise KeyError("Wrong Name: "+self.params[idk][j]+ " for "+ k+ ".")
        
            if k =='disk_elements' :
                for i in range(len(modified_rotor.disk_elements)):
                    for idj,j in enumerate(self.params[idk]):
                        try:
                            setattr(modified_rotor.disk_elements[i], j, distributions[idk][idj].value(1)[0])
                        except:
                            raise KeyError("Wrong Name: "+self.params[idk][j]+ " for "+ k+ ".")

            if k =='shaft_elements' :
                for i in range(len(modified_rotor.shaft_elements)):
                    for idj,j in enumerate(self.params[idk]):
                        try:
                            setattr(modified_rotor.shaft_elements[i], j, distributions[idk][idj].value(1)[0])
                        except:
                            raise KeyError("Wrong Name: "+self.params[idk][j]+ " for "+ k+ ".")

            if k =='shaft_materials' :
                for i in range(len(modified_rotor.shaft_elements)):
                    for idj,j in enumerate(self.params[idk]):
                        try:
                            attr2 = getattr(modified_rotor.shaft_elements[i], 'material')
                            setattr(attr2, j, distributions[idk][idj].value(1)[0])
                        except:
                            raise KeyError("Wrong Name: "+self.params[idk][j]+ " for "+ k+ ".")

        # declarando os mancais
        shaftlist = []

        for i in range(len(modified_rotor.shaft_elements)):
            material = Material(name = modified_rotor.shaft_elements[i].material.name,
                                rho = modified_rotor.shaft_elements[i].material.rho,
                                E = modified_rotor.shaft_elements[i].material.E,
                                G_s = modified_rotor.shaft_elements[i].material.G_s
                                )
            shaftlist.append(ShaftElement(L = modified_rotor.shaft_elements[i].L,
                                            idl = modified_rotor.shaft_elements[i].idl,
                                            odl = modified_rotor.shaft_elements[i].odl,
                                            material = material,
                                            shear_effects = modified_rotor.shaft_elements[i].shear_effects,
                                            rotary_inertia = modified_rotor.shaft_elements[i].rotary_inertia,
                                            gyroscopic = modified_rotor.shaft_elements[i].gyroscopic
                                            ))


        bearinglist = []

        for i in range(len(modified_rotor.bearing_elements)):
            bearinglist.append(BearingElement(n=modified_rotor.bearing_elements[i].n, 
                                        n_link=modified_rotor.bearing_elements[i].n_link ,
                                        kxx = modified_rotor.bearing_elements[i].kxx , 
                                        kyy = modified_rotor.bearing_elements[i].kyy , 
                                        cxx = modified_rotor.bearing_elements[i].cxx , 
                                        cyy = modified_rotor.bearing_elements[i].cyy , 
                                        mxx = modified_rotor.bearing_elements[i].mxx , 
                                        myy = modified_rotor.bearing_elements[i].myy ,
                                        frequency = modified_rotor.bearing_elements[i].frequency
                                        ))
            
        disklist = []

        for i in range(len(modified_rotor.disk_elements)):
            disklist.append(DiskElement(n = modified_rotor.disk_elements[i].n, 
                                        Id = modified_rotor.disk_elements[i].Id,
                                        Ip = modified_rotor.disk_elements[i].Ip,
                                        m = modified_rotor.disk_elements[i].m,
                                        scale_factor = modified_rotor.disk_elements[i].scale_factor,
                                        ))
            
        modified_rotor2 = Rotor(shaftlist, disklist, bearinglist)

        return modified_rotor2
    
    def extreme_samples(self, parameters, n_samples, seed):
        random.seed(seed)

        samples = []

        structure = {p: len(parameters[p]) for p in parameters}

        for _ in range(n_samples):
            combo = {}

            for param, n_sub in structure.items():
                combo[param] = []
                for j in range(n_sub):
                    bit = random.randint(0, 1)  # 0=min, 1=max
                    valor = parameters[param][j][bit]
                    combo[param].append(valor)

            samples.append(combo)

        return samples
    
    def switch_rotor_extremes(self,sample):
        ''' Modifing the chosen values of the rotor.
            
        '''
        
        shaft = cp.deepcopy(self.rotor.shaft_elements)
        disks = cp.deepcopy(self.rotor.disk_elements)
        bearings = cp.deepcopy(self.rotor.bearing_elements)

        modified_rotor = Rotor(shaft, disks, bearings)

        
        for idk,k in enumerate(self.elements):
            if k=='bearing_elements':
                for i in range(len(modified_rotor.bearing_elements)):
                    for j in self.params[idk]:
                        try:
                            setattr(modified_rotor.bearing_elements[i], j, sample[j][i])
                        except:
                            raise KeyError("Wrong Name: "+self.params[idk][j]+ " for "+ k+ ".")
        
            if k =='disk_elements' :
                for i in range(len(modified_rotor.disk_elements)):
                    for j in self.params[idk]:
                        try:
                            setattr(modified_rotor.disk_elements[i], j, sample[j][i])
                        except:
                            raise KeyError("Wrong Name: "+self.params[idk][j]+ " for "+ k+ ".")

            if k =='shaft_elements' :
                for i in range(len(modified_rotor.shaft_elements)):
                    for j in self.params[idk]:
                        try:
                            setattr(modified_rotor.shaft_elements[i], j, sample[j][i])
                        except:
                            raise KeyError("Wrong Name: "+self.params[idk][j]+ " for "+ k+ ".")

            if k =='shaft_materials' :
                for i in range(len(modified_rotor.shaft_elements)):
                    for j in self.params[idk]:
                        try:
                            attr2 = getattr(modified_rotor.shaft_elements[i], 'material')
                            setattr(attr2, j, sample[j][i])
                        except:
                            raise KeyError("Wrong Name: "+self.params[idk][j]+ " for "+ k+ ".")

        # declarando os mancais
        shaftlist = []

        for i in range(len(modified_rotor.shaft_elements)):
            material = Material(name = modified_rotor.shaft_elements[i].material.name,
                                rho = modified_rotor.shaft_elements[i].material.rho,
                                E = modified_rotor.shaft_elements[i].material.E,
                                G_s = modified_rotor.shaft_elements[i].material.G_s
                                )
            shaftlist.append(ShaftElement(L = modified_rotor.shaft_elements[i].L,
                                            idl = modified_rotor.shaft_elements[i].idl,
                                            odl = modified_rotor.shaft_elements[i].odl,
                                            material = material,
                                            shear_effects = modified_rotor.shaft_elements[i].shear_effects,
                                            rotary_inertia = modified_rotor.shaft_elements[i].rotary_inertia,
                                            gyroscopic = modified_rotor.shaft_elements[i].gyroscopic
                                            ))


        bearinglist = []

        for i in range(len(modified_rotor.bearing_elements)):
            bearinglist.append(BearingElement(n=modified_rotor.bearing_elements[i].n, 
                                        n_link=modified_rotor.bearing_elements[i].n_link ,
                                        kxx = modified_rotor.bearing_elements[i].kxx , 
                                        kyy = modified_rotor.bearing_elements[i].kyy , 
                                        cxx = modified_rotor.bearing_elements[i].cxx , 
                                        cyy = modified_rotor.bearing_elements[i].cyy , 
                                        mxx = modified_rotor.bearing_elements[i].mxx , 
                                        myy = modified_rotor.bearing_elements[i].myy ,
                                        frequency = modified_rotor.bearing_elements[i].frequency
                                        ))
            
        disklist = []

        for i in range(len(modified_rotor.disk_elements)):
            disklist.append(DiskElement(n = modified_rotor.disk_elements[i].n, 
                                        Id = modified_rotor.disk_elements[i].Id,
                                        Ip = modified_rotor.disk_elements[i].Ip,
                                        m = modified_rotor.disk_elements[i].m,
                                        scale_factor = modified_rotor.disk_elements[i].scale_factor,
                                        ))
            
        modified_rotor2 = Rotor(shaftlist, disklist, bearinglist)

        return modified_rotor2
        
        
    def just_to_see_Freq(self,
        inp,
        out,
        n_samples=10,
        speed_range=None,
        modes=None,
        cluster_points=False,
        seed=42,
        num_modes=12,
        num_points=10,
        rtol=0.005,
    ):
        
        values = self.limits_values()
        samples = self.extreme_samples(values,n_samples,seed)
      
        FRF_size = len(speed_range)
        freq_resp = np.empty((FRF_size, n_samples), dtype=complex)
        velc_resp = np.empty((FRF_size, n_samples), dtype=complex)
        accl_resp = np.empty((FRF_size, n_samples), dtype=complex)

        # Monte Carlo - results storage
        for i in range(n_samples):

            sample = samples[i]
            rotor_case = self.switch_rotor_extremes(sample)
            results = rotor_case.run_freq_response(
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
            rotor_case = self.switch_rotor_values()
            results = rotor_case.run_freq_response(
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
            rotor_case = self.switch_rotor_values()
            result = rotor_case.run_campbell(speed_range, frequencies, frequency_type)
            for j in range(frequencies):
                wd[j, :, y] = result.wd[:, j]
                log_dec[j, :, y] = result.log_dec[:, j]

        results = st_Campbell(
            speed_range, 
            wd, 
            log_dec)
        return results
    
    def run_stTime(self, 
                   speed, 
                   force, 
                   time, 
                   NMC): #TIME RESPONSE
        self.time = time
        self.force = force
        self.speed = speed
        self.NMC = NMC

        #nodes = self.nodes()

        xout = np.zeros((NMC, len(time), 2 * self.rotor.ndof)) #PQ 2NDOF
        yout = np.zeros((NMC, len(time), self.rotor.ndof))
        
        #resp_stochTime = np.zeros((NMC, len(time), ndof,2))
        resp_stochTime=[]

        # Monte Carlo - results storage
        for u in range(self.NMC):
            rotor_case = self.switch_rotor_values()
            response = rotor_case.run_time_response(self.speed, self.force, self.time)
            xout[u] = response.xout
            yout[u] = response.yout

        
        resp_stochTime.append(xout)
        resp_stochTime.append(yout)
        self.resp_stochTime = resp_stochTime

        results = st_Time(
            self.time,
            self.resp_stochTime,
            self.rotor.nodes,
            self.rotor.number_dof,
            self.rotor.nodes_pos,
            self.rotor.link_nodes)
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
        ndof = self.rotor.ndof

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

            rotor_case = self.switch_rotor_values()
            results = rotor_case.run_unbalance_response(
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
            number_dof=self.rotor.number_dof,
            nodes=self.rotor.nodes,
            link_nodes=self.rotor.link_nodes,
        )

        return results

                    