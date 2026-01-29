import numpy as np
from ross.units import check_units
import ross as rs
import matplotlib.pyplot as plt
from numpy.linalg import norm 
import copy
import scipy as sp
#from paretoset import paretoset
import pandas as pd
from scipy.signal import find_peaks
from ross.rotor_assembly import Rotor
import gc
from ross.stochastic.st_id_results import ST_ID_FrequencyResponseResults
from ross.stochastic.st_id_results import ST_ID_TimeResponseResults



__all__ = ["ST_ID_Rotor"]


class ST_ID_Rotor(Rotor):

    def __init__(
        self,
        shaft_elements,
        disk_elements=None,
        bearing_elements=None,
        point_mass_elements=None,
        min_w=None,
        max_w=None,
        rated_w=None,
        tag=None,
    ):
        self.shaft = shaft_elements
        self.disks = disk_elements
        self.bearings = bearing_elements
        self.points = point_mass_elements

        self.Bearings = []
        self.Disks= []
        self.Points = []
        self.Shaft = []
        aux_list = []
        self.materials = {}
        interval_materials = {}

        self.minimum = np.array([])
        self.maximum = np.array([])

        self.freq = np.empty(100)

        self.optimized = None


        self.all_params = {}
        self.elements_to_identify = 0

        #order elements
        order_b = []
        order_d = []
        order_pm = []

        if bearing_elements is not None:
            for b in self.bearings:
                order_b.append(b['n'])
            order_b.sort()

            for n in range(len(order_b)):
                for b in self.bearings:
                    if b['n'] == order_b[n]:
                        if b['tag'] == None:
                            b['tag'] = "Bearing " + str(n)
                    

                        if b.to_identify != None:
                            self.all_params[b['tag']] = b.param_values
                            self.elements_to_identify += 1
                            for value in b.interval_params.values():
                                self.minimum = np.append(self.minimum,value[0])
                                self.maximum = np.append(self.maximum,value[1])

                        self.Bearings.append(b)

        if disk_elements is not None:
            for d in self.disks:
                order_d.append(d['n'])
            order_d.sort()

            for n in range(len(order_d)):
                for d in self.disks:
                    if d['n'] == order_d[n]:
                        if d['tag'] == None:
                            d['tag'] = 'Disk ' + str(n)
                        
                        if d.to_identify != None:
                            if len(d.to_identify) == 1 and "material_id" in (d.to_identify):
                                pass
                            else:
                                self.all_params[d['tag']] = d.param_values
                                self.elements_to_identify += 1
                            if 'material_id' in d.to_identify:
                                aux_list.append(d)
                            for param, value in d.interval_params.items():
                                    self.minimum = np.append(self.minimum, value[0])
                                    self.maximum = np.append(self.maximum, value[1])

                        self.Disks.append(d)

        if point_mass_elements is not None:
            for pm in self.points:
                order_pm.append(pm['n'])
            order_pm.sort()

            for n in range(len(order_pm)):
                for pm in self.points:
                    if pm['n'] == order_pm[n]:
                        if pm['tag'] == None:
                            pm['tag'] = 'PointMass ' + str(n)

                        if pm.to_identify != None:
                            self.all_params[pm['tag']] = pm.param_values
                            self.elements_to_identify += 1
                            for value in pm.interval_params.values():
                                self.minimum = np.append(self.minimum, value[0])
                                self.maximum = np.append(self.maximum, value[1])

                        self.Points.append(pm)

        for n in range(len(self.shaft)):
            if self.shaft[n]['n'] == None:
                self.shaft[n]['n'] = n
            if self.shaft[n]['tag'] == None:
                if isinstance(self.shaft[n], (rs.stochastic.st_id_coupling_element.ST_ID_CouplingElement)):
                    self.shaft[n]['tag'] = 'Couple ' + str(n)
                else:
                    self.shaft[n]['tag'] = 'Shaft ' + str(n)
            if self.shaft[n].to_identify != None:
                if len(self.shaft[n].to_identify) == 1 and 'material_id' in (self.shaft[n].to_identify):
                    pass
                else:
                    if isinstance(self.shaft[n], (rs.stochastic.st_id_coupling_element.ST_ID_CouplingElement)):
                        self.all_params[self.shaft[n].attribute_dict['tag']] = self.shaft[n].param_values
                    else:
                        self.all_params[self.shaft[n]['tag']] = self.shaft[n].param_values
                if "material_id" in self.shaft[n].to_identify:
                    aux_list.append(self.shaft[n])
                for param, value in self.shaft[n].interval_params.items():
                    self.minimum = np.append(self.minimum, value[0])
                    self.maximum = np.append(self.maximum, value[1])
            self.Shaft.append(self.shaft[n])


        for e in aux_list:
            self.materials[e.attribute_dict['material'].name] = e.id_material.param_values
            interval_materials[e.attribute_dict['material'].name] = e.id_material.interval_params

        for mat in interval_materials.values():
            for v in mat.values():
                self.minimum = np.append(self.minimum, v[0])
                self.maximum = np.append(self.maximum, v[1])

        self.elem_w_material = aux_list

        self.all_params.update(self.materials)
    
        super().__init__(self.shaft, self.disks, self.bearings, self.points, min_w, max_w, rated_w, tag)
            
    def swarm(self,funcao, inter, it, npop, **kwargs):

        F =[]
        Xobj = []
        obj = 0

        nobj = 0 #contador
        nvar = len(inter[0])
        npop = npop
        beta = 2.0
        x = np.empty([npop,nvar])
        xmin = inter[0]
        xmax = inter[1]

        for i in range(nvar):
                x[:,i] = np.random.uniform(low = xmin[i], high = xmax[i], size = npop)

        #print(f'Tempo de inicialização = {self.fim-self.inicio} \n')
        f = np.array([funcao(xi, **kwargs)[0] for xi in x])
        obj = obj + npop
        popbest = np.copy(x)
        fpopbest = np.copy(f)
        ibest = np.argmin(f)
        pbest = np.copy(x[ibest])
        fbest = f[ibest]
        run = True

        while run == True:

            r1 = np.random.random(size=(npop, nvar))
            r2 = np.random.random(size=(npop, nvar))
            x += beta * r1 * (popbest - x) + beta * r2 * (pbest - x)
            x = np.where(x>0, x, x*-1)
            f = np.array([funcao(xi, **kwargs)[0] for xi in x])
            obj = obj+npop
            idx = f < fpopbest
            fpopbest[idx] = f[idx]
            popbest[idx, :] = x[idx, :]
            ibest = np.argmin(f)
            if f[ibest] < fbest:
                fbest = f[ibest]
                pbest = np.copy(x[ibest])
                F.append(fbest)
                Xobj.append(obj)

            nobj+=1

            if nobj >=it:
                run = False
            
            print("{:3d} {:15.8f} {:15.8f}".format(nobj, pbest[0], fbest))

        print(f'F_best = {fbest} \n X_best = {pbest} \n')

        return pbest, fbest 
    
    def build_rotor(self):

        list_bearings = []
        list_disks = []
        list_points = []
        list_shaft = []

        for b in self.Bearings:
            list_bearings.append(b.new_bearing())
        for d in self.Disks:
            list_disks.append(d.new_disk()) 
        for pm in self.Points:
            list_points.append(pm.new_PointMass())
        for s in self.Shaft:
            if isinstance(s, (rs.stochastic.st_id_coupling_element.ST_ID_CouplingElement)):
                list_shaft.append(s.new_coupling())
            else:
                list_shaft.append(s.new_shaft())

        Rotor_built = rs.Rotor(list_shaft, list_disks, list_bearings,list_points)
        return Rotor_built
    
    def plot_rotor(self):
        rotor = self.build_rotor()
        return rotor.plot_rotor()
    
    def update(self, x):

        count = 0

        for element in self.all_params:

            for bearings in self.Bearings:
                if bearings['tag'] == element:
                    for var in self.all_params[element]:
                        self.all_params[element][var] = x[count]
                        count += 1
                    bearings.param_values = self.all_params[element]

            for disks in self.Disks:
                if disks['tag'] == element:
                    for var in self.all_params[element]:
                        self.all_params[element][var] = x[count]
                        count += 1
                    disks.param_values = self.all_params[element]
    
            for point_masses in self.Points:
                if point_masses['tag'] == element:
                    for var in self.all_params[element]:
                        self.all_params[element][var] = x[count]
                        count += 1
                    point_masses.param_values = self.all_params[element]
                    
            for shafts in self.Shaft:
                if shafts['tag'] == element:
                    for var in self.all_params[element]:
                        self.all_params[element][var] = x[count]
                        count +=1
                    shafts.param_values = self.all_params[element]
                
            for m in self.materials:
               if m == element:
                   for k in self.all_params[element]:
                       for el in self.elem_w_material:     
                            if el.material['name'] == m:
                               el.material.param_values = self.all_params[element]
                       self.all_params[element][k] = x[count]
                       count += 1
    
    
    def run_idFreq(
            self,  
            data, 
            probes,
            speed_range,
            it = 50,
            npop = 20,
            modes=None,
            cluster_points=False,
            num_modes=12,
            num_points=10,
            rtol=0.005, ):

        parameters, f_best = self.swarm(self.obj_freq, np.array([self.minimum,self.maximum]), it, npop, data=data, probes=probes,speed_range=speed_range,modes=modes,cluster_points=cluster_points,num_modes=num_modes,num_points=num_points,rtol=rtol)

        self.update(parameters)
        rotor = self.build_rotor()

        results = rotor.run_freq_response(
                speed_range,
                modes,
                cluster_points,
                num_modes,
                num_points,
                rtol,
            )
        
        fr = []

        self.optimized = parameters
        
        for i in range(len(probes)):
            frfx = results.freq_resp[probes[i][0]*6+0, probes[i][1]*6+0]*np.cos(np.radians(probes[i][2]))
            frfy = results.freq_resp[probes[i][0]*6+1, probes[i][1]*6+1]*np.sin(np.radians(probes[i][2]))
            fr.append(frfx + frfy)

        results = ST_ID_FrequencyResponseResults(freq_resp = results.freq_resp, velc_resp = results.velc_resp, accl_resp = results.accl_resp, speed_range = speed_range, number_dof = rotor.number_dof, data = data, probes = probes, optimum = self.optimized)


        return results
    
    def obj_freq(self,
                 x,
                 **kwargs):
    
        
        data = kwargs['data']
        probes = kwargs['probes']
        speed_range = kwargs['speed_range']
        modes = kwargs['modes']
        cluster_points = kwargs['cluster_points']
        num_modes = kwargs['num_modes']
        num_points = kwargs['num_points']
        rtol = kwargs['rtol']

        self.update(x)
        
        rotor = self.build_rotor()

        results = rotor.run_freq_response(
                speed_range,
                modes,
                cluster_points,
                num_modes,
                num_points,
                rtol,
        )

        resps = []

        for i in range(len(probes)):
            frfx = results.freq_resp[probes[i][0]*6+0, probes[i][1]*6+0]*np.cos(np.radians(probes[i][2]))
            frfy = results.freq_resp[probes[i][0]*6+1, probes[i][1]*6+1]*np.sin(np.radians(probes[i][2]))
            resps.append(frfx + frfy)
            
        resps = np.array(resps)
        aux_var = copy.deepcopy(resps)

        if isinstance(data, np.ndarray):
            pass
        else:
            data = np.array(data)

        phase = np.angle(resps)
        phase_data = np.angle(data)

        for i in range(len(phase)):
            phase[i] = np.where(phase[i] > 0, phase[i], phase[i]+ 2*np.pi)
            phase_data[i] = np.where(phase_data[i] >0, phase_data[i], phase_data[i] + 2*np.pi)
        
        resps2 = np.sqrt(resps.real**2 + resps.imag**2)

        d2 = np.sqrt(data.real**2+data.imag**2)

        fr = 20*np.log10(resps2)
        d2 = 20*np.log10(d2)

        del rotor
        gc.collect()

        r = norm(abs(d2-fr)**2)**2/norm(d2)**2 #+ 0.2*norm(abs(phase_data-phase)**2)**2/norm(phase_data)**2
        
        return r, aux_var
    
    def obj_time(self,
                 x,
                 **kwargs
                 ):

        data = kwargs['data']
        speed = kwargs['speed']
        force = kwargs['force']
        time = kwargs['time']
        method = kwargs['method']
        probes = kwargs['probes']
        kwargs.pop('data', None)
        kwargs.pop('speed', None)
        kwargs.pop('force', None)
        kwargs.pop('time', None)
        kwargs.pop('method', None)
        kwargs.pop('probes', None)

        self.update(x)

        rotor = self.build_rotor()

        run_time = rotor.run_time_response(speed, force, time,  method = method, **kwargs) 

        resps = []

        for i in range(len(probes)):
            y = run_time.yout[:,probes[i][0]*6]*np.cos(np.radians(probes[i][1])) + run_time.yout[:,probes[i][0]*6+1]*np.sin(np.radians(probes[i][1]))
            resps.append(y)

        resps = np.array(resps)

        if isinstance(data, np.ndarray):
            pass
        else:
            data = np.array(data)


        #data_fft = np.fft.rfft(data)
        #fft = np.fft.rfft(resps)

        result = (norm(data-resps)**2/norm(data)**2)# + 0.5*(norm(data_fft-fft)**2/norm(data_fft)**2)

        return result, resps
    
    def run_idTime(
            self, 
            data,
            speed,
            force,
            time,
            probes,
            method = 'default',
            it = 50,
            npop = 20,
            **kwargs
            ):
        
        self.data_time = data
        self.Force = force
        self.time = time
        self.speed = speed
        self.probes = probes
                
        parameters, f_best = self.swarm(self.obj_time, np.array([self.minimum,self.maximum]), it, npop, data = data, speed = speed, force = force, time = time, method = method, probes = probes, **kwargs)

        self.update(parameters)
        rotor = self.build_rotor()

        self.optimized = parameters

        run_time = rotor.run_time_response(speed, force, time, method = 'Newmark')

        results = ST_ID_TimeResponseResults(rotor, time, run_time.yout, run_time.xout, probes = probes, data = data, optimum = self.optimized)

        return results

    def run_SMC_ABCFreq(
            self,  
            data, 
            probes,
            speed_range,
            rate,
            interval,
            nsample = 100,
            optimized = True,
            modes=None,
            cluster_points=False,
            num_modes=12,
            num_points=10,
            rtol=0.005,
    ):
        
        n_var = len(self.minimum)
        var = np.empty([n_var,1])
        Samples = np.empty([1,len(speed_range)])
        accrate = 0
        fbest = 1
        xbest = []
        rates = len(rate)
        w = np.ones(nsample)/nsample
        

        if optimized == True:
            minimum = np.array([])
            maximum = np.array([])
            reduction = interval[0]
            amplifie = interval[1]
            for params in self.all_params.values():
                for value in params.values():
                    minimum = np.append(minimum, reduction*value)
                    maximum = np.append(maximum, amplifie*value)
            self.minimum = minimum
            self.maximum = maximum
        else:
            pass

        p_prior = np.prod(1/(self.maximum-self.minimum))

        print(f'Valor mínimo dos parâmetros: {self.minimum} \n')
        print(f'Valor máximo dos parâmetros: {self.maximum} \n')

        print(f'Iniciando o loop de iteração \n')

        prior = np.empty([n_var,nsample])

        for i in range(n_var):
                prior[i,:] = np.random.uniform(low = self.minimum[i], high = self.maximum[i], size = nsample)

        self.update(prior)
        self.prior = copy.deepcopy(self.all_params)

        for j in range(rates):
            print(f'Gen: {j}')
            var = np.empty([n_var,nsample])
            Samples = np.empty([nsample, len(data),len(speed_range)], dtype = complex)
            w_aux = np.empty(len(w))
            f_aux = []
            i = 0
            print(f'Valor de i = {i}')

            if j == 0:

                while i < nsample:

                    x = np.empty([n_var])
                    for z in range(n_var):
                        x[z] = np.random.uniform(self.minimum[z], self.maximum[z])
                    
                    r, freqs = self.obj_freq(x, data=data, probes=probes, speed_range=speed_range, modes=modes, cluster_points=cluster_points, num_modes=num_modes, num_points=num_points, rtol=rtol)

                    print(f'Valor de r = {r}')

                    if r < rate[j]:
                        accrate += 1
                        var[:, i] = x
                        Samples[i] = freqs
                        f_aux.append(r)
                        print(f'Amostra aceita. Valor = {r}')
                        i = i+1
                        print(f'Valor de i atualizado = {i}')
                        if r < fbest:
                            fbest = r
                            xbest = x

                next_prior = copy.deepcopy(var)

            else:

                while i < nsample:

                    x_old = next_prior[:,np.random.choice(nsample, p = w)]
                    next_prior_normalized = self.normalize_prior(next_prior.T).T
                    cov = np.cov(next_prior_normalized, aweights = w)
                    x_old_mean = self.normalize(x_old)
                    x = np.random.multivariate_normal(x_old_mean, cov)
                    x = self.denormalize(x)

                    if np.all(self.minimum <= x) and np.all(x <= self.maximum):

                        r, freqs = self.obj_freq(x, data=data, probes=probes, speed_range=speed_range, modes=modes, cluster_points=cluster_points, num_modes=num_modes, num_points=num_points, rtol=rtol)

                        if r < rate[j]:
                            w_aux[i] = self.weights(w, p_prior, next_prior_normalized, x, cov)
                            accrate += 1
                            var[:, i] = x
                            Samples[i] = freqs
                            f_aux.append(r)
                            i = i+1
                            print(f'Valor de i atualizado = {i}')
                            if r < fbest:
                                fbest = r
                                xbest = x

                    else:
                        pass

                next_prior = copy.deepcopy(var)
                w = w_aux/sum(w_aux)

        self.update(var)
        distributions = copy.deepcopy(self.all_params)
        
        best = xbest

        self.update(best)
        rotor = self.build_rotor()

        results_MAP = rotor.run_freq_response(
                    speed_range,
                    modes,
                    cluster_points,
                    num_modes,
                    num_points,
                    rtol,
                )

        fr = []

        for i in range(len(probes)):
                frfx = results_MAP.freq_resp[probes[i][0]*6+0, probes[i][1]*6+0]*np.cos(np.radians(probes[i][2]))
                frfy = results_MAP.freq_resp[probes[i][0]*6+1, probes[i][1]*6+1]*np.sin(np.radians(probes[i][2]))
                fr.append(frfx + frfy)

        best_response = np.array(fr)
        
        self.samples = Samples
        self.best_response = best_response

        if isinstance(self.optimized, np.ndarray):

            self.update(self.optimized)

            rotor = self.build_rotor()

            results = rotor.run_freq_response(
                        speed_range,
                        modes,
                        cluster_points,
                        num_modes,
                        num_points,
                        rtol,
                    )
        else:
            results = results_MAP

        return ST_ID_FrequencyResponseResults(results.freq_resp, results.velc_resp, results.accl_resp, speed_range, rotor.number_dof, data = data, probes = probes, optimum = self.optimized, 
                                              curves = Samples, best_curve = best_response, distributions = distributions,
                                              MAP_freq_resp = results_MAP.freq_resp,
                                              MAP_velc_resp = results_MAP.velc_resp,
                                              MAP_accl_resp = results_MAP.accl_resp,
                                              Prior = self.prior)
    
    def normalize_prior(self, x):
        x_norm = []
        for v in x:
            x_norm.append((v-self.minimum)/(self.maximum-self.minimum))
        return np.array(x_norm)
    
    def normalize(self, x):
        x_norm = (x-self.minimum)/(self.maximum-self.minimum)
        return x_norm
    
    def denormalize(self, x_norm):
        x = x_norm*(self.maximum-self.minimum)+self.minimum
        return x
    
    def weights(self, w, p_prior, next_prior_normalized, x, cov):
        x_normalized = self.normalize(x)
        vector = np.empty(len(w))

        for i, ws in enumerate(w):
            pdf = sp.stats.multivariate_normal.pdf(x_normalized, mean = next_prior_normalized[:,i], cov = cov, allow_singular = True)
            vector[i] = ws*pdf
        
        return p_prior/sum(vector)

    
    def run_SMC_ABCTime(
            self,  
            data,
            speed,
            force,
            time,
            probes,
            rate,
            interval,
            nsample = 100,
            method = 'default',
            optimized = True,
            **kwargs
    ):

        n_var = len(self.minimum)
        var = np.empty([n_var,1])
        Samples = np.empty([1,len(time)])
        accrate = 0
        fbest = 1
        xbest = []
        rates = len(rate)
        w = np.ones(nsample)/nsample
        

        if optimized == True:
            minimum = np.array([])
            maximum = np.array([])
            reduction = interval[0]
            amplifie = interval[1]
            for params in self.all_params.values():
                for value in params.values():
                    minimum = np.append(minimum, reduction*value)
                    maximum = np.append(maximum, amplifie*value)
            self.minimum = minimum
            self.maximum = maximum
        else:
            pass

        p_prior = np.prod(1/(self.maximum-self.minimum))

        print(f'Valor mínimo dos parâmetros: {self.minimum} \n')
        print(f'Valor máximo dos parâmetros: {self.maximum} \n')

        print(f'Iniciando o loop de iteração \n')

        prior = np.empty([n_var,nsample])

        for i in range(n_var):
                prior[i,:] = np.random.uniform(low = self.minimum[i], high = self.maximum[i], size = nsample)

        self.update(prior)
        self.prior = copy.deepcopy(self.all_params)

        for j in range(rates):
            print(f'Gen: {j}')
            var = np.empty([n_var,nsample])
            Samples = np.empty([nsample, len(data),len(time)], dtype = complex)
            w_aux = np.empty(len(w))
            f_aux = []
            i = 0
            print(f'Valor de i = {i}')

            if j == 0:

                while i < nsample:

                    x = np.empty([n_var])
                    for z in range(n_var):
                        x[z] = np.random.uniform(self.minimum[z], self.maximum[z])
                    
                    r, response = self.obj_time(x, data = data, speed = speed, force = force, time = time, method = method, probes = probes, **kwargs)

                    print(f'Valor de r = {r}')

                    if r < rate[j]:
                        accrate += 1
                        var[:, i] = x
                        Samples[i] = response
                        f_aux.append(r)
                        print(f'Amostra aceita. Valor = {r}')
                        i = i+1
                        print(f'Valor de i atualizado = {i}')
                        if r < fbest:
                            fbest = r
                            xbest = x

                next_prior = copy.deepcopy(var)

            else:

                while i < nsample:

                    x_old = next_prior[:,np.random.choice(nsample, p = w)]
                    next_prior_normalized = self.normalize_prior(next_prior.T).T
                    cov = np.cov(next_prior_normalized, aweights = w)
                    x_old_mean = self.normalize(x_old)
                    x = np.random.multivariate_normal(x_old_mean, cov)
                    x = self.denormalize(x)

                    if np.all(self.minimum <= x) and np.all(x <= self.maximum):

                        r, response = self.obj_time(x, data = data, speed = speed, force = force, time = time, method = method, probes = probes, **kwargs)

                        if r < rate[j]:
                            w_aux[i] = self.weights(w, p_prior, next_prior_normalized, x, cov)
                            accrate += 1
                            var[:, i] = x
                            Samples[i] = response
                            f_aux.append(r)
                            i = i+1
                            print(f'Valor de i atualizado = {i}')
                            if r < fbest:
                                fbest = r
                                xbest = x

                    else:
                        pass

                next_prior = copy.deepcopy(var)
                w = w_aux/sum(w_aux)

        self.update(var)
        distributions = copy.deepcopy(self.all_params)
        
        best = xbest

        self.update(best)
        rotor = self.build_rotor()

        results_MAP = rotor.run_time_response(
            speed, 
            force, 
            time,  
            method = method, 
            **kwargs) 

        fr = []

        for i in range(len(probes)):
                y = results_MAP.xout[:,probes[i][0]*6]*np.cos(np.radians(probes[i][1])) + results_MAP.yout[:,probes[i][0]*6]*np.sin(np.radians(probes[i][1]))
                fr.append(y)

        best_response = np.array(fr)
        
        self.samples = Samples
        self.best_response = best_response

        if isinstance(self.optimized, np.ndarray):

            self.update(self.optimized)

            rotor = self.build_rotor()

            results = rotor.run_time_response(
            speed, 
            force, 
            time,  
            method = method, 
            **kwargs) 

        else:
            results = results_MAP

        return ST_ID_TimeResponseResults(rotor, time, results.yout, results.xout, probes = probes, data = data, MAP_yout = results_MAP.yout, MAP_xout = results_MAP.xout, optimum = self.optimized,
                                         curves = Samples, best_curve = best_response, distributions = distributions, Prior = self.prior)


