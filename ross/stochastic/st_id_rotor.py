import numpy as np
from ross.units import check_units
import ross as rs
import matplotlib.pyplot as plt
from numpy.linalg import norm 
import copy
import scipy as sp
from paretoset import paretoset
import pandas as pd
from scipy.signal import find_peaks

__all__ = ["ST_ID_Rotor"]


class ST_ID_Rotor:

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
        self.Shaft = []
        aux_list = []
        self.materials = {}
        interval_materials = {}

        self.minimum = np.array([])
        self.maximum = np.array([])

        self.freq = np.empty(100)

        if len(list_rotor)==4:
            self.points = list_rotor[3]
        else:
            self.points = []

        self.all_params = {}
        self.elements_to_identify = 0

        #order elements
        order_b = []
        order_d = []
        order_pm = []

        for b in self.bearings:
            order_b.append(b['n'])
        for d in self.disks:
            order_d.append(d['n'])
        for pm in self.points:
            order_pm.append(pm['n'])

        order_b.sort()
        order_d.sort()
        order_pm.sort()

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
                self.shaft[n]['tag'] = 'Shaft ' + str(n)
            if self.shaft[n].to_identify != None:
                if len(self.shaft[n].to_identify) == 1 and 'material_id' in (self.shaft[n].to_identify):
                    pass
                else:
                    self.all_params[self.shaft[n]['tag']] = self.shaft[n].param_values
                if "material_id" in self.shaft[n].to_identify:
                    aux_list.append(self.shaft[n])
                for param, value in self.shaft[n].interval_params.items():
                    self.minimum = np.append(self.minimum, value[0])
                    self.maximum = np.append(self.maximum, value[1])
            self.Shaft.append(self.shaft[n])


        for e in aux_list:
            self.materials[e.attribute_dict['material'].name] = e.material.param_values
            interval_materials[e.attribute_dict['material'].name] = e.material.interval_params

        for mat in interval_materials.values():
            for v in mat.values():
                self.minimum = np.append(self.minimum, v[0])
                self.maximum = np.append(self.maximum, v[1])

        self.elem_w_material = aux_list

        self.all_params.update(self.materials)
    
    def number_dof(self):

        Rotor_built = self.build_rotor()
        return Rotor_built.number_dof

    def ndof(self):
        
        Rotor_built = self.build_rotor()
        return int(Rotor_built.ndof)

    def nodes_pos(self):
        
        Rotor_built = self.build_rotor()
        return Rotor_built.nodes_pos
    
    def nodes(self):
        
        Rotor_built = self.build_rotor()
        return Rotor_built.nodes
    
    def plot_rotor(self):
        Rotor_built = self.build_rotor()
        return Rotor_built.plot_rotor()
            
    def swarm(self,funcao, inter, it, npop, *args, **kwargs):

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

        print(f'Tempo de inicialização = {self.fim-self.inicio} \n')
        f = np.array([funcao(xi, args, **kwargs)[0] for xi in x])
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
            f = np.array([funcao(xi, args, **kwargs)[0] for xi in x])
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
            list_shaft.append(s.new_shaft())

        Rotor_built = rs.Rotor(list_shaft, list_disks, list_bearings,list_points)
        return Rotor_built
    
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
            inp,
            out,
            speed_range,
            it = 50,
            npop = 20,
            modes=None,
            cluster_points=False,
            num_modes=12,
            num_points=10,
            rtol=0.005, ):

        parameters, f_best = self.swarm(self.obj_freq, np.array([self.minimum,self.maximum]), it, npop, data,inp,out,speed_range,modes,cluster_points,num_modes,num_points,rtol)

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
                
        fr = results.freq_resp[inp, out]

        return copy.deepcopy(self.all_params), f_best, copy.deepcopy(fr)
    
    def obj_freq(self,
                 x,
                 *args):
        
        data,inp,out,speed_range,modes,cluster_points,num_modes,num_points,rtol = args[0]

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
            
        
        fr = results.freq_resp[inp, out]
        d2 = np.sqrt(data.real**2+data.imag**2)

        r = norm(d2-fr)**2/norm(d2)**2
        
        return r, fr
    
    def obj_time(self,
                 x,
                 *args,
                 **kwargs
                 ):

        data, speed, force, tempo, method, node = args[0]

        self.update(x)

        rotor = self.build_rotor()

        run_time = rotor.run_time_response(speed, force, tempo,  method = method, **kwargs)

        y =  run_time.yout[:, node]

        data_fft = np.fft.rfft(data)
        fft = np.fft.rfft(y)

        result = 0.35*(norm(data-y)**2/norm(data)**2) + 0.65*(norm(data_fft-fft)**2/norm(data_fft)**2)

        return result, y
    
    def run_idTime(
            self, 
            data,
            speed,
            force,
            tempo,
            node,
            method = 'default',
            it = 50,
            npop = 20,
            **kwargs
            ):
        
        self.data_time = data
        self.Force = force
        self.time = tempo
        self.speed = speed
        self.node = node
                
        parameters, f_best = self.swarm(self.obj_time, np.array([self.minimum,self.maximum]), it, npop, data, speed, force, tempo, method, node, **kwargs)

        self.update(parameters)
        rotor = self.build_rotor()

        run_time = rotor.run_time_response(speed, force, tempo, method = 'Newmark')
        y =  run_time.yout[:, node]

        return copy.deepcopy(self.all_params), f_best, copy.deepcopy(y)

    def run_ABCFreq(
            self,  
            data, 
            inp,
            out,
            speed_range,
            rate,
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

        args = [data,inp,out,speed_range,modes,cluster_points,num_modes,num_points,rtol]

        if optimized == True:
            minimum = np.array([])
            maximum = np.array([])
            reduction = 0.9
            amplifie = 1.1
            for params in self.all_params.values():
                for value in params.values():
                    minimum = np.append(minimum, reduction*value)
                    maximum = np.append(maximum, amplifie*value)
            self.minimum = minimum
            self.maximum = maximum
        else:
            pass

        for i in range(nsample):
            x = np.empty(len(self.maximum))
            for j in range(len(x)):
                x[j] = np.random.uniform(self.minimum[j], self.maximum[j])

            r, freqs = self.obj_freq(x, args)
            print(r)

            if r < rate:
                accrate += 1
                var = np.append(var, np.array([x]).T, axis = 1)
                Samples = np.append(Samples, np.array([freqs]), axis = 0)

        var = np.delete(var, 0, axis = 1)
        Samples = np.delete(Samples, 0, axis = 0)

        means = np.mean(var, axis = 1)

        self.update(means)
        dmeans = copy.deepcopy(self.all_params)
        rotor = self.build_rotor()

        results = rotor.run_freq_response(
                speed_range,
                modes,
                cluster_points,
                num_modes,
                num_points,
                rtol,
            )

        fr = results.freq_resp[inp, out]


        std = np.std(var, axis = 1)

        ms = np.array(list(zip(means, std)))
        
        self.update(ms)
        dms = copy.deepcopy(self.all_params)
        self.update(var)
        dists = copy.deepcopy(self.all_params)
        self.update(means)

        print(f'Taxa de aceitação: {accrate/nsample*100}')

        return dmeans, dists, Samples, dms, copy.deepcopy(fr)
    
    def run_ABCTime(
            self,  
            data,
            speed,
            force,
            tempo,
            node,
            rate,
            nsample,
            method = 'default',
            optimized = True,
            **kwargs
    ):

        n_var = len(self.minimum)
        var = np.empty([n_var,1])
        Samples = np.empty([1,len(tempo)])
        accrate = 0

        args = [data, speed, force, tempo, method, node]

        if optimized == True:
            minimum = np.array([])
            maximum = np.array([])
            reduction = 0.9
            amplifie = 1.1
            for params in self.all_params.values():
                for value in params.values():
                    minimum = np.append(minimum, reduction*value)
                    maximum = np.append(maximum, amplifie*value)
            self.minimum = minimum
            self.maximum = maximum
        else:
            pass

        for i in range(nsample):
            x = np.empty(len(self.maximum))
            for j in range(len(x)):
                x[j] = np.random.uniform(self.minimum[j], self.maximum[j])

            r, freqs = self.obj_time(x, args, **kwargs)

            if r < rate:
                accrate += 1
                var = np.append(var, np.array([x]).T, axis = 1)
                Samples = np.append(Samples, np.array([freqs]), axis = 0)

        var = np.delete(var, 0, axis = 1)
        Samples = np.delete(Samples, 0, axis = 0)

        means = np.mean(var, axis = 1)

        self.update(means)
        dmeans = copy.deepcopy(self.all_params)
        rotor = self.build_rotor()

        run_time = rotor.run_time_response(speed, force, tempo, method = 'Newmark')
        y =  run_time.yout[:, node]

        std = np.std(var, axis = 1)

        ms = np.array(list(zip(means, std)))
        
        self.update(ms)
        dms = copy.deepcopy(self.all_params)
        self.update(var)
        dists = copy.deepcopy(self.all_params)
        self.update(means)

        print(f'Taxa de aceitação: {accrate/nsample*100}')

        return dmeans, dists, Samples, dms, copy.deepcopy(y)
    
    def run_pareto(self,fun1, fun2, conv = False, c1 = 1, it = 5, npop = 100, sense = ['min', 'min'], *args):

        inter = np.array([self.minimum, self.maximum])

        F =[]
        Xobj = []
        obj = 0

        nobj = 0 #contador
        nvar = len(inter[0])
        beta = 2.0
        x = np.empty([npop,nvar])
        xmin = inter[0]
        xmax = inter[1]

        f1 = np.empty([0])
        f2 = np.empty([0])

        f1v = np.empty(0)
        f2v = np.empty(0)

        for i in range(nvar):
                x[:,i] = np.random.uniform(low = xmin[i], high = xmax[i], size = npop)

        var = np.copy(x)

        for xi in  x:
                self.update(xi)
                rotor = self.build_rotor()
                f1r = fun1(rotor, args)
                f2r = fun2(rotor, args)

                f1 = np.append(f1,f1r)
                f2 = np.append(f2,f2r)

        f1v = np.concatenate((f1v,f1))
        f2v = np.concatenate((f2v,f2))

        f = np.concatenate((f1,f2))

        xb = np.concatenate((x,x))

        obj = obj + npop
        popbest = np.copy(xb)
        fpopbest = np.copy(f)
        ibest = np.argmin(f)
        pbest = np.copy(xb[ibest])
        fbest = f[ibest]
        run = True

        while run == True:

            xold = np.copy(xb)
            r1 = np.random.random(size=(2*npop, nvar))
            r2 = np.random.random(size=(2*npop, nvar))
            xb += beta * r1 * (popbest - xb) + beta * r2 * (pbest - xb)
            xb = np.where(xb>0, xb, xb*-1)

            f1 = np.empty([0])
            f2 = np.empty([0])

            for xi in  xb[0:int(len(xb)/2)]:
                self.update(xi)
                rotor = self.build_rotor()
                f1r = fun1(rotor, args)
                f2r = fun2(rotor, args)

                f1 = np.append(f1,f1r)
                f2 = np.append(f2,f2r)

            f1v = np.concatenate((f1v,f1))
            f2v = np.concatenate((f2v,f2))

            f = np.concatenate((f1,f2))

            obj = obj+npop
            idx = f < fpopbest
            fpopbest[idx] = f[idx]
            popbest[idx, :] = xb[idx, :]
            ibest = np.argmin(f)
            if f[ibest] < fbest:
                fbest = f[ibest]
                pbest = np.copy(xb[ibest])
                F.append(fbest)
                Xobj.append(obj)

            nobj+=1

            if fbest <= c1:
                break
            if nobj >=it:
                run = False

        if conv == True:
            plt.plot(Xobj, F)
            plt.show()

        print(len(f1v))
        print(len(f2v))

        functions = pd.DataFrame({'funcao1': f1v, 'funcao2': f2v})
        mask = paretoset(functions, sense = sense)
        paretoset_functions = functions[mask]

        plt.scatter(f1v, f2v)
        plt.scatter(paretoset_functions['funcao1'],paretoset_functions['funcao2'])
        plt.show()

        return f1v, f2v, xb[0:int(len(xb)/2)]

