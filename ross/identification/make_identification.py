import ross as rs
from ross import Rotor
import numpy as np
from copy import deepcopy
import inspect
from collections import defaultdict
from numpy.linalg import norm 
import copy
import gc
from ross.identification.id_results import ID_FrequencyResponseResults
import scipy as sp

class make_identification(Rotor):
    def __init__(self, rotor, parameters, error = 5/100, **kwargs):
        self.rotor = deepcopy(rotor)
        self.params = parameters
        self.error = error
            
        attribute_dict = dict(
            rotor=self.rotor,
            params=self.params,
            error=self.error,
            )
        
        self.params_to_identify, self.minimum, self.maximum = self.mapping_elements(self.rotor, self.params, self.error)

        self.building_rotor()        
    
    def mapping_elements(self, rotor, parameters, error):
        results = defaultdict(dict)
        
        minimum = []
        maximum = []

        if isinstance(parameters, list):
            for i, element in enumerate(rotor.elements):
                for p in parameters:
                    if p in inspect.signature(element.__init__).parameters.keys():
                        value = getattr(element, p)
                        if isinstance(value, list):
                            value = np.array(value)
                        results[element.tag][p] = np.random.uniform(value*(1-error), value*(1+error))
                        minimum.extend(value*(1-error))
                        maximum.extend(value*(1+error))
                    else:
                        pass
        elif isinstance(parameters, dict):
            for i, element in enumerate(rotor.elements):
                for k, v in parameters.items():
                    if k == element.tag:
                        for p in v:
                            if p in inspect.signature(element.__init__).parameters.keys():
                                value = getattr(element, p)
                                if isinstance(value, list):
                                    value = np.array(value)
                                results[element.tag][p] = np.random.uniform(value*(1-error), value*(1+error))
                                minimum.extend(value*(1-error))
                                maximum.extend(value*(1+error))
                        else:
                            pass

        return results, np.array(minimum), np.array(maximum)
    
    def dict_to_array(self):
        values = []
        for elem in self.params_to_identify.values():
            values.extend(elem.values())

        final = np.concatenate(values).ravel()

        print(final)
        return final
    
    def array_to_dict(self, new_values_array):
        it = iter(new_values_array)
        
        for elem_name, params in self.params_to_identify.items():
            for param_name, current_array in params.items():
                
                size = current_array.size
                try:
                    # Atualiza o conteúdo do array existente
                    # Dica: usar [:] ou copyto mantém a referência original se necessário
                    new_data = [next(it) for _ in range(size)]
                    self.params_to_identify[elem_name][param_name] = np.array(new_data)
                except StopIteration:
                    raise ValueError(f"The provided array is too short for the {param_name} of {elem_name}.")

        # Validação final: o array foi todo consumido?
        try:
            next(it)
        except StopIteration:
            pass

    def array_to_dict_distribution(self, new_values_matrix):
        self.params_distributions = defaultdict(dict)
        # 1. LIMPEZA INICIAL
        # Garantimos que as listas estejam vazias para os NOVOS dados
        for elem_name, params in self.params_to_identify.items():
            for param_name in params.keys():
                self.params_distributions[elem_name][param_name] = []

        # 2. POPULAÇÃO COM AS NOVAS AMOSTRAS
        for x in new_values_matrix.T:
            start = 0
            for elem_name, params in self.params_to_identify.items():
                for param_name, current_array in params.items():
                    size = current_array.size
                    end = start + size
                    
                    # Pega a fatia direto do vetor da partícula 'x'
                    new_data = x[start:end]
                    
                    # Se for escalar, pega o valor, se for curva, mantém o array
                    valor = new_data[0] if size == 1 else np.array(new_data)
                    
                    self.params_distributions[elem_name][param_name].append(valor)
                    start = end

                # Validação opcional para garantir que o vetor 'x' foi totalmente consumido
        #return self.params_distributions

    
    def building_rotor(self):

        shaft_elements = []
        disk_elements = []
        bearing_elements = []
        point_mass_elements = []
        
        # 2. Aplicamos as novas propriedades (ex: do seu dicionário de identificação)
        # Importante: certifique-se de que os elementos têm tags únicas para o mapeamento
        for element in self.rotor.elements:
            if element.tag in self.params_to_identify:
                for attr, val in self.params_to_identify[element.tag].items():
                    setattr(element, attr, val)

            sig = inspect.signature(element.__init__)
            element_class = type(element)
            acepted_params = [p for p in sig.parameters if p != 'self']
            constructor = {}
            for p in acepted_params:
                if hasattr(element, p):
                    constructor[p] = getattr(element, p)
                elif sig.parameters[p].default is not inspect.Parameter.empty:
                    continue

            new_element = element_class(**constructor)

            if isinstance(new_element, rs.BearingElement):
                bearing_elements.append(new_element)
            elif isinstance(new_element, rs.ShaftElement):
                shaft_elements.append(new_element)
            elif isinstance(new_element, rs.DiskElement):
                disk_elements.append(new_element)
            elif isinstance(new_element, rs.PointMass):
                point_mass_elements.append(new_element)

        # Montagem final do Rotor com os novos objetos "puros"
        new_rotor = rs.Rotor(
            shaft_elements=shaft_elements,
            bearing_elements=bearing_elements,
            disk_elements=disk_elements,
            point_mass_elements=point_mass_elements,
            min_w=getattr(self.rotor, 'min_w', None),
            max_w=getattr(self.rotor, 'max_w', None)
        )
        
        return new_rotor

    def swarm(self, func, inter, iterations, npop, **kwargs):
        """Particle Swarm Optimization (PSO) for parameter search.

        Parameters
        ----------
        func : callable
            Objective function to be minimized.
        inter : np.array
            Array containing [min_bounds, max_bounds] for parameters.
        iterations : int
            Number of iterations for the optimization.
        npop : int
            Population size (number of particles).

        Returns
        -------
        pbest : np.array
            The best set of parameters found.
        fbest : float
            The minimum objective function value found.
        """
        f_history = []
        x_obj = []
        obj_evaluations = 0

        it_count = 0
        nvar = len(inter[0])
        beta = 2.0
        x = np.empty([npop, nvar])
        xmin = inter[0]
        xmax = inter[1]

        for i in range(nvar):
            x[:, i] = np.random.uniform(low=xmin[i], high=xmax[i], size=npop)

        f = np.array([func(xi, **kwargs)[0] for xi in x])
        obj_evaluations += npop
        popbest = np.copy(x)
        fpopbest = np.copy(f)
        ibest = np.argmin(f)
        pbest = np.copy(x[ibest])
        fbest = f[ibest]
        run = True

        while run:
            r1 = np.random.random(size=(npop, nvar))
            r2 = np.random.random(size=(npop, nvar))
            x += beta * r1 * (popbest - x) + beta * r2 * (pbest - x)
            x = np.where(x > 0, x, x * -1)
            f = np.array([func(xi, **kwargs)[0] for xi in x])
            obj_evaluations += npop
            idx = f < fpopbest
            fpopbest[idx] = f[idx]
            popbest[idx, :] = x[idx, :]
            ibest = np.argmin(f)
            if f[ibest] < fbest:
                fbest = f[ibest]
                pbest = np.copy(x[ibest])
                f_history.append(fbest)
                x_obj.append(obj_evaluations)

            it_count += 1
            if it_count >= iterations:
                run = False
            
            print("{:3d} {:15.8f} {:15.8f}".format(it_count, pbest[0], fbest))

        print(f'F_best = {fbest} \n X_best = {pbest} \n')
        return pbest, fbest 
    

    def run_idFreq(
            self,  
            data, 
            probes,
            speed_range,
            it=50,
            npop=20,
            modes=None,
            cluster_points=False,
            num_modes=12,
            num_points=10,
            rtol=0.005):
        """Run parameter identification in frequency domain using PSO.

        Returns
        -------
        results : ST_ID_FrequencyResponseResults
            Object containing identification results.
        """
        parameters, f_best = self.swarm(self.obj_freq, np.array([self.minimum, self.maximum]), it, npop, data=data, probes=probes, speed_range=speed_range, modes=modes, cluster_points=cluster_points, num_modes=num_modes, num_points=num_points, rtol=rtol)

        self.array_to_dict(parameters)
        rotor = self.building_rotor()

        results = rotor.run_freq_response(
                speed_range,
                modes,
                cluster_points,
                num_modes,
                num_points,
                rtol,
            )
        
        self.optimized = parameters
        #return parameters
        return ID_FrequencyResponseResults(freq_resp=results.freq_resp, velc_resp=results.velc_resp, accl_resp=results.accl_resp, speed_range=speed_range, number_dof=rotor.number_dof, data=data, probes=probes, optimum=self.optimized)

    def obj_freq(self, x, **kwargs):
        """Objective function for frequency identification.
        
        Calculates the error between experimental data and numerical model in dB.
        """
        data = kwargs['data']
        probes = kwargs['probes']
        speed_range = kwargs['speed_range']
        modes = kwargs['modes']
        cluster_points = kwargs['cluster_points']
        num_modes = kwargs['num_modes']
        num_points = kwargs['num_points']
        rtol = kwargs['rtol']

        self.array_to_dict(x)
        rotor = self.building_rotor()

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
        aux_response = copy.deepcopy(resps)

        if not isinstance(data, np.ndarray):
            data = np.array(data)

        fr_db = 20*np.log10(np.abs(resps))
        data_db = 20*np.log10(np.abs(data))

        del rotor
        gc.collect()

        error_val = norm(abs(data_db - fr_db)**2)**2 / norm(data_db)**2
        return error_val, aux_response   

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
        """Parameter identification using SMC-ABC in the Frequency Domain.

        This method implements the Sequential Monte Carlo Approximate Bayesian 
        Computation (SMC-ABC) algorithm. It estimates the posterior distribution 
        of rotor parameters by iteratively filtering samples through decreasing 
        tolerance thresholds (rates).

        Parameters
        ----------
        data : array_like
            Experimental or target frequency response data.
        probes : list
            List of probe configurations [node, angle].
        speed_range : array
            Array of frequencies/speeds (rad/s) for the response.
        rate : list
            List of tolerance thresholds (epsilon) for each SMC generation.
        interval : list
            List containing [reduction_factor, amplification_factor] to define 
            parameter bounds if optimized=True.
        nsample : int, optional
            Number of samples to be accepted in each generation. Default is 100.
        optimized : bool, optional
            If True, centers and scales the search bounds based on current parameters.
            Default is True.
        modes : list, optional
            Modes to be used in the frequency response calculation.
        cluster_points : bool, optional
            Whether to use frequency clustering. Default is False.
        num_modes : int, optional
            Number of modes to calculate. Default is 12.
        num_points : int, optional
            Number of points for clustering. Default is 10.
        rtol : float, optional
            Relative tolerance for frequency response. Default is 0.005.

        Returns
        -------
        results : ST_ID_FrequencyResponseResults
            An object containing the accepted samples, MAP response, and 
            identified distributions.
        """
        n_var = len(self.minimum)
        var_samples = np.empty([n_var, 1])
        samples_freqs_history = np.empty([1, len(speed_range)])
        acc_rate = 0
        fbest = 1
        xbest = []
        num_rates = len(rate)
        w = np.ones(nsample) / nsample

        if optimized == True:
            minimum_bounds = np.array([])
            maximum_bounds = np.array([])
            reduction = interval[0]
            amplification = interval[1]
            for params in self.params_to_identify.values():
                for value in params.values():
                    minimum_bounds = np.append(minimum_bounds, reduction * value)
                    maximum_bounds = np.append(maximum_bounds, amplification * value)
            self.minimum = minimum_bounds
            self.maximum = maximum_bounds
        else:
            pass

        p_prior = np.prod(1 / (self.maximum - self.minimum))

        print(f'Minimum parameter values: {self.minimum} \n')
        print(f'Maximum parameter values: {self.maximum} \n')
        print(f'Starting iteration loop \n')

        prior_samples = np.empty([n_var, nsample])

        for i in range(n_var):
            prior_samples[i, :] = np.random.uniform(low=self.minimum[i], high=self.maximum[i], size=nsample)

        self.array_to_dict_distribution(prior_samples)
        self.prior = copy.deepcopy(self.params_distributions)
        print(f'prior:{self.prior}')

        for j in range(num_rates):
            print(f'Gen: {j}')
            var_samples = np.empty([n_var, nsample])
            samples_freqs = np.empty([nsample, len(data), len(speed_range)], dtype=complex)
            w_aux = np.empty(len(w))
            f_aux = []
            i = 0
            print(f'Value of i = {i}')

            if j == 0:
                while i < nsample:
                    x = np.empty([n_var])
                    for z in range(n_var):
                        x[z] = np.random.uniform(self.minimum[z], self.maximum[z])
                    
                    r, freqs = self.obj_freq(x, data=data, probes=probes, speed_range=speed_range, modes=modes, cluster_points=cluster_points, num_modes=num_modes, num_points=num_points, rtol=rtol)

                    print(f'Value of r = {r}')

                    if r < rate[j]:
                        acc_rate += 1
                        var_samples[:, i] = x
                        samples_freqs[i] = freqs
                        f_aux.append(r)
                        print(f'Sample accepted. Value = {r}')
                        i = i + 1
                        print(f'Updated value of i = {i}')
                        if r < fbest:
                            fbest = r
                            xbest = x

                next_prior = copy.deepcopy(var_samples)

            else:
                while i < nsample:
                    x_old = next_prior[:, np.random.choice(nsample, p=w)]
                    next_prior_normalized = self.normalize_prior(next_prior.T).T
                    cov = np.cov(next_prior_normalized, aweights=w)
                    x_old_mean = self.normalize(x_old)
                    x = np.random.multivariate_normal(x_old_mean, cov)
                    x = self.denormalize(x)

                    if np.all(self.minimum <= x) and np.all(x <= self.maximum):

                        r, freqs = self.obj_freq(x, data=data, probes=probes, speed_range=speed_range, modes=modes, cluster_points=cluster_points, num_modes=num_modes, num_points=num_points, rtol=rtol)

                        if r < rate[j]:
                            w_aux[i] = self.calculate_weights(w, p_prior, next_prior_normalized, x, cov)
                            acc_rate += 1
                            var_samples[:, i] = x
                            samples_freqs[i] = freqs
                            f_aux.append(r)
                            i = i + 1
                            print(f'Updated value of i = {i}')
                            if r < fbest:
                                fbest = r
                                xbest = x
                    else:
                        pass

                next_prior = copy.deepcopy(var_samples)
                w = w_aux / sum(w_aux)

        self.array_to_dict_distribution(var_samples)
        distributions = copy.deepcopy(self.params_distributions)
        
        best = xbest

        self.array_to_dict(best)
        rotor = self.building_rotor()

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
        
        self.samples = samples_freqs
        self.best_response = best_response

        if isinstance(self.optimized, np.ndarray):
            self.array_to_dict(self.optimized)
            rotor = self.building_rotor()
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

        print(f'prior:{self.prior}')

        return ID_FrequencyResponseResults(results.freq_resp, results.velc_resp, results.accl_resp, speed_range, rotor.number_dof, data = data, probes = probes, optimum = self.optimized, 
                                              curves = samples_freqs, best_curve = best_response, distributions = distributions,
                                              MAP_freq_resp = results_MAP.freq_resp,
                                              MAP_velc_resp = results_MAP.velc_resp,
                                              MAP_accl_resp = results_MAP.accl_resp,
                                              Prior = self.prior)
    
    def normalize_prior(self, x):
        """Normalize prior samples."""
        return np.array([(v - self.minimum) / (self.maximum - self.minimum) for v in x])
    
    def normalize(self, x):
        """Normalize parameter vector."""
        return (x - self.minimum) / (self.maximum - self.minimum)
    
    def denormalize(self, x_norm):
        """Denormalize parameter vector."""
        return x_norm * (self.maximum - self.minimum) + self.minimum
    
    def calculate_weights(self, w, p_prior, next_prior_norm, x, cov):
        """Calculate SMC-ABC particle weights."""
        x_norm = self.normalize(x)
        vector = np.array([ws * sp.stats.multivariate_normal.pdf(x_norm, mean=next_prior_norm[:, i], cov=cov, allow_singular=True) for i, ws in enumerate(w)])
        return p_prior / sum(vector) 

    
        