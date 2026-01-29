"""STOCHASTIC ROSS plotting module.

This module returns graphs for each type of analyses in st_rotor_assembly.py.
"""

import copy
import inspect
from abc import ABC
from collections.abc import Iterable
from pathlib import Path
from warnings import warn

import numpy as np
import toml
from plotly import express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from ross.plotly_theme import tableau_colors
from ross.units import Q_
from ross.probe import Probe

# set Plotly palette of colors
colors1 = px.colors.qualitative.Dark24
colors2 = px.colors.qualitative.Light24

__all__ = []


class ST_ID_Results(ABC):
    """Results class.

    This class is a general abstract class to be implemented in other classes
    for post-processing results, in order to add saving and loading data options.
    """

    def save(self, file):
        """Save results in a .toml file.

        This function will save the simulation results to a .toml file.
        The file will have all the argument's names and values that are needed to
        reinstantiate the class.

        Parameters
        ----------
        file : str, pathlib.Path
            The name of the file the results will be saved in.

        Examples
        --------
        >>> # Example running a stochastic unbalance response
        >>> from tempfile import tempdir
        >>> from pathlib import Path
        >>> import ross.stochastic as srs

        >>> # Running an example
        >>> rotors = srs.st_rotor_example()
        >>> freq_range = np.linspace(0, 500, 31)
        >>> n = 3
        >>> m = np.random.uniform(0.001, 0.002, 10)
        >>> p = 0.0
        >>> results = rotors.run_unbalance_response(n, m, p, freq_range)

        >>> # create path for a temporary file
        >>> file = Path(tempdir) / 'results.toml'
        >>> results.save(file)
        """
        # get __init__ arguments
        signature = inspect.signature(self.__init__)
        args_list = list(signature.parameters)
        args = {arg: getattr(self, arg) for arg in args_list}
        try:
            data = toml.load(file)
        except FileNotFoundError:
            data = {}

        data[f"{self.__class__.__name__}"] = args
        with open(file, "w") as f:
            toml.dump(data, f, encoder=toml.TomlNumpyEncoder())

    @classmethod
    def read_toml_data(cls, data):
        """Read and parse data stored in a .toml file.

        The data passed to this method needs to be according to the
        format saved in the .toml file by the .save() method.

        Parameters
        ----------
        data : dict
            Dictionary obtained from toml.load().

        Returns
        -------
        The result object.
        """
        return cls(**data)

    @classmethod
    def load(cls, file):
        """Load results from a .toml file.

        This function will load the simulation results from a .toml file.
        The file must have all the argument's names and values that are needed to
        reinstantiate the class.

        Parameters
        ----------
        file : str, pathlib.Path
            The name of the file the results will be loaded from.

        Examples
        --------
        >>> # Example running a stochastic unbalance response
        >>> from tempfile import tempdir
        >>> from pathlib import Path
        >>> import ross.stochastic as srs

        >>> # Running an example
        >>> rotors = srs.st_rotor_example()
        >>> freq_range = np.linspace(0, 500, 31)
        >>> n = 3
        >>> m = np.random.uniform(0.001, 0.002, 10)
        >>> p = 0.0
        >>> results = rotors.run_unbalance_response(n, m, p, freq_range)

        >>> # create path for a temporary file
        >>> file = Path(tempdir) / 'results.toml'
        >>> results.save(file)

        >>> # Loading file
        >>> results2 = srs.ST_ForcedResponseResults.load(file)
        >>> results2.forced_resp.all() == results.forced_resp.all()
        True
        """
        data = toml.load(file)
        # extract single dictionary in the data
        data = list(data.values())[0]
        for key, value in data.items():
            if isinstance(value, Iterable):
                data[key] = np.array(value)
                if data[key].dtype == np.dtype("<U49"):
                    data[key] = np.array(value).astype(np.complex128)
        return cls.read_toml_data(data)


class ST_ID_FrequencyResponseResults(ST_ID_Results):
    """Class used to store results and provide plots for Frequency Response.

    Parameters
    ----------
    freq_resp : array
        Array with the frequency response (displacement).
    velc_resp : array
        Array with the frequency response (velocity).
    accl_resp : array
        Array with the frequency response (acceleration).
    speed_range : array
        Array with the speed range in rad/s.
    number_dof : int
        Number of degrees of freedom per node.

    Returns
    -------
    subplots : Plotly graph_objects.make_subplots()
        Plotly figure with Amplitude vs Frequency
    """

    def __init__(self, freq_resp, velc_resp, accl_resp, speed_range, number_dof, **kwargs):
        self.freq_resp = freq_resp
        self.velc_resp = velc_resp
        self.accl_resp = accl_resp
        self.speed_range = speed_range
        self.number_dof = number_dof
        self.MAP_freq_resp = kwargs.get('MAP_freq_resp', None)
        self.MAP_velc_resp = kwargs.get('MAP_velc_resp', None)
        self.MAP_accl_resp = kwargs.get('MAP_accl_resp', None)
        self.data = kwargs['data']
        self.probes = kwargs['probes']
        self.optimum = kwargs.get('optimum', None)
        self.curves = kwargs.get('curves', None)
        self.best_curve_smc = kwargs.get('best_curve', None)
        self.distributions = kwargs.get('distributions', None)
        self.Prior = kwargs.get('Prior', None)

        self.dof_dict = {"0": "x", "1": "y", "2": "z", "3": "α", "4": "β", "5": "θ"}

    def plot_magnitude(
        self,
        data,
        conf_interval = [],
        frequency_units="rad/s",
        amplitude_units="m/N",
        fig=None,
        type = None,
        **mag_kwargs,
    ):
        """Plot frequency response (magnitude) using Plotly.

        This method plots the frequency response magnitude given an output and
        an input using Plotly.
        It is possible to plot the magnitude with different units, depending on the unit entered in 'amplitude_units'. If '[length]/[force]', it displays the displacement unit (m); If '[speed]/[force]', it displays the velocity unit (m/s); If '[acceleration]/[force]', it displays the acceleration  unit (m/s**2).

        Parameters
        ----------
        inp : int
            Input.
        out : int
            Output.
        frequency_units : str, optional
            Units for the x axis.
            Default is "rad/s"
        amplitude_units : str, optional
            Units for the response magnitude.
            Acceptable units dimensionality are:

            '[length]' - Displays the magnitude with units (m/N);

            '[speed]' - Displays the magnitude with units (m/s/N);

            '[acceleration]' - Displays the magnitude with units (m/s**2/N).

            Default is "m/N" 0 to peak.
            To use peak to peak use '<unit> pkpk' (e.g. 'm/N pkpk')
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        mag_kwargs : optional
            Additional key word arguments can be passed to change the plot layout only
            (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        """

        i = data

        fig = go.Figure()
        idx = len(fig.data)

        inp = self.probes[i][0]
        out = self.probes[i][1]

        inpn = inp // self.number_dof
        idof = self.dof_dict[str(inp % self.number_dof)]
        outn = out // self.number_dof
        odof = self.dof_dict[str(out % self.number_dof)]

        frequency_range = Q_(self.speed_range, "rad/s").to(frequency_units).m

        dummy_var = Q_(1, amplitude_units)
        y_label = "Magnitude"
        if dummy_var.check("[length]/[force]"):
                    mag = np.abs(self.freq_resp)
                    mag = Q_(mag, "m/N").to(amplitude_units).m
        elif dummy_var.check("[speed]/[force]"):
                    mag = np.abs(self.velc_resp)
                    mag = Q_(mag, "m/s/N").to(amplitude_units).m
        elif dummy_var.check("[acceleration]/[force]"):
                    mag = np.abs(self.accl_resp)
                    mag = Q_(mag, "m/s**2/N").to(amplitude_units).m
        else:
                    raise ValueError(
                        "Not supported unit. Options are '[length]/[force]', '[speed]/[force]', '[acceleration]/[force]'"
                    )
                
        y = mag[self.probes[i][0]*6, self.probes[i][1]*6, :]*np.cos(np.radians(self.probes[i][2])) + mag[self.probes[i][0]*6+1, self.probes[i][1]*6+1, :]*np.sin(np.radians(self.probes[i][2]))

        fig.add_trace(
                    go.Scatter(
                        x=frequency_range,
                        y=np.abs(self.data[i]),
                        mode="lines",
                        line=dict(color=list(tableau_colors)[idx]),
                        name=f"Data {i} | inp: node {inp} <br>out: node {out}",
                        legendgroup=f"Data {i}",
                        showlegend=True,
                        hovertemplate=f"Frequency ({frequency_units}): %{{x:.2f}}<br>Amplitude ({amplitude_units}): %{{y:.2e}}",
                    )
                )
        
        idx = len(fig.data)

        if isinstance(self.optimum, np.ndarray):
            
            fig.add_trace(
                        go.Scatter(
                            x=frequency_range,
                            y=y,
                            mode="lines",
                            line=dict(color=list(tableau_colors)[idx]),
                            name=f"Identified",
                            legendgroup=f"inp: node {inpn} | dof: {idof}<br>out: node {outn} | dof: {odof}",
                            showlegend=True,
                            hovertemplate=f"Frequency ({frequency_units}): %{{x:.2f}}<br>Amplitude ({amplitude_units}): %{{y:.2e}}",
                        ),
                    )
        else:
              pass
        
        if isinstance(self.curves, np.ndarray):

            dummy_var = Q_(1, amplitude_units)
            y_label = "Magnitude"
            if dummy_var.check("[length]/[force]"):
                        mag_MAP = np.abs(self.MAP_freq_resp)
                        mag_MAP = Q_(mag_MAP, "m/N").to(amplitude_units).m
            elif dummy_var.check("[speed]/[force]"):
                        mag_MAP = np.abs(self.MAP_velc_resp)
                        mag_MAP = Q_(mag_MAP, "m/s/N").to(amplitude_units).m
            elif dummy_var.check("[acceleration]/[force]"):
                        mag_MAP = np.abs(self.MAP_accl_resp)
                        mag_MAP = Q_(mag_MAP, "m/s**2/N").to(amplitude_units).m
            else:
                        raise ValueError(
                            "Not supported unit. Options are '[length]/[force]', '[speed]/[force]', '[acceleration]/[force]'"
                        )
            
            y_MAP = mag_MAP[self.probes[i][0]*6, self.probes[i][1]*6, :]*np.cos(np.radians(self.probes[i][2])) + mag_MAP[self.probes[i][0]*6+1, self.probes[i][1]*6+1, :]*np.sin(np.radians(self.probes[i][2]))

            idx = len(fig.data)

            fig.add_trace(
                        go.Scatter(
                            x=frequency_range,
                            y=y_MAP,
                            mode="lines",
                            line=dict(color=list(tableau_colors)[idx]),
                            name=f"MAP",
                            opacity = 0.5,
                            legendgroup=f"inp: node {inpn} | dof: {idof}<br>out: node {outn} | dof: {odof}",
                            showlegend=True,
                            hovertemplate=f"Frequency ({frequency_units}): %{{x:.2f}}<br>Amplitude ({amplitude_units}): %{{y:.2e}}",
                        ),
                    )
              

            x = np.concatenate((frequency_range, frequency_range[::-1]))
            for i, p in enumerate(conf_interval):
                p1 = np.percentile(np.abs(self.curves[:,data,:]), 50 + p / 2, axis=0)
                p2 = np.percentile(np.abs(self.curves[:,data,:]), 50 - p / 2, axis=0)
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=np.concatenate((p1, p2[::-1])),
                        mode="lines",
                        line=dict(width=1, color=colors1[i]),
                        fill="toself",
                        fillcolor=colors1[i],
                        opacity=0.5,
                        name="confidence interval: {}%".format(p),
                        legendgroup="conf{}".format(i),
                        hovertemplate=("Frequency: %{x:.2f}<br>" + "Amplitude: %{y:.2e}"),
                    )
                )
        

        fig.update_xaxes(
                    title_text=f"Frequency ({frequency_units})",
                    range=[np.min(frequency_range), np.max(frequency_range)],
                )
        fig.update_yaxes(title_text=f"{y_label} ({amplitude_units})")
        fig.update_layout(**mag_kwargs)

        if type != None:
              fig.update_yaxes(type = type)

        return fig
    
    def plot_histogram_zero(self, elements = [], parameters = [], prior = True):

        for values in self.distributions.values():
            for values_aux in values.values():
                n_samples = len(values_aux)

        rows = len(parameters)
        cols = len(elements)

        fig = make_subplots(rows=rows, cols=cols)

        for i, element in enumerate(elements):
              for j, parameter in enumerate(parameters):
                    if prior == True:
                        categories = ['Posterior']*n_samples + ['Prior']*n_samples
                        x_data = np.concatenate((self.distributions[element][parameter], self.Prior[element][parameter]))
                    else:
                        categories = ['Posterior']*n_samples
                        x_data = self.distributions[element][parameter]
                    
                    unique_categories = sorted(list(set(categories)))

                    for category in unique_categories:
                        
                        category_data = x_data[np.array(categories) == category]

                        fig.add_trace(go.Histogram(
                            x=category_data,
                            name=category,  
                            opacity=0.7, 
                        ),
                        row = j+1,
                        col = i+1)
        
        return fig
    
    def plot_histogram(self, elements=[], parameters=[], prior=True, 
                       histogram_kwargs_posterior=None, histogram_kwargs_prior=None):
    
        '''for values in self.distributions.values():
            for values_aux in values.values():
                n_samples = len(values_aux)
                break
            break'''

        rows = len(parameters)
        cols = len(elements)

        fig = make_subplots(rows=rows, cols=cols,
                            subplot_titles=[f'<b>{element}</b>' for element in elements for _ in range(rows)]
                           )

        # Definir configurações padrão para o histograma POSTERIOR
        default_posterior_kwargs = {
            'opacity': 0.7,
            'histnorm': 'probability',
            'marker_color': '#1f77b4',       # Azul padrão do Plotly
            'marker_line_width': 1,
            'marker_line_color': 'black',
        }
        # Atualiza as configurações padrão do Posterior com as fornecidas pelo usuário
        if histogram_kwargs_posterior is None:
            histogram_kwargs_posterior = {}
        final_posterior_kwargs = {**default_posterior_kwargs, **histogram_kwargs_posterior}

        # Definir configurações padrão para o histograma PRIOR
        default_prior_kwargs = {
            'opacity': 0.7,
            'histnorm': 'probability',
            'marker_color': '#ff7f0e',       # Laranja padrão do Plotly
            'marker_line_width': 1,
            'marker_line_color': 'black',
        }
        # Atualiza as configurações padrão do Prior com as fornecidas pelo usuário
        if histogram_kwargs_prior is None:
            histogram_kwargs_prior = {}
        final_prior_kwargs = {**default_prior_kwargs, **histogram_kwargs_prior}

        # Controlar quais traces mostram a legenda (apenas uma vez para cada)
        show_legend_for_traces = {
            'Posterior': True,
            'Prior': True
        }

        for i, element in enumerate(elements):
            for j, parameter in enumerate(parameters):
                # Dados para o Posterior
                posterior_data = self.distributions[element][parameter]
                
                # Adiciona o trace POSTERIOR
                fig.add_trace(go.Histogram(
                    x=posterior_data,
                    name='Posterior',
                    showlegend=show_legend_for_traces['Posterior'] and (i == 0 and j == 0),
                    **final_posterior_kwargs # Desempacota as configurações do Posterior
                ),
                row=j + 1,
                col=i + 1)

                if prior:
                    # Dados para o Prior
                    prior_data = self.Prior[element][parameter]
                    
                    # Adiciona o trace PRIOR
                    fig.add_trace(go.Histogram(
                        x=prior_data,
                        name='Prior',
                        showlegend=show_legend_for_traces['Prior'] and (i == 0 and j == 0),
                        **final_prior_kwargs # Desempacota as configurações do Prior
                    ),
                    row=j + 1,
                    col=i + 1)
                
                # Atualizar o título do eixo X
                fig.update_xaxes(title_text=f'<b>{parameter}</b>', row=j+1, col=i+1)
                
                # O título do eixo Y agora pode ser mais genérico ou depender do 'histnorm'
                # da primeira categoria plotada ou de uma preferência.
                # Aqui, vamos usar 'Densidade' se algum dos histnorm for density, senão 'Contagem'.
                y_axis_title = '<b>Count</b>'
                if final_posterior_kwargs['histnorm'] in ['probability', 'probability density'] or \
                   (prior and final_prior_kwargs['histnorm'] in ['probability', 'probability density']):
                    y_axis_title = '<b>Density</b>'
                fig.update_yaxes(title_text=y_axis_title, row=j+1, col=i+1)


        # Configurações gerais do layout
        fig.update_layout(
            barmode='overlay', # Para sobrepor os histogramas
            title_text='<b>Parameters Distribution</b>', # Título geral do gráfico
            title_x=0.5, # Centraliza o título
            hovermode='x unified', # Melhora a interatividade ao passar o mouse
            legend_title_text='<b>Tipo de Distribuição</b>', # Título da legenda
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Ajusta os títulos dos subplots
        for i, element in enumerate(elements):
            fig.layout.annotations[i*rows].update(text=f'<b>{element}</b>', font_size=16)

        return fig

class ST_ID_TimeResponseResults(ST_ID_Results):

    def __init__(self, rotor, t, yout, xout, **kwargs):
        self.rotor = rotor
        self.t = t
        self.yout = yout
        self.xout = xout
        self.probes = kwargs['probes']
        self.data = kwargs['data']
        self.MAP_yout = kwargs.get('MAP_yout', None)
        self.MAP_xout = kwargs.get('MAP_xout', None)
        self.optimum = kwargs.get('optimum', None)
        self.curves = kwargs.get('curves', None)
        self.best_curve_smc = kwargs.get('best_curve', None)
        self.distributions = kwargs.get('distributions', None)
        self.Prior = kwargs.get('Prior', None)

        
    def data_time_response(
        self,
        probe,
        probe_units="rad",
        displacement_units="m",
        time_units="s",
    ):
        """Return the time response given a list of probes in DataFrame format.

        Parameters
        ----------
        probe : list
            List with rs.Probe objects.
        probe_units : str, option
            Units for probe orientation.
            Default is "rad".
        displacement_units : str, optional
            Displacement units.
            Default is 'm'.
        time_units : str
            Time units.
            Default is 's'.

        Returns
        -------
         df : pd.DataFrame
            DataFrame storing the time response measured by probes.
        """
        data = {}

        nodes = self.rotor.nodes
        link_nodes = self.rotor.link_nodes
        ndof = self.rotor.number_dof

        for i, p in enumerate(probe):
            probe_direction = "radial"
            try:
                node = p.node
                angle = p.angle
                probe_tag = p.tag or p.get_label(i + 1)
                if p.direction == "axial":
                    if ndof == 6:
                        probe_direction = p.direction
                    else:
                        continue
            except AttributeError:
                node = p[0]
                warn(
                    "The use of tuples in the probe argument is deprecated. Use the Probe class instead.",
                    DeprecationWarning,
                )
                try:
                    angle = Q_(p[1], probe_units).to("rad").m
                except TypeError:
                    angle = p[1]
                try:
                    probe_tag = p[2]
                except IndexError:
                    probe_tag = f"Probe {i+1} - Node {p[0]}"

            data[f"angle[{i}]"] = angle
            data[f"probe_tag[{i}]"] = probe_tag
            data[f"probe_dir[{i}]"] = probe_direction

            fix_dof = (node - nodes[-1] - 1) * ndof // 2 if node in link_nodes else 0

            if probe_direction == "radial":
                dofx = ndof * node - fix_dof
                dofy = ndof * node + 1 - fix_dof

                # fmt: off
                operator = np.array(
                    [[np.cos(angle), np.sin(angle)],
                    [-np.sin(angle), np.cos(angle)]]
                )

                _probe_resp = operator @ np.vstack((self.yout[:, dofx], self.yout[:, dofy]))
                probe_resp = _probe_resp[0,:]
                # fmt: on
            else:
                dofz = ndof * node + 2 - fix_dof
                probe_resp = self.yout[:, dofz]

            probe_resp = Q_(probe_resp, "m").to(displacement_units).m
            data[f"probe_resp[{i}]"] = probe_resp

        data["time"] = Q_(self.t, "s").to(time_units).m
        df = pd.DataFrame(data)

        return df
    
    def plot(
        self,
        data,
        conf_interval = [],
        probe_units="rad",
        displacement_units="m",
        time_units="s",
        fig=None,
        **kwargs,
    ):
        """Plot time response.

        This method plots the time response given a list of probes with their nodes
        and orientations.

        Parameters
        ----------
        probe : list
            List with rs.Probe objects.
        probe_units : str, option
            Units for probe orientation.
            Default is "rad".
        displacement_units : str, optional
            Displacement units.
            Default is 'm'.
        time_units : str
            Time units.
            Default is 's'.
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        kwargs : optional
            Additional key word arguments can be passed to change the plot layout only
            (e.g. width=1000, height=800, ...).
            *See Plotly Python Figure Reference for more information.

        Returns
        -------
        fig : Plotly graph_objects.Figure()
            The figure object with the plot.
        """

        if fig is None:
            fig = go.Figure()

        probe_list = []

        for i, p in enumerate(self.probes):
             probe_list.append(Probe(p[0], p[1], tag = f'probe_tag {i}'))

        df = self.data_time_response(probe_list, probe_units, displacement_units, time_units)
        _time = df["time"].values

        fig.add_trace(
             go.Scatter(
                  x =_time,
                  y = self.data[data],
                  mode = 'lines',
                  name = f'Data {data}',
                  showlegend = True,
                  hovertemplate=f"Time ({time_units}): %{{x:.2f}}<br>Amplitude ({displacement_units}): %{{y:.2e}}"
             )
        )

        
        if isinstance(self.optimum, np.ndarray):
                probe_tag = df[f"probe_tag[{data}]"].values[0]
                probe_resp = df[f"probe_resp[{data}]"].values

                fig.add_trace(
                    go.Scatter(
                        x=_time,
                        y=Q_(probe_resp, "m").to(displacement_units).m,
                        mode="lines",
                        name=f"Identified {probe_tag}",
                        legendgroup=probe_tag,
                        showlegend=True,
                        hovertemplate=f"Time ({time_units}): %{{x:.2f}}<br>Amplitude ({displacement_units}): %{{y:.2e}}",
                    )
                )
        else:
                pass
        
        if isinstance(self.curves, np.ndarray):

            mag_MAP = Q_(self.MAP_yout, "m").to(displacement_units).m

            y_MAP = mag_MAP[:,self.probes[i][0]*6]*np.cos(np.radians(self.probes[i][1])) + mag_MAP[:,self.probes[i][0]*6+1]*np.sin(np.radians(self.probes[i][1]))

            idx = len(fig.data)

            fig.add_trace(
                        go.Scatter(
                            x= _time,
                            y= y_MAP,
                            mode="lines",
                            line=dict(color=list(tableau_colors)[idx]),
                            name=f"MAP",
                            opacity = 0.5,
                            legendgroup=f"MAP {probe_tag}",
                            showlegend=True,
                            hovertemplate=f"Time ({time_units}): %{{x:.2f}}<br>Amplitude ({displacement_units}): %{{y:.2e}}",
                        ),
                    )
              

            x = np.concatenate((self.t, self.t[::-1]))
            for i, p in enumerate(conf_interval):
                p1 = np.percentile(np.abs(self.curves[:,data,:]), 50 + p / 2, axis=0)
                p2 = np.percentile(np.abs(self.curves[:,data,:]), 50 - p / 2, axis=0)
                fig.add_trace(
                    go.Scatter(
                        x= Q_(x, "s").to(time_units).m,
                        y=Q_(np.concatenate((p1, p2[::-1])), "m").to(displacement_units).m,
                        mode="lines",
                        line=dict(width=1, color=colors1[i]),
                        fill="toself",
                        fillcolor=colors1[i],
                        opacity=0.5,
                        name="confidence interval: {}%".format(p),
                        legendgroup="conf{}".format(i),
                        hovertemplate=("Time: %{x:.3f}<br>" + "Amplitude: %{y:.2e}"),
                    )
                )
        
        

        fig.update_xaxes(title_text=f"Time ({time_units})")
        fig.update_yaxes(title_text=f"Amplitude ({displacement_units})")
        fig.update_layout(**kwargs)

        return fig
    
    def plot_histogram(self, elements=[], parameters=[], prior=True, 
                       histogram_kwargs_posterior=None, histogram_kwargs_prior=None):
    
        '''for values in self.distributions.values():
            for values_aux in values.values():
                n_samples = len(values_aux)
                break
            break'''

        rows = len(parameters)
        cols = len(elements)

        fig = make_subplots(rows=rows, cols=cols,
                            subplot_titles=[f'<b>{element}</b>' for element in elements for _ in range(rows)]
                           )

        # Definir configurações padrão para o histograma POSTERIOR
        default_posterior_kwargs = {
            'opacity': 0.7,
            'histnorm': 'probability',
            'marker_color': '#1f77b4',       # Azul padrão do Plotly
            'marker_line_width': 1,
            'marker_line_color': 'black',
        }
        # Atualiza as configurações padrão do Posterior com as fornecidas pelo usuário
        if histogram_kwargs_posterior is None:
            histogram_kwargs_posterior = {}
        final_posterior_kwargs = {**default_posterior_kwargs, **histogram_kwargs_posterior}

        # Definir configurações padrão para o histograma PRIOR
        default_prior_kwargs = {
            'opacity': 0.7,
            'histnorm': 'probability',
            'marker_color': '#ff7f0e',       # Laranja padrão do Plotly
            'marker_line_width': 1,
            'marker_line_color': 'black',
        }
        # Atualiza as configurações padrão do Prior com as fornecidas pelo usuário
        if histogram_kwargs_prior is None:
            histogram_kwargs_prior = {}
        final_prior_kwargs = {**default_prior_kwargs, **histogram_kwargs_prior}

        # Controlar quais traces mostram a legenda (apenas uma vez para cada)
        show_legend_for_traces = {
            'Posterior': True,
            'Prior': True
        }

        for i, element in enumerate(elements):
            for j, parameter in enumerate(parameters):
                # Dados para o Posterior
                posterior_data = self.distributions[element][parameter]
                
                # Adiciona o trace POSTERIOR
                fig.add_trace(go.Histogram(
                    x=posterior_data,
                    name='Posterior',
                    showlegend=show_legend_for_traces['Posterior'] and (i == 0 and j == 0),
                    **final_posterior_kwargs # Desempacota as configurações do Posterior
                ),
                row=j + 1,
                col=i + 1)

                if prior:
                    # Dados para o Prior
                    prior_data = self.Prior[element][parameter]
                    
                    # Adiciona o trace PRIOR
                    fig.add_trace(go.Histogram(
                        x=prior_data,
                        name='Prior',
                        showlegend=show_legend_for_traces['Prior'] and (i == 0 and j == 0),
                        **final_prior_kwargs # Desempacota as configurações do Prior
                    ),
                    row=j + 1,
                    col=i + 1)
                
                # Atualizar o título do eixo X
                fig.update_xaxes(title_text=f'<b>{parameter}</b>', row=j+1, col=i+1)
                
                # O título do eixo Y agora pode ser mais genérico ou depender do 'histnorm'
                # da primeira categoria plotada ou de uma preferência.
                # Aqui, vamos usar 'Densidade' se algum dos histnorm for density, senão 'Contagem'.
                y_axis_title = '<b>Count</b>'
                if final_posterior_kwargs['histnorm'] in ['probability', 'probability density'] or \
                   (prior and final_prior_kwargs['histnorm'] in ['probability', 'probability density']):
                    y_axis_title = '<b>Density</b>'
                fig.update_yaxes(title_text=y_axis_title, row=j+1, col=i+1)


        # Configurações gerais do layout
        fig.update_layout(
            barmode='overlay', # Para sobrepor os histogramas
            title_text='<b>Parameters Distribution</b>', # Título geral do gráfico
            title_x=0.5, # Centraliza o título
            hovermode='x unified', # Melhora a interatividade ao passar o mouse
            legend_title_text='<b>Tipo de Distribuição</b>', # Título da legenda
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Ajusta os títulos dos subplots
        for i, element in enumerate(elements):
            fig.layout.annotations[i*rows].update(text=f'<b>{element}</b>', font_size=16)

        return fig