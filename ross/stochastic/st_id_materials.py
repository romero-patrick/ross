"""Materials module for IDENTIFY ROSS.

This module creates an instance of random material for stochastic analysis.
"""

import numpy as np
from ross.units import check_units
import ross as rs

__all__ = ["ST_ID_Material"]

class ST_ID_Material:
    """Material used on shaft and disks.

    Class used to create a stochastic material and define its properties.
    Density and at least 2 arguments from E, G_s and Poisson should be
    provided.

    Parameters
    ----------
    name : str
        Material name.
    rho : float, pint.Quantity
        Density (kg/m**3).
    E : float, pint.Quantity, optional
        Young's modulus (N/m**2).
    G_s : float, pint.Quantity, optional
        Shear modulus (N/m**2).
    Poisson : float, optional
        Poisson ratio (dimensionless).
    color : str, optional
        Color that will be used on plots.
    to_identify : list
        List of the object attributes to become stochastic.
        Possibilities:
            ["rho", "E", "G_s"]

    Examples
    --------    
    >>> AISI4140 = ID_Material(name="AISI4140", rho=7850, E=203.2e9, G_s=80e9)
    >>> Steel = ID_Material(name="Steel", rho=7850, E=211e9, G_s=[80e9, 82e9], to_identify = ['G_s'])
    >>> AISI4140['rho']
    >>> 7850
    >>> Steel['G_s']
    >>> 81116119708.10237 
    """

    def __init__(
        self,
        name,
        rho,
        E = None,
        G_s = None,
        Poisson = None,
        color = '#525252',
        to_identify = None
    ):
        self.name = str(name)
        if " " in name:
            raise ValueError("Spaces are not allowed in Material name")

        given_args = []
        for arg in ["E", "G_s", "Poisson"]:
            if locals()[arg] is not None:
                given_args.append(arg)
        if len(given_args) != 2:
            raise ValueError(
                "Exactly 2 arguments from E, G_s and Poisson should be provided"
            )

        attribute_dict = dict(
            name = self.name,
            rho = rho,
            E = E,
            G_s = G_s,
            Poisson = Poisson,
            color = color
        )

        self.attribute_dict = attribute_dict
        self.to_identify = to_identify
        self.interval_params = {}
        self.param_values = {}

        if to_identify != None:

            if 'name' in to_identify:
                raise KeyError('name can not be a variable to identify')
            for params in self.to_identify:
                if type(self.attribute_dict[params]) != list and type(self.attribute_dict[params]) != np.ndarray:
                    raise KeyError(f'The parameter {params} must be a list or numpy.ndarray')
                
            for params in self.to_identify:
                self.interval_params[params] = self.attribute_dict[params]

            for params in self.to_identify:
                value = np.random.uniform(self.attribute_dict[params][0],self.attribute_dict[params][1])
                self.param_values[params] = value
                self.attribute_dict[params] = value

    def __getitem__(self, key):    
            """Return the value for a given key from attribute_dict.

            Parameters
            ----------
            key : str
                A class parameter as string.

            Raises
            ------
            KeyError
                Raises an error if the parameter doesn't belong to the class.

            Returns
            -------
            Return the value for the given key.

            Example
            -------
            >>> AISI4140 = ID_Material(name="AISI4140", rho=7850, E=203.2e9, G_s=80e9)
            >>> AISI4140['rho']
            >>> 7850
            """

            if key not in self.attribute_dict.keys():
                raise KeyError("Object does not have parameter: {}.".format(key))

            return self.attribute_dict[key]
    
    def __setitem__(self, key, value):
            """Set new parameter values for the object.

            Function to change a parameter value.
            It's not allowed to add new parameters to the object.

            Parameters
            ----------
            key : str
                A class parameter as string.
            value : The corresponding value for the attribute_dict's key.
                ***check the correct type for each key in ID_Material
                docstring.

            Raises
            ------
            KeyError
                Raises an error if the parameter doesn't belong to the class.

            Example
            -------
            >>> AISI4140 = ID_Material(name="AISI4140", rho=7850, E=203.2e9, G_s=80e9)
            >>> AISI4140['rho']
            >>> 7850
            >>> AISI4140['rho'] = 7800
            >>> AISI4140['rho']
            >>> 7800
            """

            if key not in self.attribute_dict.keys():
                raise KeyError("Object does not have parameter: {}.".format(key))
            self.attribute_dict[key] = value        
            if key in self.param_values:
                self.param_values[key] = value

    def __repr__(self):
        return (
            f"Material"
            f"(name={self.attribute_dict['name']}, rho={self.attribute_dict['rho']}, G_s={self.attribute_dict['G_s']},"
            f"E={self.attribute_dict['E']}, color={self.attribute_dict['color']!r})"
        )

    def generator(self):

        args = []
        
        for value in self.attribute_dict.values():
            args.append(value)

        new_args = [args]
        
        f_list = (rs.Material(*arg) for arg in new_args)

        return f_list
    
    def new_material(self):
        
        for params, value in self.param_values.items():
            self.attribute_dict[params] = value

        return list(iter(self.generator()))[0]