"""Distributions module.

This module defines the Distributions class and defines
the distribution of each random parameter.
"""
from collections.abc import Iterable

import numpy as np
import scipy as sp

from ross.units import check_units

__all__ = ["ST_Distribution"]

class ST_Distribution:
    """Create instance of Distribution for parameters.

    Class used to create a probabilistic distribution and define its properties.
    Type of PDF, its conditions and the parameter or the rotor associated 
    should be provided.

    Parameters
    ----------
    name : str, list
        Distribution type.
    info : float, list
        The information needed to built the Distribution.
    param : str, list, pint.Quantity
        The random parameter or a list of random parameters.
    

    Examples
    --------
    >>> # Steel with random Young's modulus.
    >>> from st_distributions import ST_Distribution 
    >>> import numpy as np
    >>> E = np.random.uniform(208e9, 211e9, 5)
    >>> dist_E = ST_Distribution(name="Normal", info=[208e9, 211e9], param="E")
    >>> dist_E.value(1)
    array([6.32013755e+10])
    """

    @check_units
    def __init__(
        self, name, info, param, **kwargs
    ):
        self.name = name
        if " " in name:
            raise ValueError("Spaces are not allowed in Distribution name")
        self.param = str(param)
        self.info = info
        
        attribute_dict = dict(
            name=name,
            info=info,
            param=param,
        )
        self.attribute_dict = attribute_dict
        

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
        >>> from st_distributions import ST_Distribution 
        >>> dist_ky = ST_Distribution(name="Normal", info=[7e5, 1e2], param="kxx")
        >>> dist_ky.info
        [700000.0, 100.0]
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
        value : The corresponding value for the attrbiute_dict's key.
            ***check the correct type for each key in ST_Material
            docstring.

        Raises
        ------
        KeyError
            Raises an error if the parameter doesn't belong to the class.

        Example
        -------
        >>> from st_distributions import ST_Distribution 
        >>> import numpy as np
        >>> E = np.random.uniform(208e9, 211e9, 5)
        >>> dist_ky = ST_Distribution(name="Uniform", info=[7e5, 1e2], param="kxx")
        >>> dist_ky.info = [7.5e6,1e2]
        >>> dist_ky.info
        [7500000.0, 100.0]
        """
        if key not in self.attribute_dict.keys():
            raise KeyError("Object does not have parameter: {}.".format(key))
        self.attribute_dict[key] = value
         

    def value(self,size):
        """Evaluate an array with random values of the PDF.

        Parameters
        ----------
        size : int
            The size of the array with random values.
        """

        if self.name == "Normal":
            Dist = sp.stats.norm(loc = self.info[0], scale=self.info[1])
            val = Dist.rvs(size)[0:size]

                
        elif self.name == "Uniform":
            Dist = sp.stats.uniform(loc = self.info[0], scale=self.info[1]-self.info[0])
            val = Dist.rvs(size)[0:size]

                
        else:
            raise KeyError("Wrong Name: "+self.name+".")

        return val