"""Bearing element module for IDENTIFY ROSS.

This module creates an instance of random bearing for stochastic analysis.
"""

import numpy as np
from ross.units import check_units
import ross as rs
from ross import BearingElement

__all__ = ["ST_ID_BearingElement"]

class ST_ID_BearingElement(BearingElement):
    """Random bearing element.

    Creates an object containing a random instances of
    BearingElement.

    Considering constant coefficients, use a list with minimum and maximum values to make it random

    Parameters
    ----------
    kxx : float, array, pint.Quantity
        Direct stiffness in the x direction (N/m).
    cxx : float, array, pint.Quantity
        Direct damping in the x direction (N*s/m).
    kyy : float, array, pint.Quantity, optional
        Direct stiffness in the y direction (N/m).
        Defaults to kxx
    cyy : float, array, pint.Quantity, optional
        Direct damping in the y direction (N*s/m).
        Default is to cxx
    kxy : float, array, pint.Quantity, optional
        Cross stiffness between xy directions (N/m).
        Default is 0
    kyx : float, array, pint.Quantity, optional
        Cross stiffness between yx directions (N/m).
        Default is 0
    kzz : float, array, pint.Quantity, optional
        Direct stiffness in the z direction (N/m).
        Default is 0
    cxy : float, array, pint.Quantity, optional
        Cross damping between xy directions (N*s/m).
        Default is 0
    cyx : float, array, pint.Quantity, optional
        Cross damping between yx directions (N*s/m).
        Default is 0
    czz : float, array, pint.Quantity, optional
        Direct damping in the z direction (N*s/m).
        Default is 0
    frequency : array, pint.Quantity, optional
        Array with the frequencies (rad/s).
    tag : str, optional
        A tag to name the element
        Default is None
    n_link : int, optional
        Node to which the bearing will connect. If None the bearing is
        connected to ground.
        Default is None.
    scale_factor : float, optional
        The scale factor is used to scale the bearing drawing.
        Default is 1.
    to_identify : list
        List of the object attributes to become stochastic.
        Possibilities:
            ["kxx", "kxy", "kyx", "kyy", "kzz", "cxx", "cxy", "cyx", "cyy", "czz"]

    Examples
    --------

    >>> n = 0
    >>> kxx = [1.0e7,1.5e7]
    >>> kyy = 1.5e7
    >>> kzz = 5.0e5
    >>> bearing = ID_BearingElement(n=n, kxx=kxx, kyy=kyy, kzz=kzz,
    ...                              cxx=0, cyy=0, to_identify = ['kxx'])
    >>> bearing['kxx']
    >>> 10183172.838301644
"""

    @check_units
    def __init__(
        self,
        n,
        kxx,
        cxx,
        mxx=0,
        kyy=None,
        kxy=0,
        kyx=0,
        cyy=None,
        cxy=0,
        cyx=0,
        myy=None,
        mxy=0,
        myx=0,
        kzz=0,
        czz=0,
        mzz=0,
        frequency=None,
        tag=None,
        n_link=None,
        scale_factor=1,
        color="#355d7a",
        to_identify = None,
    ):
               
        
        attribute_dict = dict(
            n = n,
            kxx = kxx,
            cxx = cxx,
            mxx = mxx,
            kyy = kyy,
            kxy = kxy,
            kyx = kyx,
            cyy = cyy,
            cxy = cxy,
            cyx = cyx,
            myy = myy,
            mxy = mxy,
            myx = myx,
            kzz = kzz,
            czz = czz,
            mzz = mzz,
            frequency = frequency,
            tag = tag,
            n_link = n_link,
            scale_factor = scale_factor,
            color = color,
        )

        self.attribute_dict = attribute_dict
        self.to_identify = to_identify
        self.interval_params = {}
        self.param_values = {}

        if to_identify != None:
            if "frequency" in to_identify:
                raise ValueError("frequency can not be a variable to identify") 

            for params in self.to_identify:
                if type(self.attribute_dict[params]) != list and type(self.attribute_dict[params]) != np.ndarray:
                    raise KeyError(f'The parameter {params} must be a list or numpy.ndarray')
                
            for params in self.to_identify:
                self.interval_params[params] = self.attribute_dict[params]

            for params in self.to_identify:
                value = np.random.uniform(self.attribute_dict[params][0],self.attribute_dict[params][1])
                self.param_values[params] = value
                self.attribute_dict[params] = value

        

        super().__init__(attribute_dict['n'],
                         attribute_dict["kxx"],
                         attribute_dict['cxx'],
                         attribute_dict['mxx'],
                         attribute_dict["kyy"],
                         attribute_dict["kxy"],
                         attribute_dict["kyx"],
                         attribute_dict['cyy'],
                         attribute_dict['cxy'],
                         attribute_dict['cyx'],
                         attribute_dict["myy"],
                         attribute_dict["mxy"],
                         attribute_dict['myx'],
                         attribute_dict['kzz'],
                         attribute_dict['czz'],
                         attribute_dict['mzz'],
                         attribute_dict['frequency'],
                         attribute_dict['tag'],
                         attribute_dict['n_link'],
                         attribute_dict['scale_factor'],
                         attribute_dict['color'])


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

        >>> n = 0
        >>> kxx = [1.0e7,1.5e7]
        >>> kyy = 1.5e7
        >>> kzz = 5.0e5
        >>> bearing = ID_BearingElement(n=n, kxx=kxx, kyy=kyy, kzz=kzz,
        ...                              cxx=0, cyy=0, to_identify = ['kxx'])
        >>> bearing['kxx']
        >>> 10183172.838301644
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
            ***check the correct type for each key in ID_BearingElement
            docstring.

        Raises
        ------
        KeyError
            Raises an error if the parameter doesn't belong to the class.

        Example
        -------
        >>> n = 0
        >>> kxx = [1.0e7,1.5e7]
        >>> kyy = 1.5e7
        >>> kzz = 5.0e5
        >>> bearing = ID_BearingElement(n=n, kxx=kxx, kyy=kyy, kzz=kzz,
        ...                              cxx=0, cyy=0, to_identify = ['kxx'])
        >>> bearing['kyy'] = 2e7
        >>> bearing['kyy']
        >>> 20000000.0
        """

        if key not in self.attribute_dict.keys():
            raise KeyError("Object does not have parameter: {}.".format(key))
        self.attribute_dict[key] = value        
        if key in self.param_values:
            self.param_values[key] = value

    
    def generator(self):

        args = []
        
        for value in self.attribute_dict.values():
            args.append(value)

        new_args = [args]

        f_list = (rs.BearingElement(*arg) for arg in new_args)
        
        return f_list
    
    def new_bearing(self):

        for params, value in self.param_values.items():
            self.attribute_dict[params] = value

        return list(iter(self.generator()))[0]

