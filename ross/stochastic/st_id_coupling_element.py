"""Coupling element module for IDENTIFY ROSS.

This module creates an instance of random coupling element for stochastic analysis.
"""

import numpy as np
from ross.units import check_units
import ross as rs
from ross import CouplingElement

__all__ = ["ST_ID_CouplingElement"]

class ST_ID_CouplingElement(CouplingElement):
    """Random coupling element.

    This class creates a  random coupling element from input data of inertia and mass
    from the left station and right station, and also translational and rotational
    stiffness and damping values. The matrices will be defined considering the
    same local coordinate vector of the `ShaftElement`.

    Parameters
    ----------
    m_l : float, pint.Quantity
        Mass of the left station of coupling element (kg).
    m_r : float, pint.Quantity
        Mass of the right station of coupling element (kg).
    Ip_l : float, pint.Quantity
        Polar moment of inertia of the left station of the coupling element (kg).
    Ip_r : float, pint.Quantity
        Polar moment of inertia of the right station of the coupling element (kg.m²).
    Id_l : float, pint.Quantity, optional
        Diametral moment of inertia of the left station of the coupling element (kg.m²).
        If not given, it is assumed to be half of `Ip_l`.
    Id_r : float, pint.Quantity, optional
        Diametral moment of inertia of the right station of the coupling element (kg.m²).
        If not given, it is assumed to be half of `Ip_r`.
    kt_x : float, optional
        Translational stiffness in `x` (N/m).
        Default is 0.
    kt_y : float, optional
        Translational stiffness in `y` (N/m).
        Default is 0.
    kt_z : float, optional
        Axial stiffness (N/m).
        Default is 0.
    kr_x : float, optional
        Rotational stiffness in `x` (N.m/rad).
        Default is 0.
    kr_y : float, optional
        Rotational stiffness in `y` (N.m/rad).
        Default is 0.
    kr_z : float, optional
        Torsional stiffness (N.m/rad).
        Default is 0.
    ct_x : float, optional
        Translational damping in `x` (N.s/m).
        Default is 0.
    ct_y : float, optional
        Translational damping in `y` (N.s/m).
        Default is 0.
    ct_z : float, optional
        Axial damping (N.s/m).
        Default is 0.
    cr_x : float, optional
        Rotational damping in `x` (N.m.s/rad).
        Default is 0.
    cr_y : float, optional
        Rotational damping in `y` (N.m.s/rad).
        Default is 0.
    cr_z : float, optional
        Torsional damping (N.m.s/rad).
        Default is 0.
    o_d : float, optional
        Outer diameter (m). This parameter is primarily used for visualization
        purposes and does not affect calculations.
    L : float, optional
        Element length (m). This parameter is primarily used for visualization
        purposes and does not affect calculations.
    n : int, optional
        Element number (coincident with it's first node).
        If not given, it will be set when the rotor is assembled
        according to the element's position in the list supplied to
        the rotor constructor.
    tag : str, optional
        A tag to name the element
        Default is None
    scale_factor: float or str, optional
        The scale factor is used to scale the coupling drawing.
        Default is 1.
    color : str, optional
        A color to be used when the element is represented.
        Default is '#647e91'.
    to_identify : list
        List of the object attributes to become stochastic.
        Possibilities:
            [m_l, m_r, Ip_l, Ip_r, Id_l, Id_r, kt_x, kt_y, kt_z, kr_x, kr_y, kr_z, ct_x, ct_y, ct_z, cr_x, cr_y, cr_z, o_d]

    Examples
    --------
"""

    @check_units
    def __init__(
        self,
        m_l,
        m_r,
        Ip_l,
        Ip_r,
        Id_l=0,
        Id_r=0,
        kt_x=0,
        kt_y=0,
        kt_z=0,
        kr_x=0,
        kr_y=0,
        kr_z=0,
        ct_x=0,
        ct_y=0,
        ct_z=0,
        cr_x=0,
        cr_y=0,
        cr_z=0,
        o_d=None,
        L=None,
        n=None,
        tag=None,
        scale_factor=1,
        color="#647e91",
        to_identify = None,
    ):
               
        
        attribute_dict = dict(
            m_l = m_l,
            m_r = m_r,
            Ip_l = Ip_l,
            Ip_r = Ip_r,
            Id_l = Id_l,
            Id_r = Id_r,
            kt_x = kt_x,
            kt_y = kt_y,
            kt_z = kt_z,
            kr_x = kr_x,
            kr_y = kr_y,
            kr_z = kr_z,
            ct_x = ct_x,
            ct_y = ct_y,
            ct_z = ct_z,
            cr_x = cr_x,
            cr_y = cr_y,
            cr_z = cr_z,
            o_d = o_d,
            L = L,
            n = n,
            tag = tag,
            scale_factor = scale_factor,
            color = color,
        )

        self.attribute_dict = attribute_dict
        self.to_identify = to_identify
        self.interval_params = {}
        self.param_values = {}

        if to_identify != None:
            for params in self.to_identify:
                if type(self.attribute_dict[params]) != list and type(self.attribute_dict[params]) != np.ndarray:
                    raise KeyError(f'The parameter {params} must be a list or numpy.ndarray')
                
            for params in self.to_identify:
                self.interval_params[params] = self.attribute_dict[params]

            for params in self.to_identify:
                value = np.random.uniform(self.attribute_dict[params][0],self.attribute_dict[params][1])
                self.param_values[params] = value
                self.attribute_dict[params] = value

        

        super().__init__(attribute_dict['m_l'],
                         attribute_dict['m_r'],
                         attribute_dict['Ip_l'],
                         attribute_dict['Ip_r'],
                         attribute_dict['Id_l'],
                         attribute_dict['Id_r'],
                         attribute_dict['kt_x'],
                         attribute_dict['kt_y'],
                         attribute_dict['kt_z'],
                         attribute_dict['kr_x'],
                         attribute_dict['kr_y'],
                         attribute_dict['kr_z'],
                         attribute_dict['ct_x'],
                         attribute_dict['ct_y'],
                         attribute_dict['ct_z'],
                         attribute_dict['cr_x'],
                         attribute_dict['cr_y'],
                         attribute_dict['cr_z'],
                         attribute_dict['o_d'],
                         attribute_dict['L'],
                         attribute_dict['n'],
                         attribute_dict['tag'],
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

        f_list = (rs.CouplingElement(*arg) for arg in new_args)
        
        return f_list
    
    def new_coupling(self):

        for params, value in self.param_values.items():
            self.attribute_dict[params] = value

        return list(iter(self.generator()))[0]