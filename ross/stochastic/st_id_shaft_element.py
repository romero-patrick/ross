"""Shaft element module for IDENTIFY ROSS.

This module creates an instance of random shaft for stochastic analysis.
"""

import numpy as np
from ross.units import check_units
import ross as rs
from ross import ShaftElement

__all__ = ["ST_ID_ShaftElement"]

class ST_ID_ShaftElement(ShaftElement):
    """Creates an object containing a random instances of
    ShaftElement.

    Parameters
    ----------
    L : float, pint.Quantity
        Element length.
    idl : float, pint.Quantity
        Inner diameter of the element at the left node (m).
    odl : float, pint.Quantity
        Outer diameter of the element at the left node (m).
    idr : float, pint.Quantity, optional
        Inner diameter of the element at the right node (m).
        Default is equal to idl value for cylindrical element.
    odr : float, pint.Quantity, optional
        Outer diameter of the element at the right node (m).
        Default is equal to odl value for cylindrical element.
    material : ross.Material
        Shaft material.
    alpha : float, optional
        Proportional damping coefficient, associated to the element Mass matrix
    beta : float, optional
        Proportional damping coefficient, associated to the element Stiffness matrix
    n : int, optional
        Element number, coincident it's first node.
        If not given, it will be set when the rotor is assembled
        according to the element's position in the list supplied to
        the rotor constructor.
    axial_force : float, optional
        Axial force (N).
        Default is zero.
    torque : float, optional
        Torque moment (N*m).
        Default is zero.
    shear_effects : bool, optional
        Determine if shear effects are taken into account;
        Default is True.
    rotary_inertia : bool, optional
        Determine if rotary_inertia effects are taken into account;
        Default is True.
    gyroscopic : bool, optional
        Determine if gyroscopic effects are taken into account;
        Default is True.
    tag : str, optional
        Element tag;
        Default is None.
    to_identify : list
        List of the object attributes to become stochastic.
        Possibilities:
            ["L", "idl", "odl", "idr", "odr", "axial_force", "torque", "alpha", "beta"]

    Examples
    --------
    >>> from ross.materials import steel
    >>> shaft1 = ID_ShaftElement(L=0.5, idl=0.0, odl=0.01, idr=0.0, odr=0.01,
    ...                           material=steel, n=0, axial_force=[10,12], torque=30, to_identify = ["axial_force"])
    >>> shaft2 = ID_ShaftElement(L=0.5, idl=0.05, odl=0.1, idr=0.05, odr=0.15,
    ...                           alpha=0.01, beta=100, material=steel,
    ...                           rotary_inertia=False, shear_effects=False)
    """


    def __init__(
            self,
            L,
            idl,
            odl,
            idr = None,
            odr = None,
            material = None,
            n = None,
            axial_force = 0.0,
            torque = 0.0,
            shear_effects = True,
            rotary_inertia = True,
            gyroscopic = True,
            shear_method_calc = 'cowper',
            tag = None,
            alpha = 0.0,
            beta = 0.0,
            to_identify = None
    ):
        
        attribute_dict = dict(
            L = L,
            idl = idl,
            odl = odl,
            idr = idr,
            odr = odr,
            material = material,
            n = n,
            axial_foce = axial_force,
            torque = torque,
            shear_effects = shear_effects,
            rotary_inertia = rotary_inertia,
            gyroscopic = gyroscopic,
            shear_method_calc = shear_method_calc,
            tag = tag,
            alpha = alpha,
            beta = beta,
        )

        self.attribute_dict = attribute_dict
        self.to_identify = to_identify
        self.interval_params = {}
        self.param_values = {}
        self.id_material = material

        if to_identify != None:
            for params in self.to_identify:
                if type(self.attribute_dict[params]) != list and type(self.attribute_dict[params]) != np.ndarray:
                    raise KeyError(f'The parameter {params} must be a list or numpy.ndarray')
                
            for params in self.to_identify:
                self.interval_params[params] = self.attribute_dict[params]
                value = np.random.uniform(self.attribute_dict[params][0],self.attribute_dict[params][1])
                self.param_values[params] = value
                self.attribute_dict[params] = value

        if type(self.id_material) != rs.materials.Material:
            self.attribute_dict['material'] = self.id_material.new_material()
            if self.id_material.to_identify != None:
                if self.to_identify == None:
                    self.to_identify = ['material_id']
                else:
                    self.to_identify.append('material_id')

        super().__init__(attribute_dict['L'],
                         attribute_dict['idl'],
                         attribute_dict['odl'],
                         attribute_dict['idr'],
                         attribute_dict['odr'],
                         attribute_dict['material'],
                         attribute_dict['n'],
                         attribute_dict['axial_foce'],
                         attribute_dict['torque'],
                         attribute_dict['shear_effects'],
                         attribute_dict['rotary_inertia'],
                         attribute_dict['gyroscopic'],
                         attribute_dict['shear_method_calc'],
                         attribute_dict['tag'],
                         attribute_dict['alpha'],
                         attribute_dict['beta'])

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
            >>> from ross.materials import steel
            >>> shaft1 = ID_ShaftElement(L=0.5, idl=0.0, odl=0.01, idr=0.0, odr=0.01,
            ...                           material=steel, n=0, axial_force=[10,12], torque=30, to_identify = ["axial_force"])
            >>> shaft1['L']
            >>> 0.5
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
                ***check the correct type for each key in ID_ShaftElement
                docstring.

            Raises
            ------
            KeyError
                Raises an error if the parameter doesn't belong to the class.

            Example
            -------
            >>> from ross.materials import steel
            >>> shaft2 = ID_ShaftElement(L=0.5, idl=0.05, odl=0.1, idr=0.05, odr=0.15,
            ...                           alpha=0.01, beta=100, material=steel,
            ...                           rotary_inertia=False, shear_effects=False)
            >>> shaft2['idl'] = 0.1
            >>> shaft2['idl']
            >>> 0.1
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

        f_list = (rs.ShaftElement(*arg) for arg in new_args)

        return f_list
    
    def new_shaft(self):

        self.attribute_dict['material'] = self.id_material.new_material()

        for params, value in self.param_values.items():
                self.attribute_dict[params] = value
            
        return list(iter(self.generator()))[0]