"""Shaft element module for STOCHASTIC ROSS.

This module creates an instance of random shaft element for stochastic
analysis.
"""

#import ross
from ross.shaft_element import ShaftElement
#from stochastic.st_materials import ST_Material2
from ross.units import Q_, check_units

from st_materials import ST_Material2

__all__ = ["ST_ShaftElement", "st_shaft_example"]


class ST_ShaftElement2:
    """Random shaft element.

    Creates an object containing a generator with random instances of
    ShaftElement.

    Parameters
    ----------
    L : float, pint.Quantity, list
        Element length.
        Input a list to make it random.
    idl : float, pint.Quantity, list
        Inner diameter of the element at the left position.
        Input a list to make it random.
    odl : float, pint.Quantity, list
        Outer diameter of the element at the left position.
        Input a list to make it random.
    idr : float, pint.Quantity, list, optional
        Inner diameter of the element at the right position
        Default is equal to idl value (cylindrical element)
        Input a list to make it random.
    odr : float, pint.Quantity, list, optional
        Outer diameter of the element at the right position.
        Default is equal to odl value (cylindrical element)
        Input a list to make it random.
    material : ross.material, list of ross.material
        Shaft material.
        Input a list to make it random.
    n : int, optional
        Element number (coincident with it's first node).
        If not given, it will be set when the rotor is assembled
        according to the element's position in the list supplied to
    axial_force : float, list, optional
        Axial force.
        Input a list to make it random.
        Default is 0.
    torque : float, list, optional
        Torque
        Input a list to make it random.
        Default is 0.
    shear_effects : bool, optional
        Determine if shear effects are taken into account.
        Default is True.
    rotary_inertia : bool, optional
        Determine if rotary_inertia effects are taken into account.
        Default is True.
    gyroscopic : bool, optional
        Determine if gyroscopic effects are taken into account.
        Default is True.
    shear_method_calc : str, optional
        Determines which shear calculation method the user will adopt.
        Default is 'cowper'
    is_random : list
        List of the object attributes to become random.
        Possibilities:
            ["L", "idl", "odl", "idr", "odr", "material", "axial_force", "torque"]

    Example
    -------
    >>> import numpy as np
    >>> import ross.stochastic as srs
    >>> size = 5
    >>> E = np.random.uniform(208e9, 211e9, size)
    >>> st_steel = srs.ST_Material(name="Steel", rho=7810, E=E, G_s=81.2e9)
    >>> elms = srs.ST_ShaftElement(L=1,
    ...                            idl=0,
    ...                            odl=np.random.uniform(0.1, 0.2, size),
    ...                            material=st_steel,
    ...                            is_random=["odl", "material"],
    ...                            )
    >>> len(list(iter(elms)))
    5
    """

    @check_units
    def __init__(
        self,
        L,
        idl,
        odl,
        idr=None,
        odr=None,
        material=None,
        n=None,
        axial_force=0,
        torque=0,
        shear_effects=True,
        rotary_inertia=True,
        gyroscopic=True,
        shear_method_calc="cowper",
        is_random=None,
    ):
        if idr is None:
            idr = idl
            if "idl" in is_random and "idr" not in is_random:
                is_random.append("idr")
        if odr is None:
            odr = odl
            if "odl" in is_random and "odr" not in is_random:
                is_random.append("odr")
        if isinstance(material, ST_Material2):
            material = list(iter(material))[0]

        attribute_dict = dict(
            L=L,
            idl=idl,
            odl=odl,
            idr=idr,
            odr=odr,
            material=material,
            n=n,
            axial_force=axial_force,
            torque=torque,
            shear_effects=shear_effects,
            rotary_inertia=rotary_inertia,
            gyroscopic=gyroscopic,
            shear_method_calc=shear_method_calc,
            tag=None,
        )
        self.is_random = is_random
        self.attribute_dict = attribute_dict

    def __iter__(self):
        """Return an iterator for the container.

        Returns
        -------
        An iterator over random shaft elements.

        Examples
        --------
        >>> import ross.stochastic as srs
        >>> elm = srs.st_shaft_example()
        >>> len(list(iter(elm)))
        2
        """
        return iter(self.random_var(self.is_random, self.attribute_dict))

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
        >>> import numpy as np
        >>> import ross.stochastic as srs
        >>> size = 5
        >>> E = np.random.uniform(208e9, 211e9, size)
        >>> st_steel = srs.ST_Material(name="Steel", rho=7810, E=E, G_s=81.2e9)
        >>> elms = srs.ST_ShaftElement(L=1,
        ...                            idl=0,
        ...                            odl=np.random.uniform(0.1, 0.2, size),
        ...                            material=st_steel,
        ...                            is_random=["odl", "material"],
        ...                            )
        >>> elms["L"]
        1
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
            ***check the correct type for each key in ST_ShaftElement
            docstring.

        Raises
        ------
        KeyError
            Raises an error if the parameter doesn't belong to the class.

        Example
        -------
        >>> import numpy as np
        >>> import ross.stochastic as srs
        >>> size = 5
        >>> E = np.random.uniform(208e9, 211e9, size)
        >>> st_steel = srs.ST_Material(name="Steel", rho=7810, E=E, G_s=81.2e9)
        >>> elms = srs.ST_ShaftElement(L=1,
        ...                            idl=0,
        ...                            odl=np.random.uniform(0.1, 0.2, size),
        ...                            material=st_steel,
        ...                            is_random=["odl", "material"],
        ...                            )
        >>> elms["odl"] = np.linspace(0.1, 0.2, 5)
        >>> elms["odl"]
        array([0.1  , 0.125, 0.15 , 0.175, 0.2  ])
        """
        if key not in self.attribute_dict.keys():
            raise KeyError("Object does not have parameter: {}.".format(key))
        self.attribute_dict[key] = value

    def random_var(self, is_random, *args):
        """Generate a list of objects as random attributes.

        This function creates a list of objects with random values for selected
        attributes from ross.ShaftElement.

        Parameters
        ----------
        is_random : list
            List of the object attributes to become stochastic.
        *args : dict
            Dictionary instanciating the ross.ShaftElement class.
            The attributes that are supposed to be stochastic should be
            set as lists of random variables.

        Returns
        -------
        f_list : generator
            Generator of random objects.
        """
        new_args = []
        for i in range(len(self.is_random)):
            arg = []
            for key, param in self.attribute_dict.items():
                if key in is_random:
                    arg.append(param.value(1)[0])
                else:
                    arg.append(param)
            new_args.append(arg)
        f_list = (ShaftElement(*arg) for arg in new_args)

        return f_list