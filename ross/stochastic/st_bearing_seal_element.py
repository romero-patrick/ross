"""Bearing element module for STOCHASTIC ROSS.

This module creates an instance of random bearing for stochastic analysis.
"""
import numpy as np

import ross

from ross.bearing_seal_element import BearingElement
from ross.fluid_flow import fluid_flow as flow
from ross.fluid_flow.fluid_flow_coefficients import (
    calculate_stiffness_and_damping_coefficients,
)

from ross.units import check_units

__all__ = ["ST_BearingElement2"]


class ST_BearingElement2:
    """Random bearing element.

    Creates an object containing a list with random instances of
    BearingElement.

    Considering constant coefficients, use an 1-D array to make it random.
    Considering varying coefficients to the frequency, use a 2-D array to
    make it random (*see the Examples below).

    Parameters
    ----------
    n: int
        Node which the bearing will be located in
    kxx: float, 1-D array, 2-D array
        Direct stiffness in the x direction.
    cxx: float, 1-D array, 2-D array
        Direct damping in the x direction.
    kyy: float, 1-D array, 2-D array, optional
        Direct stiffness in the y direction.
        (defaults to kxx)
    kxy: float, 1-D array, 2-D array, optional
        Cross coupled stiffness in the x direction.
        (defaults to 0)
    kyx: float, 1-D array, 2-D array, optional
        Cross coupled stiffness in the y direction.
        (defaults to 0)
    cyy: float, 1-D array, 2-D array, optional
        Direct damping in the y direction.
        (defaults to cxx)
    cxy: float, 1-D array, 2-D array, optional
        Cross coupled damping in the x direction.
        (defaults to 0)
    cyx: float, 1-D array, 2-D array, optional
        Cross coupled damping in the y direction.
        (defaults to 0)
    frequency: array, optional
        Array with the frequencies (rad/s).
    tag: str, optional
        A tag to name the element
        Default is None.
    n_link: int, optional
        Node to which the bearing will connect. If None the bearing is
        connected to ground.
        Default is None.
    scale_factor: float, optional
        The scale factor is used to scale the bearing drawing.
        Default is 1.
    is_random : list
        List of the object attributes to become stochastic.
        Possibilities:
            ["kxx", "kxy", "kyx", "kyy", "cxx", "cxy", "cyx", "cyy"]

    Attributes
    ----------
    elements_list : list
        display the list with random bearing elements.

    Example
    -------
    >>> import numpy as np
    >>> import ross.stochastic as srs

    # Uncertanties on constant bearing coefficients

    >>> s = 10
    >>> kxx = np.random.uniform(1e6, 2e6, s)
    >>> cxx = np.random.uniform(1e3, 2e3, s)
    >>> elms = srs.ST_BearingElement(n=1,
    ...                              kxx=kxx,
    ...                              cxx=cxx,
    ...                              is_random = ["kxx", "cxx"],
    ...                              )
    >>> len(list(iter(elms)))
    10

    # Uncertanties on bearing coefficients varying with frequency

    >>> s = 5
    >>> kxx = [np.random.uniform(1e6, 2e6, s),
    ...        np.random.uniform(2.3e6, 3.3e6, s)]
    >>> cxx = [np.random.uniform(1e3, 2e3, s),
    ...        np.random.uniform(2.1e3, 3.1e3, s)]
    >>> frequency = np.linspace(500, 800, len(kxx))
    >>> elms = srs.ST_BearingElement(n=1,
    ...                              kxx=kxx,
    ...                              cxx=cxx,
    ...                              frequency=frequency,
    ...                              is_random = ["kxx", "cxx"],
    ...                              )
    >>> len(list(iter(elms)))
    5
    """

    @check_units
    def __init__(
        self,
        n,
        kxx,
        cxx,
        mxx=None,
        kyy=None,
        kxy=0,
        kyx=0,
        cyy=None,
        cxy=0,
        cyx=0,
        myy=None,
        mxy=0,
        myx=0,
        frequency=None,
        tag=None,
        n_link=None,
        scale_factor=1,
        is_random=None,
    ):
        if "frequency" in is_random:
            raise ValueError("frequency can not be a random variable")
        
    #avaliar isso abaixo

        '''
        if kyy is None:
            kyy = kxx
            if "kxx" in is_random and "kyy" not in is_random:
                is_random.append("kyy")
        if cyy is None:
            cyy = cxx
            if "cxx" in is_random and "cyy" not in is_random:
                is_random.append("cyy")

        if myy is None:
            if mxx is None:
                myy = 0
                mxx = 0
            else:
                myy = mxx
            if "mxx" in is_random and "myy" not in is_random:
                is_random.append("myy")
        '''
        attribute_dict = dict(
            n=n,
            kxx=kxx,
            cxx=cxx,
            mxx=mxx,
            kyy=kyy,
            kxy=kxy,
            kyx=kyx,
            cyy=cyy,
            cxy=cxy,
            cyx=cyx,
            myy=myy,
            mxy=mxy,
            myx=myx,
            frequency=frequency,
            tag=tag,
            n_link=n_link,
            scale_factor=scale_factor,
        )
        self.is_random = is_random
        self.attribute_dict = attribute_dict
        self.n = n
        self.kxx= kxx
        self.tag = tag

    def __iter__(self):
        """Return an iterator for the container.

        Returns
        -------
        An iterator over random bearing elements.

        Examples
        --------
        >>> import ross.stochastic as srs
        >>> bearing = srs.st_bearing_example()
        >>> len(list(iter(bearing)))
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
        >>> s = 5
        >>> kxx = np.random.uniform(1e6, 2e6, s)
        >>> cxx = np.random.uniform(1e3, 2e3, s)
        >>> elms = srs.ST_BearingElement(n=1,
        ...                              kxx=kxx,
        ...                              cxx=cxx,
        ...                              is_random = ["kxx", "cxx"],
        ...                              )
        >>> elms["n"]
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
            ***check the correct type for each key in ST_BearingElement
            docstring.

        Raises
        ------
        KeyError
            Raises an error if the parameter doesn't belong to the class.

        Example
        -------
        >>> import numpy as np
        >>> import ross.stochastic as srs
        >>> s = 5
        >>> kxx = np.random.uniform(1e6, 2e6, s)
        >>> cxx = np.random.uniform(1e3, 2e3, s)
        >>> elms = srs.ST_BearingElement(n=1,
        ...                              kxx=kxx,
        ...                              cxx=cxx,
        ...                              is_random = ["kxx", "cxx"],
        ...                              )
        >>> elms["kxx"] = np.linspace(3e6, 5e6, 5)
        >>> elms["kxx"]
        array([3000000., 3500000., 4000000., 4500000., 5000000.])
        """
        if key not in self.attribute_dict.keys():
            raise KeyError("Object does not have parameter: {}.".format(key))
        self.attribute_dict[key] = value

    def random_var(self, is_random, *args):
        """Generate a list of objects as random attributes.

        This function creates a list of objects with random values for selected
        attributes from ross.BearingElement.

        Parameters
        ----------
        is_random : list
            List of the object attributes to become stochastic.
        *args : dict
            Dictionary instanciating the ross.BearingElement class.
            The attributes that are supposed to be stochastic should be
            set as lists of random variables.

        Returns
        -------
        f_list : generator
            Generator of random objects.
        """
        #args_dict = args[0]
        new_args = []
        
        for i in range(len(self.is_random)):
            arg = []
            for key, param in self.attribute_dict.items():
                #param pode ser uma lista de distribuições, como tratar(?)
                if key in is_random:
                    self[key]
                    arg.append([param.value(1)[0]])
                else:
                    arg.append(param)
            new_args.append(arg)

        f_list = (BearingElement(*arg) for arg in new_args)

        return f_list