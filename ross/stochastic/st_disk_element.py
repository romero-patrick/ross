"""Disk element module for STOCHASTIC ROSS.

This module creates an instance of random disk element for stochastic
analysis.
"""
import numpy as np

from ross.disk_element import DiskElement
from ross.stochastic.st_materials import ST_Material
from ross.stochastic.st_results_elements import plot_histogram
from ross.units import check_units

__all__ = ["ST_DiskElement", "st_disk_example"]


class ST_DiskElement2:
    """Random disk element.

    Creates an object containing a list with random instances of DiskElement.

    Parameters
    ----------
    n: int
        Node in which the disk will be inserted.
    m : float, list
        Mass of the disk element.
        Input a list to make it random.
    Id : float, list
        Diametral moment of inertia.
        Input a list to make it random.
    Ip : float, list
        Polar moment of inertia
        Input a list to make it random.
    tag : str, optional
        A tag to name the element
        Default is None
    color : str, optional
        A color to be used when the element is represented.
        Default is "Firebrick".
    is_random : list
        List of the object attributes to become random.
        Possibilities:
            ["m", "Id", "Ip"]

    Example
    -------
    >>> import numpy as np
    >>> import ross.stochastic as srs
    >>> elms = srs.ST_DiskElement(n=1,
    ...                           m=30.0,
    ...                           Id=np.random.uniform(0.20, 0.40, 5),
    ...                           Ip=np.random.uniform(0.15, 0.25, 5),
    ...                           is_random=["Id", "Ip"],
    ...                           )
    >>> len(list(iter(elms)))
    5
    """

    @check_units
    def __init__(
        self,
        n,
        m,
        Id,
        Ip,
        tag=None,
        color="Firebrick",
        is_random=None,
    ):
        attribute_dict = dict(n=n, m=m, Id=Id, Ip=Ip, tag=tag, color=color)

        self.is_random = is_random
        self.attribute_dict = attribute_dict

    def __iter__(self):
        """Return an iterator for the container.

        Returns
        -------
        An iterator over random disk elements.

        Examples
        --------
        >>> import ross.stochastic as srs
        >>> elm = srs.st_disk_example()
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
        >>> elms = srs.ST_DiskElement(n=1,
        ...                           m=30.0,
        ...                           Id=np.random.uniform(0.20, 0.40, 5),
        ...                           Ip=np.random.uniform(0.15, 0.25, 5),
        ...                           is_random=["Id", "Ip"],
        ...                           )
        >>> elms["m"]
        30.0
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
            ***check the correct type for each key in ST_DiskElement
            docstring.

        Raises
        ------
        KeyError
            Raises an error if the parameter doesn't belong to the class.

        Example
        -------
        >>> import numpy as np
        >>> import ross.stochastic as srs
        >>> elms = srs.ST_DiskElement(n=1,
        ...                           m=30.0,
        ...                           Id=np.random.uniform(0.20, 0.40, 5),
        ...                           Ip=np.random.uniform(0.15, 0.25, 5),
        ...                           is_random=["Id", "Ip"],
        ...                           )
        >>> elms["Id"] = np.linspace(0.1, 0.3, 5)
        >>> elms["Id"]
        array([0.1 , 0.15, 0.2 , 0.25, 0.3 ])
        """
        if key not in self.attribute_dict.keys():
            raise KeyError("Object does not have parameter: {}.".format(key))
        self.attribute_dict[key] = value

    def random_var(self, is_random, *args):
        """Generate a list of objects as random attributes.

        This function creates a list of objects with random values for selected
        attributes from ross.DiskElement.

        Parameters
        ----------
        is_random : list
            List of the object attributes to become stochastic.
        *args : dict
            Dictionary instanciating the ross.DiskElement class.
            The attributes that are supposed to be stochastic should be
            set as lists of random variables.

        Returns
        -------
        f_list : generator
            Generator of random objects.
        """
        args_dict = args[0]
        new_args = []
        for i in range(len(self.is_random)):
            arg = []
            for key, param in args_dict.items():
                if key in is_random:
                    arg.append(param.value(1)[0])
                else:
                    arg.append(param)
            new_args.append(arg)
        f_list = (DiskElement(*arg) for arg in new_args)

        return f_list