"""Point Mass module for IDENTIFY ROSS.

This module creates an instance of random point mass for stochastic analysis.
"""

import numpy as np
from ross.units import check_units
import ross as rs

__all__ = ["ST_ID_PointMass"]


class ST_ID_PointMass:
    """A point mass element.

    This class will create a stochastic point mass element.
    This element can be used to link other elements in the analysis.
    The mass provided to the element can be different on the x and y direction
    (e.g. different support inertia for x and y directions).

    Parameters
    ----------
    n: int
        Node which the bearing will be located in.
    m: float, pint.Quantity, optional
        Mass for the element.
    mx: float, pint.Quantity, optional
        Mass for the element on the x direction.
    my: float, pint.Quantity, optional
        Mass for the element on the y direction.
    tag: str
        A tag to name the element
    color : str, optional
        A color to be used when the element is represented.
        Default is "DarkSalmon".
    to_identify : list
        List of the object attributes to become stochastic.
        Possibilities:
            ["m", "mx", "my"]

    Examples
    --------
    >>> p0 = ID_PointMass(n=0, m=[2,2.5], to_identify = ["m"])
    >>> p0["m"]
    >>> 2.315340084073693
    """

    def __init__(
        self,
        n = None,
        m = None,
        mx = None,
        my = None,
        tag = None,
        color = 'DarkSalmon',
        to_identify = None,
        erro = None
    ):
        attribute_dict = dict(
            n = n,
            m = m,
            mx = mx,
            my = my,
            tag = tag,
            color = color
        )

        self.attribute_dict = attribute_dict
        self.to_identify = to_identify
        self.interval_params = {}
        self.param_values = {}

        if self.to_identify:
                
            for params in self.to_identify:
                original_value = self.attribute_dict[params]
                
                if erro is None:
                    if not isinstance(original_value, (list, np.ndarray)):
                        raise KeyError(f'Parameter {params} must be a list/ndarray when erro is None')
                    
                    interval = original_value

                else:
                    if isinstance(original_value, (list, np.ndarray)):
                        raise KeyError(f'Parameter {params} should not be a list/ndarray when erro is informed')
                    
                    interval = [original_value * (1 - erro), original_value * (1 + erro)]
                
                self.interval_params[params] = interval
                new_value = np.random.uniform(interval[0], interval[1])
                self.param_values[params] = new_value
                self.attribute_dict[params] = new_value

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
        >>> p0 = ID_PointMass(n=0, m=[2,2.5], to_identify = ["m"])
        >>> p0["n"]
        >>> 0
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
                ***check the correct type for each key in ID_PointMass
                docstring.

            Raises
            ------
            KeyError
                Raises an error if the parameter doesn't belong to the class.

            Example
            -------
            >>> p0 = ID_PointMass(n=0, m=[2,2.5], to_identify = ["m"])
            >>> p0["n"] = 1
            >>> p0["n"]
            >>> 1
            """

            if key not in self.attribute_dict.keys():
                raise KeyError("Object does not have parameter: {}.".format(key))
            self.attribute_dict[key] = value        
            if key in self.param_values:
                self.param_values[key] = value

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(n={self.attribute_dict['n']}, mx={self.attribute_dict['mx']:{0}.{5}},"
            f" my={self.attribute_dict['my']:{0}.{5}}, tag={self.attribute_dict['tag']!r})"
        )

    def generator(self):

        args = []
        
        for value in self.attribute_dict.values():
            args.append(value)

        new_args = [args]

        f_list = (rs.PointMass(*arg) for arg in new_args)

        return f_list
    
    def new_PointMass(self):

        for params, value in self.param_values.items():
            self.attribute_dict[params] = value

        return list(iter(self.generator()))[0]