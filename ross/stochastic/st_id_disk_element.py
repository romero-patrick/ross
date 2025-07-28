"""Disk element module for IDENTIFY ROSS.

This module creates an instance of random disk for stochastic analysis.
"""

import numpy as np
from ross.units import check_units
import ross as rs
from ross import DiskElement

__all__ = ["ST_ID_DiskElement", "ST_ID_Disk_FromGeometry"]

class ST_ID_DiskElement(DiskElement):
    """A disk element.

    This class creates a stochastic disk element from input data of inertia and mass.

    Parameters
    ----------
    n: int
        Node in which the disk will be inserted.
    m : float, pint.Quantity
        Mass of the disk element.
    Id : float, pint.Quantity
        Diametral moment of inertia.
    Ip : float, pint.Quantity
        Polar moment of inertia
    tag : str, optional
        A tag to name the element
        Default is None
    scale_factor: float or str, optional
        The scale factor is used to scale the disk drawing.
        For disks it is also possible to provide 'mass' as the scale factor.
        In this case the code will calculate scale factors for each disk based
        on the disk with the higher mass. Notice that in this case you have to
        create all disks with the scale_factor='mass'.
        Default is 1.
    color : str, optional
        A color to be used when the element is represented.
        Default is 'Firebrick'.
    to_identify : list
        List of the object attributes to become stochastic.
        Possibilities:
            ["m", "Id", "Ip"]

    Examples
    --------
    >>> disk = ID_DiskElement(n=0, m=32, Id=[0.2,0.22], Ip=0.3, to_identify = ["Id"])
    >>> disk["Ip"]
    >>> 0.3
    """

    @check_units
    def __init__(
        self,
        n,
        m,
        Id,
        Ip,
        tag = None,
        scale_factor = 1.0,
        color = 'Firebrick',
        to_identify = None
    ):
        attribute_dict = dict(
            n = n,
            m = m,
            Id = Id,
            Ip = Ip,
            tag = tag,
            scale_factor = scale_factor,
            color = color
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

        super().__init__(attribute_dict['n'],
                         attribute_dict['m'],
                         attribute_dict['Id'],
                         attribute_dict['Ip'],
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
            >>> disk = ID_DiskElement(n=0, m=32, Id=[0.2,0.22], Ip=0.3, to_identify = ["Id"])
            >>> disk["Id"]
            >>> 0.20279638577734319
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
                ***check the correct type for each key in ID_DiskElement
                docstring.

            Raises
            ------
            KeyError
                Raises an error if the parameter doesn't belong to the class.

            Example
            -------
            >>> disk = ID_DiskElement(n=0, m=32, Id=[0.2,0.22], Ip=0.3, to_identify = ["Id"])
            >>> disk["Ip"] = 0.4
            >>> 0.4
            """

            if key not in self.attribute_dict.keys():
                raise KeyError("Object does not have parameter: {}.".format(key))
            self.attribute_dict[key] = value        
            if key in self.param_values:
                self.param_values[key] = value

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(Id={self.attribute_dict['Id']:{0}.{5}}, Ip={self.attribute_dict['Ip']:{0}.{5}}, "
            f"m={self.attribute_dict['m']:{0}.{5}}, color={self.attribute_dict['color']!r}, "
            f"n={self.attribute_dict['n']}, scale_factor={self.attribute_dict['scale_factor']}, tag={self.attribute_dict['tag']!r})"
        )

    def generator(self):

        args = []
        
        for value in self.attribute_dict.values():
            args.append(value)

        new_args = [args]
        
        f_list = (rs.DiskElement(*arg) for arg in new_args)

        return f_list
    
    def new_disk(self):

        for params, value in self.param_values.items():
            self.attribute_dict[params] = value

        return list(iter(self.generator()))[0]