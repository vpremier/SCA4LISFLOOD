# SCA4LISFLOOD

These scripts are used to evaluate the snow module of LISFLOOD and derive a snowmelt coefficient based on Earth Observation (EO) data. In detail, the code:
- reproduces the LISFLOOD snow module; 
- implements two snow cover fraction parametrizations (Zaitchik and Rodell, 2009 and Swenson et Lawrence, 2012) to convert the Snow Water Equivalent (SWE) in Snow Cover Fraction (SCF);
- implements two methods to derive the Snowmelt Coefficient (Cm) - the first based on the number of snow covered days (Pistocchi et al., 2017) and the second based on an optimization that minimizes the error between the modelled and observed SCF.
