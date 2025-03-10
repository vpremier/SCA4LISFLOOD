# SCA4LISFLOOD: Snow Module Evaluation for LISFLOOD

This repository contains scripts used to evaluate the snow module of LISFLOOD and derive a snowmelt coefficient based on Earth Observation (EO) data. Specifically, the code:

- Reproduces the LISFLOOD snow module (see [LISFLOOD GitHub](https://github.com/ec-jrc/lisflood-code)).
- Implements two snow cover fraction parameterizations (Zaitchik and Rodell, 2009, and Swenson & Lawrence, 2012) to convert Snow Water Equivalent (SWE) into Snow Cover Fraction (SCF).
- Implements two methods to derive the Snowmelt Coefficient (Cm):  
    - The first method is based on the number of snow-covered days (Pistocchi et al., 2017).
    - The second method is an optimization approach that minimizes the error between modeled and observed SCF.

### Main Scripts

- **`main.py`**: Used to reproduce the SWE and SCF.
- **`optimization.py`**: Used for deriving the snowmelt coefficient through the optimization approach.

### Folder Structure

- **`Arve/`**: Contains input and output data for running the example.  
For other basins, refer to the following links:
    - LISFLOOD parameters: [LISFLOOD Static and Parameter Maps for EFAS](https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/CEMS-EFAS/LISFLOOD_static_and_parameter_maps_for_EFAS/)
    - Meteorological forcings: [CEMS-EFAS Meteorological Forcings](https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/CEMS-EFAS/meteorological_forcings/)
    - Snow cover fraction time-series: [Zenodo Snow Cover Fraction Data](https://zenodo.org/records/14961639)


### Contact Information

For any questions or inquiries, please contact:

- **Name**: Valentina Premier
- **Email**: valentina.premier@eurac.edu
- **Organization**: Eurac Research

- **Name**: Francesca Moschini
- **Email**: francesca.moschini@ec.europa.eu
- **Organization**: European Commission, Joint Research Centre (JRC)
