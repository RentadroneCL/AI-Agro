# AI Agro (Precision Agriculture with Drones)

![GitHub](https://img.shields.io/github/license/RentadroneCL/Precision_Agriculture)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg)](CODE_OF_CONDUCT.md)
[![Open Source Helpers](https://www.codetriage.com/rentadronecl/precision_agriculture/badges/users.svg)](https://www.codetriage.com/rentadronecl/precision_agriculture)

[Project Documentation](https://rentadronecl.github.io/docs/precision_agriculture)

## Forum

This project is part of the [UNICEF Innovation Fund Discourse community](https://unicef-if.discourse.group/c/projects/rentadrone/10). You can post comments or questions about each category of [Rentadrone Developers](https://rentadrone.cl/developers/) algorithms. We encourage users to participate in the forum and to engage with fellow users.

Model Precision_Agriculture

## Summary of the solution

Remote sensing has as one of its objectives, to be able to provide useful information in the shortest possible time for decision-making. Therefore, it is considered a fundamental tool in precision agriculture, since it allows the monitoring of crops throughout the growing season, providing timely information as a diagnostic evaluation. This task must identify the factor that operates in a restrictive manner and decide, in a timely manner, on corrective agronomic intervention.

A promising approach to this is one that integrates data derived from temporal, mosaic, multispectral, and thermal imaging. Both processes allow us to obtain products such as: Thermal maps and Normalized vegetation index maps; These products allow us to identify stress zones which serve as support in agricultural management tasks.

That is why our objective is to develop an Open Source platform, distributed on a GitHub platform, that is capable of generating local calculations and mapping (plant by plant) of most important  vegetation indices, through the processing of images taken with UAV.

Key words: Vegetation index, phenological status, agricultural management, Open Source platform.

## Software Features (To Do List):

- [x] Open Source, distributed on GitHub platform
- [x] Able to map the state of health in different types of crops visible in multispectral photographs taken with drones, allowing the calculation of the main types of vegetation indices (NVDI, GNDVI, NDRE, LCI, OSAVI, etc.)
- [x] At a minimum, it must be able to process JPG and TIFF (Multispectral Radiometric) images.
- [x] Possibility of generating multispectral orthomosaics for each band and for each vegetation index. In addition, it must be possible to extract pixel intensity values in case calculations of geolocated variables are required.
- [x] Be able to perform batch processes with batches of photo files.
- [ ] Online and Local Multiplatform Operation.
- [ ] Generate KMZ maps, using the GPS information in metadata of the photos
- [ ] Have a module for generating statistical reports regarding the number and types of problems found in photographs.

##  Multispectral band wavelengths available.

Today the sensors of the cameras on board UAV can capture spectral images in the wavelengths of red, red- edge, near infrared and thermal (Table Nº1).

Table Nº1: Multispectral bands

--
| Band | Wavelength |
| -- | -- |
Blue | 450 nm
Green | 560 nm
Red | 650 nm
Red Edge | 730 nm
Near infrared | 840 nm

## **Vegetation index calculations**

The following spectral index can be generated from these lengths (Table Nº2).

**Table Nº2:** Spectral index generated from the available wavelengths of camera on board UAV.

| Índex | Equation |
| -- | -- |
Normalised Difference Index | NDVI = ( Rnir- Rr)/(Rnir+Rr)
Green Normalized Difference Vegetation Index | GNDVI = (Rnir - Rgreen)/(Rnir + Rgreen)
Normalised Difference Red Edge | NDRE = (Rnir - Red edge)/ (Red edge + NIR)
Leaf Chlorophyll Index | LCI = (Rnir - Red edge)/(Rnir + Red)
Optimized Soil Adjusted Vegetation Index | OSAVI = (Nir-Red)/(Nir+Red+0.16)

## **Defining plant health status labels**

| | NDVI 1 | NDVI 1 < NDVI 2 |
| -- | -- |--|
Rank | Description | Description
-1 to 0 | Water, Bare Soils | Water, Bare Soils
0 to 0,15 | Soils with sparse, sparse vegetation or crops in the initial stage of development (sprouting) | Poor vigor, weak plants
0,15 to 0,30 | Plants in intermediate stage of development (leaf production) | Bad leaf / flower ratio
0,30 to 0,45 | Plants in intermediate stage of development (leaf production) | Bad flower / fruit ratio; fruits with low sugar content, lack of color in the fruits, fruits of low caliber
0,45 to 0,60 | Plants in the adult stage or phase (fruit production) | Bad flower / fruit ratio; fruits with low sugar content, lack of color in the fruits, fruits of low caliber
0,60 to >0,80 | Plants in the adult stage or stage (Fruit maturity) | Bad flower / fruit ratio; fruits with low sugar content, lack of color in the fruits, fruits of low caliber

## **Limitations of this solution**

1. The multispectral orthomosaics have to be built before using the tool
2. Process the RGB and multispectral bands separately
3. Must know the format of the bands you will be using and the metadata of each image (tiff, GeoTiff)
4. At this moment, the methodology and support is only for Phantom 4 RTK Multispectral user.


## METHODOLOGY

To complete the main objective we consider following diagram methodology (Image Nº1). Was proposed, which reflects the process of generating the information necessary for decision- making during the management of a production cycles of a crop in general.

![Process_Diag](https://github.com/RentadroneCL/AI-Agro/edit/master/Process_Diag.jpg)



Several diagrams of sub- processes were also proposed.
1- To assess the growth status of plants. Image Nº2,  NDVI multi-time series.


![Diagrama2](https://github.com/RentadroneCL/AI-Agro/edit/master/Diagrama2.jpg)


Minimum Viable Product (MVD) for NDVI

![MPV-NDVI](https://github.com/RentadroneCL/AI-Agro/edit/master/MPV-NDVI.jpg)

# Contributing

Contributions are welcome and will be fully credited. We accept contributions via Pull Requests on GitHub.

## Pull Request Checklist

Before sending your pull requests, make sure you followed this list.

- Read [contributing guidelines](CONTRIBUTING.md).
- Read [Code of Conduct](CODE_OF_CONDUCT.md).
- Check if my changes are consistent with the [guidelines](https://github.com/RentadroneCL/Photovoltaic_Fault_Detector/blob/master/CONTRIBUTING.md#general-guidelines-and-philosophy-for-contribution).
- Changes are consistent with the [Coding Style](https://github.com/RentadroneCL/Photovoltaic_Fault_Detector/blob/master/CONTRIBUTING.md#c-coding-style).
