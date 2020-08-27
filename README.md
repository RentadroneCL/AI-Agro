# Precision Agriculture

Model Precision_Agriculture
## SUMMARY

Remote sensing has as one of its objectives, to be able to provide useful information in the shortest possible time for decision-making. Therefore, it is considered a fundamental tool in precision agriculture, since it allows the monitoring of crops throughout the growing season, providing timely information as a diagnostic evaluation. This task must identify the factor that operates in a restrictive manner and decide, in a timely manner, on corrective agronomic intervention.

A promising approach to this is one that integrates data derived from temporal, mosaic, multispectral, and thermal imaging. Both processes allow us to obtain products such as: Thermal maps and Normalized vegetation index maps; These products allow us to identify stress zones which serve as support in agricultural management tasks.

That is why our objective is to develop an Open Source platform, distributed on a GitHub platform that is capable of generating diagnostic tools or early warning of water stress, health status (attack by pests or diseases), phenological status, nutritional deficiencies, productivity. and performance, among others; by capturing the variations in the reflectivity of the plants during the different growth stages, through the processing of images taken with UAV.

Key words: Vegetation index, phenological status, agricultural management, Open Source platform.

## INTRODUCTION

Among the biophysical parameters, the most important ones that can be determined through the use of vegetation index are: ***the chlorophyll content of the leaves (Chl), the leaf area index (LAI) and the Humidity.***

***The chlorophyll content (Chl)*** is an indicator of the ability of the vegetation to carry out photosynthesis, a basic process in the growth and survival of plants and is also directly related to the potencial of plants for the absorption of atmospheric CO2.  

***The leaf area index (LAI)***, its defined as the area of one side of leaves per unit area of soil, provides information on the plant canopy and is a basic parameter for climatic, agronomic and ecological studies.

***The Humidity***, is others bio-indicator of physiological stage of the  plant (health condition and phenology). Plants need a certain amount of moisture to carry out transpiration and other processes.  The transpiration is a processes in which the expel water into the atmosphere through microscopic leaf opening called stomata. As the plant grows, two phenomena occur; turgor and plasmolysis. Turgor is the phenomenon by which cells swell or fill with water and plasmolysis is the opposite process, cells naturally lose water as they wilt. In many cases the response of the vegetation to external aggression such as a disease or to situations of water stress is to increase it temperature.   New remote sensing methods based on high- resolution thermal images have demonstrated their  potential for detecting water stress and estimating photosynthetic performance through detection of fluorescence and chlorophyll activity emitted by vegetation.  

Today the sensors of the cameras on board UAV of capturing spectral in the wavelengths of red, red- edge, near infrared and thermal (Table Nº1).

**Table Nº1:** Multispectral band wavelengths available. 

Multispectral bands
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
Enhanced Vegetation Index | EVI = 2.5( Rn- Rr)/ (Rn+ 6*Rr)-(7.5*Rb + 1)*
Leaf area index | LAI= (3.618 x EVI – 0.118) > 0*
Normalized Difference Water Index | NDWI = (Rnir - Swir) / (Rnir + Swir)*

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
2. Charge the RGB bands separately
3. Must know the format of the bands you will be using and the metadata of each image (tiff, GeoTiff)
4. In this case the methodology and support only will be for Phantom 4 RTK Multispectral user.


## **TO DO LIST**

For the deveploment this work we follow the following checklist:
    
**- Mosaico vegetation index**
 
 1- NDVI
 - Vineyards
 - Musacles 
 - Other crops

2- GNDVI
 - Vineyards
 - Musacles 
 - Other crops

3- NDRE
 - Vineyards
 - Musacles 
 - Other crops

4- LCI
 - Vineyards
 - Musacles 
 - Other crops

5- OSAVI
 - Vineyards
 - Musacles 
 - Other crops

**-Labelin**

1- Weed detection
 - Vineyards
 - Musacles 
 - Other crops

2- Disease detection
 - Vineyards
 - Musacles 
 - Other crops
 
## METHODOLOGY

To complete the main objective we consider following diagram methodology (Image Nº1). Was proposed, which reflects the process of generating the information necessary for decision- making during the management of a production cycles of a crop in general. 

![Process_Diag](https://github.com/RentadroneCL/Precision_Agriculture/blob/master/Process_Diag.jpg)



Several diagrams of sub- processes were also proposed.
1- To assess the growth status of plants. Image Nº2,  NDVI multi-time series. 


![Diagrama2](https://github.com/RentadroneCL/Precision_Agriculture/blob/master/Diagrama2.jpg)


Minimum Viable Product (MVD) for NDVI

![MPV-NDVI](https://github.com/RentadroneCL/Precision_Agriculture/blob/master/MPV-NDVI.jpg)

