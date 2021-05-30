# PopulationSpatialization


## Introduction

## Data


Data contains the pixel-level attribute statistics required for regression modeling. Experimental area (Wuhan) is divided into pixels with 500m (308 x 250), 200m (770 x 625) and 100m (1540 x 1250) resolution  respectively. In each gridxxx.csv, the fields used are explained in the following table.


|  Field   | Description  |  Field   | Description  |
|  ----  | ----  |  ----  | ----  |
| count_poi1  | Leisure and entertainment | count_poi2  | Accommodation |
| count_poi3  | Parking lot | count_poi7  | Medical service |
| count_poi8  | Hospital | count_poi11  | Residential community |
| count_poi14  | Government agency | count_poi18  | Auto service |
| count_poi21  | Research and education | count_poi22  | Shopping |
| count_poi23  | Financial services | count_poi24  | Restaurant |
| building_area  | area of building patch data | mobile_night  | counts of mobile positioning data |
| sub_id  | id of the street that pixel belongs to | county_id  | id of the district that pixel belongs to |


## Code

#### Implementation Environment
- Python 3.x
- You need to import numpy, pandas and sklearn

#### How To Run The Code

In populationSpatialization.py, you need to set RESOLUTION, N_ROW and N_COLUMN first for choosing pixel resolution, and then you can run populationSpatialization.py to implent population estimation.

## Result
Result contains the predicted population of three resolutions (500m, 200m and 100m).The id and the predicted population of the pixel are listed in the file.