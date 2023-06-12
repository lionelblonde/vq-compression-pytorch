"""This file contains the metadata realated to the input data."""


BAND_NAMES = [
    "B01",  # coastal aerosol
    "B02",  # blue
    "B03",  # green
    "B04",  # red
    "B05",  # vegetation red edge
    "B06",  # vegetation red edge
    "B07",  # vegetation red edge
    "B08",  # NIR
    "B8A",  # narrow NIR
    "B09",  # water vapour
    # "B10",  # SWIR - cirrus  # in sentinel data but not in BigEarthNet dataset (!?)
    "B11",  # SWIR
    "B12",  # SWIR
]


RGB_BANDS_NAMES = [
    "B04",
    "B03",
    "B02",
]


BANDS_10M = [
    "B04",
    "B03",
    "B02",
    "B08",
]


BANDS_20M = [
    "B05",
    "B06",
    "B07",
    "B8A",
    "B11",
    "B12",
]


BANDS_60M = [
    "B01",
    "B09",
]


LABELS = [
    "Continuous urban fabric",
    "Discontinuous urban fabric",
    "Industrial or commercial units",
    "Road and rail networks and associated land",
    "Port areas",
    "Airports",
    "Mineral extraction sites",
    "Dump sites",
    "Construction sites",
    "Green urban areas",
    "Sport and leisure facilities",
    "Non-irrigated arable land",
    "Permanently irrigated land",
    "Rice fields",
    "Vineyards",
    "Fruit trees and berry plantations",
    "Olive groves",
    "Pastures",
    "Annual crops associated with permanent crops",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agro-forestry areas",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grassland",
    "Moors and heathland",
    "Sclerophyllous vegetation",
    "Transitional woodland/shrub",
    "Beaches, dunes, sands",
    "Bare rock",
    "Sparsely vegetated areas",
    "Burnt areas",
    "Inland marshes",
    "Peatbogs",
    "Salt marshes",
    "Salines",
    "Intertidal flats",
    "Water courses",
    "Water bodies",
    "Coastal lagoons",
    "Estuaries",
    "Sea and ocean",
]

# values provided by TUB, at the repository:
# https://git.tu-berlin.de/rsim/bigearthnet-models-tf/-/tree/master/models
BAND_STATS = {
    "mean": {
        "B02": 429.9430203,
        "B03": 614.21682446,
        "B04": 590.23569706,
        "B05": 950.68368468,
        "B06": 1792.46290469,
        "B07": 2075.46795189,
        "B08": 2218.94553375,
        "B8A": 2266.46036911,
        "B11": 1594.42694882,
        "B12": 1009.32729131,
        "VV": -12.619993741972035,
        "VH": -19.29044597721542,
    },
    "std": {
        "B02": 572.41639287,
        "B03": 582.87945694,
        "B04": 675.88746967,
        "B05": 729.89827633,
        "B06": 1096.01480586,
        "B07": 1273.45393088,
        "B08": 1365.45589904,
        "B8A": 1356.13789355,
        "B11": 1079.19066363,
        "B12": 818.86747235,
        "VV": 5.115911777546365,
        "VH": 5.464428464912864,
    },
}
