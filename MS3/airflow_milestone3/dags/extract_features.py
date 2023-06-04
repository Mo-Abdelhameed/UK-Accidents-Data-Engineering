import pandas as pd
from convertbng.util import convert_osgb36_to_lonlat

def extract(path):
    df = pd.read_csv(path)
    easting = df['location_easting_osgr'].tolist()
    northing = df['location_northing_osgr'].tolist()
    long, lat = convert_osgb36_to_lonlat(easting, northing)
    df['longitude'] = long
    df['latitude'] = lat
    normal = (df['weather_conditions_Normal'] == 1) & (df['light_conditions_Normal'] == 1) & \
             (df['road_surface_conditions_Normal'] == 1) & (df['carriageway_hazards_Normal'] == 1) & \
             (df['junction_detail_Normal'] == 1)

    df['normal_conditions'] = normal.astype(int)
    df.to_csv('/opt/airflow/data/integrated_csv.csv')
