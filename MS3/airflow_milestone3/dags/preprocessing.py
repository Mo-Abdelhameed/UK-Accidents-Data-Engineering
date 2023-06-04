import pandas as pd
import numpy as np

csv_lookup = {'Column Name': [], 'Original value': [], 'Mapped value': []}

def read_data(path):
    return pd.read_csv(path)

def percentage_of_missing_values(df):
    columns_with_missing_values = df.columns[df.isna().any()]
    return df[columns_with_missing_values].isna().mean().sort_values(ascending=False) * 100

def append_lookup(col_name, original, new):
    csv_lookup['Column Name'].append(col_name)
    csv_lookup['Original value'].append(original)
    csv_lookup['Mapped value'].append(new)

def drop_less_than_one_percent(df):
    columns_with_missing_values = df.columns[df.isna().any()]
    columns_to_drop = df[columns_with_missing_values].isna().mean() * 100
    columns_to_drop = columns_to_drop[columns_to_drop < 1]
    l = columns_to_drop.index.tolist()
    condition = df[l[0]].isna()
    for i in range(1, len(l)):
        condition |= df[l[i]].isna()
    return df[~condition]
    

def fill_with_knn(df_training, df_fill, x, y):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier
    l = []
    l.append(y)
    y = l
    df_training = df_training[x + y]
    df_train = df_training.dropna()

    X = df_train[x]
    Y = df_train[y]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_transformed = scaler.transform(X_train)
    X_test_transformed = scaler.transform(X_test)

    knn = KNeighborsClassifier()
    knn.fit(X_train_transformed, Y_train)
    score = knn.score(X_test_transformed, Y_test) * 100

    scaler = StandardScaler()
    X_transformed = scaler.fit_transform(X)

    knn = KNeighborsClassifier()
    knn.fit(X_transformed, Y)
    
    rows_to_predict = df_fill[df_fill[y[0]].isna()]
    rows_to_predict_x = scaler.transform(rows_to_predict[x])
    rows_to_predict[y] = knn.predict(rows_to_predict_x).reshape(-1,1)

    df_fill[y] = rows_to_predict[y]

def fill_missing(df):
    data_urban_rural = pd.read_csv('/opt/airflow/data/1995_Accidents_UK.csv')
    data_trunk_road = data_urban_rural.copy()
    data_urban_rural = data_urban_rural[data_urban_rural['urban_or_rural_area'] != 'Unallocated']
    data_trunk_road = data_trunk_road[data_trunk_road['trunk_road_flag'] != 'Data missing or out of range']
    fill_with_knn(df_training=data_urban_rural, df_fill=df, x=['location_easting_osgr', 'location_northing_osgr'], y='urban_or_rural_area')
    fill_with_knn(df_training=data_trunk_road, df_fill=df, x=['location_easting_osgr', 'location_northing_osgr'], y='trunk_road_flag')
    return df

def fill_local_authority(df):
    is_digit = df['local_authority_district'].astype(str).str.isdigit()
    missing = df[is_digit]
    existing = df[~is_digit]
    missing['local_authority_district'] = np.nan
    fill_with_knn(df_training=existing, df_fill=missing, x=['location_easting_osgr', 'location_northing_osgr'], y='local_authority_district')
    return pd.concat([existing, missing])

def drop_more_than_50(df):
    perc = percentage_of_missing_values(df)
    columns_to_drop = df[perc[perc > 50].index.tolist()]
    return df.drop(columns_to_drop, axis='columns')

def fill_with_mode1(df):
    columns = percentage_of_missing_values(df).index.tolist()
    for i in columns:
        value_counts = df[i].value_counts()
        mode_percentage = (value_counts / value_counts.sum()).values[0]
        if mode_percentage >= 0.95:
            df.drop(i, axis='columns', inplace=True)
    return df

def impute_arbitrary(df):
    df['junction_control'].fillna('Missing', inplace=True)
    append_lookup('junction_control', np.nan, 'Missing')
    return df

def fill_with_mode2(df):
    columns = df.columns
    numeric_columns = df._get_numeric_data().columns
    categorical_columns = list(set(columns) - set(numeric_columns))
    missing_columns = df.columns[df.isna().any()]
    categorical_missing = list(set(categorical_columns).intersection(set(missing_columns)))
    condition = df[categorical_missing].isna().mean() * 100 < 50
    categorical_missing = condition[condition].index
    for i, col in enumerate(categorical_missing):
            df[col].fillna(df[col].mode()[0],inplace=True)
    return df

def drop_columns(df):
    return df.drop(['accident_index', 'accident_year', 'accident_reference', 'first_road_number'], axis='columns')

def drop_duplicated(df):
    duplicate = df.duplicated().sum()
    if(duplicate > 0):
        df = df.drop_duplicates()
    return df

def impute_data(df):
    df = drop_less_than_one_percent(df)
    df = fill_missing(df)
    df = fill_local_authority(df)
    df = drop_more_than_50(df)
    df = fill_with_mode1(df)
    df = impute_arbitrary(df)
    df = fill_with_mode2(df)
    df = drop_columns(df)
    df = drop_duplicated(df)
    return df

def discretize_light(df):
    mapp = {}
    unique_light_conditions = df['light_conditions'].unique()
    unique_light_conditions.tolist()
    for i in unique_light_conditions:
        if i == 'Daylight' or i == 'Darkness - lights lit':
            mapp[i] = 'Normal'
        else:
            mapp[i] = 'Non-normal'

    df['light_conditions'] = df['light_conditions'].map(mapp)
    return df

def discretize_road_surface(df):
    mapp = {}
    unique_light_conditions = df['road_surface_conditions'].unique()
    unique_light_conditions.tolist()
    for i in unique_light_conditions:
        if i == 'Dry':
            mapp[i] = 'Normal'
        else:
            mapp[i] = 'Non-normal'

    df['road_surface_conditions'] = df['road_surface_conditions'].map(mapp)
    return df

def discretize_carriageway(df):
    mapp = {}
    unique_light_conditions = df['carriageway_hazards'].unique()
    unique_light_conditions.tolist()
    for i in unique_light_conditions:
        if i == 'None':
            mapp[i] = 'Normal'
        else:
            mapp[i] = 'Non-normal'

    df['carriageway_hazards'] = df['carriageway_hazards'].map(mapp)
    return df

def discretize_pedestrian(df):
    mapp = {}
    unique_light_conditions = df['pedestrian_crossing_physical_facilities'].unique()
    unique_light_conditions.tolist()
    for i in unique_light_conditions:
        if i == 'No physical crossing facilities within 50 metres':
            mapp[i] = 0
        else:
            mapp[i] = 1

    df['pedestrian_crossing_physical_facilities'] = df['pedestrian_crossing_physical_facilities'].map(mapp)
    return df

def discretize_weather(df):
    mapp = {}
    unique_light_conditions = df['weather_conditions'].unique()
    unique_light_conditions.tolist()
    for i in unique_light_conditions:
        if i == 'Fine no high winds':
            mapp[i] = 'Normal'
        else:
            mapp[i] = 'Non-Normal'
    df['weather_conditions'] = df['weather_conditions'].map(mapp)
    return df

def discretize_junction_detail(df):
    mapp = {}
    unique_light_conditions = df['junction_detail'].unique()
    unique_light_conditions.tolist()
    for i in unique_light_conditions:
        if i == 'Not at junction or within 20 metres':
            mapp[i] = 'Normal'
        else:
            mapp[i] = 'Non-Normal'

    df['junction_detail'] = df['junction_detail'].map(mapp)
    return df

def discretize_junction_control(df):
    mapp = {}
    unique_light_conditions = df['junction_control'].unique()
    unique_light_conditions.tolist()
    for i in unique_light_conditions:
        if i == 'Give way or uncontrolled':
            mapp[i] = 'Uncontrolled'
        elif i == 'Missing':
            mapp[i] = 'Missing'
        else:
            mapp[i] = 'Controlled'

    df['junction_control'] = df['junction_control'].map(mapp)
    return df

def discretization(df):
    df['Week number'] = df['date'].dt.isocalendar().week
    df = df[df['speed_limit'] >= 30]
    df['speed_limit'] = round(df['speed_limit'], -1)
    df = discretize_light(df)
    df = discretize_road_surface(df)
    df = discretize_carriageway(df)
    df = discretize_pedestrian(df)
    df = discretize_weather(df)
    df = discretize_junction_detail(df)
    df = discretize_junction_control(df)
    return df, df.select_dtypes(include=np.number).columns.tolist()

def encoding(df):
    categorical_columns = df.select_dtypes(exclude=["number"]).columns.tolist()
    binary_columns = df[categorical_columns].nunique() == 2
    binary_columns = binary_columns[binary_columns].index.tolist()
    for col in binary_columns:
        encoding = pd.get_dummies(df[col], drop_first=True)
        df['{}_{}'.format(col, encoding.columns[0])] = encoding
        df.drop(col, axis='columns', inplace=True)
    from sklearn.preprocessing import LabelEncoder
    df['accident_severity'] = LabelEncoder().fit_transform(df['accident_severity'])
    append_lookup('accident_severity', 'Slight', 2)
    append_lookup('accident_severity', 'Serious', 1)
    append_lookup('accident_severity', 'Fatal', 0)

    categorical_columns = df.select_dtypes(exclude=["number"]).columns.tolist()
    s = df[categorical_columns].nunique()
    columns = s[(s < 10)].index.tolist()

    for col in columns:
        encoding = pd.get_dummies(df[col], prefix=col)
        df.drop(col, axis='columns', inplace=True)
        df = pd.concat([df, encoding], axis='columns')

    excluded = df.drop(['date', 'time'], axis=1)
    categorical_columns = excluded.select_dtypes(exclude=["number"]).columns.tolist()

    def calculate_top_categories(df, variable, how_many):
        return [
            x for x in df[variable].value_counts().sort_values(
                ascending=False).head(how_many).index
        ]

    def one_hot_encode(df, variable, top_x_labels):
        for label in top_x_labels:
            df[variable + '_' + label] = np.where(
                df[variable] == label, 1, 0) 

    for i in categorical_columns:
        top = calculate_top_categories(df, i, 10)
        one_hot_encode(df, i, top)
        df.drop(i, axis=1, inplace=True)
    return df
    
def add_columns(df):
    df['Weekend'] = (df['date'].dt.dayofweek > 4).astype(int)
    df['month'] = df['date'].dt.month
    df['datetime'] = df['date'].astype(str) + ' ' + df['time']
    df['datetime'] = pd.to_datetime(df['datetime'])
    mapp = {}
    for i in range(24):
        if 5 <= i <= 11:
            mapp[str(i)] = 'Morning'
        elif 12 <= i <= 18: 
            mapp[str(i)] = 'Afternoon'
        elif 19 <= i <= 24: 
            mapp[str(i)] = 'Evening'
        else:
            mapp[str(i)] = 'Dawn'
    df['time'] = (df['datetime'].dt.hour).astype(str).map(mapp)
    one_hot = pd.get_dummies(df['time'])
    df = pd.concat([df, one_hot], axis='columns')
    df.drop(['time', 'date', 'datetime'], inplace=True, axis=1)
    return df

def remove_outliers(df):
    from sklearn.neighbors import LocalOutlierFactor as LOF
    X = df.drop(['location_easting_osgr', 'location_northing_osgr'], axis=1)
    predictions =  LOF().fit_predict(X[['speed_limit', 'number_of_vehicles', 'number_of_casualties']])
    df['outlier'] = predictions
    df = df[df['outlier'] == 1]
    df.drop('outlier', axis=1, inplace=True)
    return df

def normalization(df, to_be_normalized):
    from sklearn.preprocessing import MinMaxScaler
    min_max_scaler = MinMaxScaler()
    x = df[to_be_normalized]
    x = x.loc[:, ~x.columns.isin(['location_northing_osgr', 'location_easting_osgr'])]
    df[x.columns] = min_max_scaler.fit_transform(x)
    return df


def preprocess_data(path):
    df = pd.read_csv(path, parse_dates=['date'], na_values=['Data missing or out of range', -1, '-1'])
    df = impute_data(df)
    df, to_be_normalized = discretization(df)
    df = encoding(df)
    df = add_columns(df)
    print('outliers Start')
    df = remove_outliers(df)
    print('outliers End')
    df = normalization(df, to_be_normalized)
    df.to_csv('/opt/airflow/data/preprocessed.csv', index=False)
    pd.DataFrame(csv_lookup).set_index('Column Name').to_csv('/opt/airflow/data/lookup.csv')
