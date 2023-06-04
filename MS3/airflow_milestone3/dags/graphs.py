import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def get_day_counts(df):
    days = ['Saturday', 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    days_counts = []
    for day in days:
        days_counts.append(df['day_of_week_{}'.format(day)].sum())
    data = {'days': days, 'days_counts': days_counts}
    return pd.DataFrame(data)
    
def get_timing_counts(df):
    timings = ['Dawn', 'Morning', 'Afternoon', 'Evening']
    timing_counts = []
    for timing in timings:
        timing_counts.append(df[timing].sum())
    data = {'timings': timings, 'timing_counts': timing_counts}
    return pd.DataFrame(data)

def get_severity_counts(df):
    severities = ['Slight', 'Serious', 'Fatal']
    severity_counts = df['accident_severity'].value_counts().sort_values(ascending=False).values.tolist()
    data = {'severities': severities, 'severity_counts': severity_counts}
    return pd.DataFrame(data)

def accidents_per_weekday(filename):
    df = pd.read_csv(filename)
    fig = px.bar(get_day_counts(df),
                y='days_counts', x='days', text_auto='.2s',
                title="Number of accidents per day of week",
                color='days_counts', labels={'days_counts': 'Number of Accidents', 'days': ''})
    fig.update_traces(textfont_size=12, textangle=0, textposition='inside', cliponaxis=False)
    return fig

def accidents_density(filename):
    df = pd.read_csv(filename)
    fig = px.density_mapbox(df, lat='latitude', lon='longitude', radius=10,opacity=0.9, 
                        center=dict(lat=51.5072, lon=0.1276), zoom=10,
                        mapbox_style="stamen-terrain", height=800)
    return fig

def timing_graph(filename):  
    df = pd.read_csv(filename)     
    fig = px.bar(get_timing_counts(df),
                y='timing_counts', x='timings', text_auto='.2s',
                title="Number of accidents per Timing",
                color='timing_counts', labels={'timing_counts': 'Number of Accidents', 'timings': 'Timings'})
    fig.update_traces(textfont_size=12, textangle=0, textposition='inside', cliponaxis=False)
    return fig

def sunburst(filename):
    df = pd.read_csv(filename) 
    junction_filter = df['junction_detail_Normal'] == 0
    junction = junction_filter.sum()

    weather_filter = (~junction_filter) & (df['weather_conditions_Normal'] == 0)
    weather = weather_filter.sum()

    light_filter = (~junction_filter) & (~weather_filter) & (df['light_conditions_Normal'] == 0)
    light = light_filter.sum()

    road_filter = (~junction_filter) & (~weather_filter) & (~light_filter) & (df['road_surface_conditions_Normal'] == 0)
    road = road_filter.sum()

    carriageway_filter = (~junction_filter) & (~weather_filter) & (~light_filter) & (~road_filter) & (df['carriageway_hazards_Normal'] == 0)
    carriageway = carriageway_filter.sum()

    normal_accidents_count = df['normal_conditions'].sum()
    non_normal_accidents_count = df.shape[0] - normal_accidents_count

    fig = go.Figure(go.Sunburst(
    labels=["Normal Conditions", "Non-Normal Conditions", "Junction", "Weather", 'Lights', 'Road', 'Carriageway'],
    parents=["", "", "Non-Normal Conditions", "Non-Normal Conditions", "Non-Normal Conditions", "Non-Normal Conditions", "Non-Normal Conditions"],
    values=[normal_accidents_count, non_normal_accidents_count, junction, weather, light, road, carriageway],
    textinfo='label+percent parent',
    branchvalues='total'
))
    fig.update_layout(margin = dict(t=50, l=0, r=0, b=0),  title={
                'text': 'Percentage of Normal vs Non-Normal Conditions and sub-categories',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'})


    return fig
    
def severity_plot(filename):
    df = pd.read_csv(filename)
    fig = px.bar(get_severity_counts(df),
                y='severities', x='severity_counts', text_auto='.2s',
                title="Number of Accidents per Accident Severity",
                color='severities', labels={'severity_counts': 'Number of Accidents', 'severities': 'Accident Severity'})
    fig.update_traces(textfont_size=12, textangle=0, textposition='outside', cliponaxis=False)
    return fig
