from dash import dcc, html, Dash
from graphs import accidents_per_weekday, accidents_density, timing_graph, sunburst, severity_plot

def create_dashboard(filename):
    
    bottom_graphs = [
        dcc.Graph(figure=accidents_per_weekday(filename), className='graph'),
        dcc.Graph(figure=timing_graph(filename), className='graph'),
        dcc.Graph(figure=sunburst(filename), className='graph'),
        dcc.Graph(figure=severity_plot(filename), className='graph')
    ]    

    main_div = html.Div(
    children=[
        html.Div(children=[html.H2('Density Map', className='center'), dcc.Graph(figure=accidents_density(filename), className='graph'),], ),
        html.Div(children=bottom_graphs, className='bottom')
    ], 
    className='main',
)
    app = Dash(__name__)
    app.layout = html.Div(
        children=[ html.H1(children="UK 1981 Accidents dataset", className='center'), main_div]
    )
    
    app.run(host='0.0.0.0', port=8050)
