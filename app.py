import os
import dash
from dash import dcc, html
import plotly.express as px
from dash.dependencies import Input, Output
import pandas as pd

# App settings
CSV_FILE = '1M_points.csv'
PLOT_DIMS = 2

# Read in the data
df = pd.read_csv(f"./embeddings_data/{CSV_FILE}")

# Check if the num rows > 500k, assert to many rows for 3d
if df.shape[0] > 500000:
    assert PLOT_DIMS == 2, "Too many rows for 3D plot. Set PLOT_DIMS to 2."

# Create a Plotly 3D scatter plot with color coding by class label
if PLOT_DIMS == 2:
    fig = px.scatter(df, x='x', y='y', color='label', hover_data=['image_path'], opacity=0.5, render_mode='webgl')
elif PLOT_DIMS == 3:
    fig = px.scatter_3d(df, x='x', y='y', z='z', color='label', hover_data=['image_path'])
else:
    raise ValueError("Invalid number of dimensions. Choose 2 or 3.")

# Set up Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1(f"Image Embeddings Visualization w/ {df.shape[0]} Images"),
    html.P("Click on a point in the scatter plot to display the image and its class label. Click on the legend to toggle classes."),
    html.Div(className='container', children=[
        dcc.Graph(id='scatter-plot', figure=fig),
        html.Div(id='image-display', children=[
            html.Img(id='selected-image', src=''),
            html.P(id='selected-label')
        ])
    ])   
])

@app.callback(
    [Output('selected-image', 'src'),
     Output('selected-label', 'children')],
    Input('scatter-plot', 'clickData')
)
def display_image_and_label(clickData):
    if clickData is None:
        return 'https://placedog.net/640/224?random', 'Wild Doge appears!'
    # Get the index of the clicked point
    image_url = clickData['points'][0]['customdata'][0]
    # Get the corresponding image path and label
    image_path = image_url.replace('./assets/', '')
    label = os.path.basename(os.path.dirname(image_url))
    return app.get_asset_url(image_path), f"Class: {label}"

if __name__ == '__main__':
    app.run_server(debug=True)