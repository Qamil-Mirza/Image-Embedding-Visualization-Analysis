import os
import dash
from dash import dcc, html
import plotly.express as px
from dash.dependencies import Input, Output
import pandas as pd

# App settings
CSV_FILE = 'main.csv'
PLOT_DIMS = 2

# Read in the data
df = pd.read_csv(f"./embeddings_data/{CSV_FILE}")
PLOT_TITLE = f"Image Embeddings Visualization: {df.shape[0]} Images - {PLOT_DIMS}D"

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

fig.update_layout(title_text=PLOT_TITLE, title_x=0.5)

# Set up Dash app layout
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("SUPA Embeddings Visualizer"),
    html.P("Click on a point to see the image and class label or use the lasso/box-select tool to select multiple points."),
    html.P("Click on the legend to toggle classes on/off."),
    dcc.Graph(id='scatter-plot', figure=fig),
    html.H4("Selected Points"),
    html.Div(id='select-data')   
])

# Combined Callback

@app.callback(
    Output('select-data', 'children'),
    [Input('scatter-plot', 'clickData'),
     Input('scatter-plot', 'selectedData')]
)
def display_images(clickData, selectedData):
    items = []

    # Handle clickData for a single point
    if clickData:
        items = []
        image_url = clickData['points'][0]['customdata'][0]
        image_path = image_url.replace('./assets/', '')
        label = os.path.basename(os.path.dirname(image_url))
        items.append(
            html.Div([
                html.Img(src=app.get_asset_url(image_path), style={'height': '150px', 'margin': '5px'}),
                html.P(f"Class: {label}", style={'text-align': 'center'})
            ], style={'display': 'inline-block', 'margin': '10px'})
        )

    # Handle selectedData for multiple points
    if selectedData:
        items = []
        for point in selectedData['points']:
            image_url = point['customdata'][0]
            image_path = image_url.replace('./assets/', '')
            label = os.path.basename(os.path.dirname(image_url))
            items.append(
                html.Div([
                    html.Img(src=app.get_asset_url(image_path), style={'height': '150px', 'margin': '5px'}),
                    html.P(f"Class: {label}", style={'text-align': 'center'})
                ], style={'display': 'inline-block', 'margin': '10px'})
            )
    
    # Handle the case when no points are selected or clicked
    if not items:
        items = [html.Div([
            html.Img(src='https://placedog.net/640/224?random'),
            html.P('No Points Selected. A wild Doge appears!')
        ])]

    return items

if __name__ == '__main__':
    app.run_server(debug=True)
