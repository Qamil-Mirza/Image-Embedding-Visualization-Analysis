import os
import dash
from dash import dcc, html
import plotly.express as px
from dash.dependencies import Input, Output
import pandas as pd

# Read in the data
CSV_FILE = 'main.csv'
df = pd.read_csv(f"./embeddings_data/{CSV_FILE}")

# Create a Plotly 3D scatter plot with color coding by class label
fig = px.scatter_3d(df, x='x', y='y', z='z', color='label', hover_data=['image_path'])

# Set up Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1(f"Image Embeddings Visualization w/ {df.shape[0]} Images"),
    dcc.Graph(id='scatter-plot', figure=fig),
    html.Div(id='image-display', children=[
        html.Img(id='selected-image', src='', style={'width': '300px', 'height': '300px'}),
        html.Div(id='selected-label', style={'margin-top': '10px', 'font-size': '20px'})
    ])
])

@app.callback(
    [Output('selected-image', 'src'),
     Output('selected-label', 'children')],
    Input('scatter-plot', 'clickData')
)
def display_image_and_label(clickData):
    if clickData is None:
        return '', ''
    # Get the index of the clicked point
    image_url = clickData['points'][0]['customdata'][0]
    # Get the corresponding image path and label
    image_path = image_url.replace('./assets/', '')
    label = os.path.basename(os.path.dirname(image_url))
    print(f"Selected image: {image_path}")
    return app.get_asset_url(image_path), f"Class: {label}"

if __name__ == '__main__':
    app.run_server(debug=True)