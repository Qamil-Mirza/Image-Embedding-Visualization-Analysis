import os
import dash
from dash import dcc, html
import plotly.express as px
from dash.dependencies import Input, Output
import pandas as pd

# Read in the data
df = pd.read_csv('embedding_image_df.csv')

# Create a Plotly 3D scatter plot with color coding by class label
fig = px.scatter_3d(df, x='x', y='y', z='z', color='label', hover_data=['image_path'])

# Set up Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
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
    point_index = clickData['points'][0]['pointNumber']
    # Get the corresponding image path and label
    image_path = df.iloc[point_index]['image_path'].replace('./assets/', '')
    label = df.iloc[point_index]['label']
    return app.get_asset_url(image_path), f"Class: {label}"

if __name__ == '__main__':
    app.run_server(debug=True)