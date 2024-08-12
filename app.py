import os
import dash
from dash import dcc, html, callback_context
import plotly.express as px
from dash.dependencies import Input, Output, State
import pandas as pd
import math

# App settings
CSV_FILE = 'main.csv'
PLOT_DIMS = 2
IMAGES_PER_PAGE = 12
INITIAL_PAGE_NUM = 1

# Initial setup
page_num = INITIAL_PAGE_NUM

df = pd.read_csv(f"./embeddings_data/{CSV_FILE}")
PLOT_TITLE = f"Image Embeddings Visualization: {df.shape[0]} Images - {PLOT_DIMS}D"

if df.shape[0] > 500000:
    assert PLOT_DIMS == 2, "Too many rows for 3D plot. Set PLOT_DIMS to 2."

if PLOT_DIMS == 2:
    fig = px.scatter(df, x='x', y='y', color='label', hover_data=['image_path'], opacity=0.5, render_mode='webgl')
elif PLOT_DIMS == 3:
    fig = px.scatter_3d(df, x='x', y='y', z='z', color='label', hover_data=['image_path'])
else:
    raise ValueError("Invalid number of dimensions. Choose 2 or 3.")

fig.update_layout(title_text=PLOT_TITLE, title_x=0.5, clickmode='event+select')

# Set up Dash app layout
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("SUPA Embeddings Visualizer"),
    html.P("Click on a point to see the image and class label or use the lasso/box-select tool to select multiple points."),
    html.P("Click on the legend to toggle classes on/off. Hold down shift while clicking on points to cherry pick multiple points"),
    dcc.Graph(id='scatter-plot', figure=fig),
    html.H4("Selected Points"),
    html.Div(id='select-data'),
    html.Div([
        html.Button('<', id='decrement-button', n_clicks=0),
        html.P(id='page-num-display', children=f'{page_num}'),
        html.Button('>', id='increment-button', n_clicks=0),
    ], id='pagination'),
    html.Div(id='hidden-page-num', style={'display': 'none'}, children=f'{page_num}')
])

@app.callback(
    Output('select-data', 'children'),
    [Input('scatter-plot', 'clickData'),
     Input('scatter-plot', 'selectedData'),
     Input('hidden-page-num', 'children')]
)
def display_images(clickData, selectedData, page_num):
    items = []
    page_num = int(page_num)

    start_index = (page_num - 1) * IMAGES_PER_PAGE
    end_index = start_index + IMAGES_PER_PAGE

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
        for point in selectedData['points'][start_index:end_index]:
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

# Pagination
@app.callback(
    [Output('page-num-display', 'children'),
     Output('hidden-page-num', 'children')],
    [Input('increment-button', 'n_clicks'),
     Input('decrement-button', 'n_clicks'),
     Input('scatter-plot', 'relayoutData')],
    [State('hidden-page-num', 'children'),
     State('scatter-plot', 'selectedData')]
)
def update_page_num(increment_clicks, decrement_clicks, relayoutData, page_num, selectedData):
    page_num = int(page_num)
    ctx = callback_context

    if not ctx.triggered:
        return f'{page_num}', f'{page_num}'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Check if double-click occurred (reset page number to 1)
    if relayoutData and 'xaxis.range' in relayoutData and 'yaxis.range' in relayoutData:
        if 'autosize' in relayoutData:
            page_num = INITIAL_PAGE_NUM
        else:
            page_num = int(page_num)

    # Calculate the total number of pages
    total_items = len(selectedData['points']) if selectedData else 0
    total_pages = math.ceil(total_items / IMAGES_PER_PAGE)

    if button_id == 'increment-button' and increment_clicks and page_num < total_pages:
        page_num += 1
    elif button_id == 'decrement-button' and decrement_clicks and page_num > 1:
        page_num -= 1
    else:
        page_num = 1

    return f'{page_num}', f'{page_num}'

# Button Disable
@app.callback(
    [Output('decrement-button', 'disabled'),
     Output('increment-button', 'disabled')],
    [Input('hidden-page-num', 'children'),
     Input('scatter-plot', 'selectedData')],
    [State('scatter-plot', 'selectedData')]
)
def update_button_disabled(page_num, selectedData, stateSelectedData):
    selectedData = selectedData or stateSelectedData
    total_items = len(selectedData['points']) if selectedData else 0
    total_pages = math.ceil(total_items / IMAGES_PER_PAGE)
    page_num = int(page_num)
    
    if total_items <= 12:
        return True, True
    
    decrement_disabled = page_num <= 1
    increment_disabled = page_num >= total_pages

    return decrement_disabled, increment_disabled

if __name__ == '__main__':
    app.run_server(debug=True)
