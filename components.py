from dash import dcc, html
import dash_bootstrap_components as dbc

def get_upload_component():
    return dcc.Upload(
        id='upload-data',
        children=html.Button('Upload Data'),
        multiple=False
    )
