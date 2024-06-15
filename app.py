import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import numpy as np
import pandas as pd
from dash import dash_table
from ssi_module import calculate_ssi
from planck_module import planck

# Constants
CORRECTION_FACTOR_DAYLIGHT = 14388 / 14380
CORRECTION_FACTOR_ILLUM_A = 14350 / 14388
MIN_WAVELENGTH = 300
MAX_WAVELENGTH = 830
DEFAULT_DAYLIGHT_CCT = 5000
DEFAULT_BLACKBODY_CCT = 3200
MIN_DAYLIGHT_CCT = 4000
MAX_DAYLIGHT_CCT = 25000
MIN_BLACKBODY_CCT = 1000
MAX_BLACKBODY_CCT = 10000
wavelengths = np.linspace(300, 780, 480)

# Function to interpolate and normalize spectra
def interpolate_and_normalize(spec):
    wavelengths = np.arange(MIN_WAVELENGTH, MAX_WAVELENGTH + 1)
    spec_resample = np.interp(wavelengths, spec['wavelength'], spec['intensity'])
    spec_resample /= spec_resample[np.where(wavelengths == 560)]
    return pd.DataFrame({'wavelength': wavelengths, 'intensity': spec_resample})

# Load test spectra from the provided text file
file_path_test = 'testSources_test.csv'
file_path_ref = 'daylighttestSources.csv'
test_spectra_df = pd.read_csv(file_path_test)
ref_spectra_df = pd.read_csv(file_path_ref)

# Print the data to verify loading
print(test_spectra_df.head())
print(ref_spectra_df.head())

# Interpolate and normalize each test spectrum
warm_led_spec = interpolate_and_normalize(test_spectra_df[['wavelength', 'Warm LED']].rename(columns={'Warm LED': 'intensity'}))
cool_led_spec = interpolate_and_normalize(test_spectra_df[['wavelength', 'Cool LED']].rename(columns={'Cool LED': 'intensity'}))
hmi_spec = interpolate_and_normalize(test_spectra_df[['wavelength', 'HMI']].rename(columns={'HMI': 'intensity'}))
xenon_spec = interpolate_and_normalize(test_spectra_df[['wavelength', 'Xenon']].rename(columns={'Xenon': 'intensity'}))
D50_spec = interpolate_and_normalize(ref_spectra_df[['wavelength', 'D50']].rename(columns={'D50': 'intensity'}))
D55_spec = interpolate_and_normalize(ref_spectra_df[['wavelength', 'D55']].rename(columns={'D55': 'intensity'}))
D65_spec = interpolate_and_normalize(ref_spectra_df[['wavelength', 'D65']].rename(columns={'D65': 'intensity'}))
D75_spec = interpolate_and_normalize(ref_spectra_df[['wavelength', 'D75']].rename(columns={'D75': 'intensity'}))
CCTCustomBB = 3200
CCTCustomDay = 5000
fluorescent_specs = {}
for i in range(1, 13):
    name = f'F{i}'
    fluorescent_specs[name] = interpolate_and_normalize(test_spectra_df[['wavelength', name]].rename(columns={name: 'intensity'}))

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

# Define layout
app.layout = dbc.Container(fluid=True, children=[
    dbc.NavbarSimple(
        brand="SSI Calculator",
        brand_href="#",
        color="dark",
        dark=True,
        fluid=True,
    ),
    dcc.Store(id='stored-cct-value'),

    dbc.Tabs([
        dbc.Tab(label="Calculations", children=[
            dbc.Row([
                dbc.Col(width=4, children=[  # Half the screen width for settings, reference, and spectral data
                    dbc.Card([
                        dbc.CardHeader("Test Spectrum"),
                        dbc.CardBody([
                            dcc.Dropdown(
                                id='testChoice',
                                options=[{'label': f'F{i}', 'value': f'F{i}'} for i in range(1, 13)] + [
                                    {'label': 'Warm LED', 'value': 'Warm LED'},
                                    {'label': 'Cool LED', 'value': 'Cool LED'},
                                    {'label': 'HMI', 'value': 'HMI'},
                                    {'label': 'Xenon', 'value': 'Xenon'},
                                    {'label': 'Custom', 'value': 'Custom'}
                                ],
                                value='Warm LED'
                            ),
                            html.Div(id='customTestSpecInputs')
                        ])
                    ]),
                    dbc.Card([
                        dbc.CardHeader("Reference Spectrum"),
                        dbc.CardBody([
                            dcc.Dropdown(
                                id='refChoice',
                                options=[
                                    {'label': 'Blackbody: A', 'value': 'A'},
                                    #{'label': 'Blackbody: Custom CCT', 'value': 'Custom_Blackbody'},
                                    {'label': 'Daylight: D50', 'value': 'D50'},
                                    {'label': 'Daylight: D55', 'value': 'D55'},
                                    {'label': 'Daylight: D65', 'value': 'D65'},
                                    {'label': 'Daylight: D75', 'value': 'D75'},
                                    #{'label': 'Daylight: Custom CCT', 'value': 'Custom_Daylight'}
                                ],
                                value='D50'
                            ),
                            html.Div(id='refSpecInputs')
                        ])
                    ]),
                    dbc.Card([
                        dbc.CardHeader("Spectral Data"),
                        dbc.CardBody([
                            dcc.Tabs([
                                dcc.Tab(label='Test Spectrum', children=[
                                    html.Div(id='spectraTest'),
                                ]),
                                dcc.Tab(label='Reference Spectrum', children=[
                                    html.Div(id='spectraRef'),
                                ]),
                            ]),
                        ])
                    ]),
                ]),
                dbc.Col(width=8, children=[  # Half the screen width for the graph
                    dbc.Card([
                        dbc.CardHeader("Graph"),
                        dbc.CardBody([dcc.Graph(id='plotRef')]),
                    ]),
                    dbc.Card([
                        dbc.CardHeader("Spectral Similarity Index (SSI)"),
                        dbc.CardBody([html.H4(id='ssiText', className='card-text')]),
                    ]),
                ]),
            ]),
        ]),
        dbc.Tab(label="About", children=[
            dbc.Container([
                html.H2('Introduction'),
                dcc.Markdown('Include the content of INTRODUCTION.md here...'),
                html.H2('Software'),
                html.P('Built using ...'),
                html.H2('License Terms'),
                dcc.Markdown('Include the content of LICENSE.md here...'),
            ])
        ]),
    ]),
])
# Callbacks for updating UI and calculations
@app.callback(
    Output('customTestSpecInputs', 'children'),
    Input('testChoice', 'value')
)
def update_test_spec_inputs(test_choice):
    if test_choice == 'Custom':
        return [
            dbc.Row([
                dbc.Label("Minimum Wavelength"),
                dbc.Input(type="number", id="testWlMin", value=MIN_WAVELENGTH, min=MIN_WAVELENGTH, max=400),
            ]),
            dbc.Row([
                dbc.Label("Maximum Wavelength"),
                dbc.Input(type="number", id="testWlMax", value=MAX_WAVELENGTH, min=700, max=MAX_WAVELENGTH),
            ]),
            dbc.Row([
                dbc.Label("Wavelength Increments"),
                dbc.Input(type="number", id="testWlInc", value=1, min=0.1, max=10),
            ]),
        ]
    return []



# @app.callback(
#     Output('refSpecInputs', 'children'),
#     Input('refChoice', 'value')
# )
# def update_ref_spec_inputs(ref_choice):
#     if ref_choice == 'Daylight':
#         return [
#             dbc.RadioItems(
#                 id='refCieD',
#                 options=[
#                     {'label': 'CCT', 'value': 'CCT'},
#                     {'label': 'D50', 'value': 5000 * CORRECTION_FACTOR_DAYLIGHT},
#                     {'label': 'D55', 'value': 5500 * CORRECTION_FACTOR_DAYLIGHT},
#                     {'label': 'D65', 'value': 6500 * CORRECTION_FACTOR_DAYLIGHT},
#                     {'label': 'D75', 'value': 7500 * CORRECTION_FACTOR_DAYLIGHT},
#                 ],
#                 inline=True,
#                 value='CCT'
#             ),
#             dbc.Row([
#                 dbc.Label("CCT"),
#                 dbc.Input(type="number", id="refCctD", value=DEFAULT_DAYLIGHT_CCT, min=MIN_DAYLIGHT_CCT, max=MAX_DAYLIGHT_CCT, disabled=False),
#             ]),
#         ]
#     elif ref_choice == 'Blackbody':
#         return [
#             dbc.RadioItems(
#                 id='refCieP',
#                 options=[
#                     {'label': 'CCT', 'value': 'CCT'},
#                     {'label': 'A', 'value': 2855.542},
#                 ],
#                 inline=True,
#                 value='CCT'
#             ),
#             dbc.Row([
#                 dbc.Label("CCT"),
#                 dbc.Input(type="number", id="refCctP", value=DEFAULT_BLACKBODY_CCT, min=MIN_BLACKBODY_CCT, max=MAX_BLACKBODY_CCT, disabled=False),
#             ]),
#         ]
#     return []

@app.callback(
    Output('refSpecInputs', 'children'),
    Output('stored-cct-value', 'data'),
    Input('refChoice', 'value')
)
def update_ref_spec_inputs(ref_choice):
    if ref_choice == 'D50':
        cct_value = 5000 * CORRECTION_FACTOR_DAYLIGHT
    elif ref_choice == 'D55':
        cct_value = 5500 * CORRECTION_FACTOR_DAYLIGHT
    elif ref_choice == 'D65':
        cct_value = 6500 * CORRECTION_FACTOR_DAYLIGHT
    elif ref_choice == 'D75':
        cct_value = 7500 * CORRECTION_FACTOR_DAYLIGHT
    elif ref_choice == 'Custom_Blackbody':
        cct_value = 3200
    elif ref_choice == 'A':
        cct_value = 2855.542 
    else:
        return [], None

    return dbc.Row([
        dbc.Label("CCT"),
        dbc.Input(
            type="number",
            id="refCct",
            value=cct_value,
            min=MIN_DAYLIGHT_CCT if ref_choice in ['D50', 'D55', 'D65', 'D75', 'Custom_Daylight'] else MIN_BLACKBODY_CCT,
            max=MAX_DAYLIGHT_CCT if ref_choice in ['D50', 'D55', 'D65', 'D75', 'Custom_Daylight'] else MAX_BLACKBODY_CCT,
            disabled=ref_choice not in ['Custom_Blackbody', 'Custom_Daylight']
        )
    ]), cct_value



# Callbacks for rendering tables and plots
@app.callback(
    Output('spectraTest', 'children'),
    Input('testChoice', 'value')
)
def update_spectra_test_table(test_choice):
    if test_choice == 'Warm LED':
        df = warm_led_spec
    elif test_choice == 'Cool LED':
        df = cool_led_spec
    elif test_choice == 'HMI':
        df = hmi_spec
    elif test_choice == 'Xenon':
        df = xenon_spec
    elif test_choice in fluorescent_specs:
        df = fluorescent_specs[test_choice]
    else:
        return html.Div("Select a test spectrum to view data.")
    
    return dash_table.DataTable(
        data=df.to_dict('records'),
        columns=[{"name": i, "id": i} for i in df.columns],
        editable=True,  # Allows users to edit the data in the table
        sort_action="native",  # Allows sorting by columns
        page_action="native",  # Enables pagination
        page_size=1000,  # Number of rows per page
        export_format="csv",
        export_headers="display",
        merge_duplicate_headers=True
    )

# @app.callback(
#     [Output('refCctD', 'value'),
#      Output('refCctD', 'disabled')],
#     Input('refCieD', 'value')
# )
# def update_cct_input(ref_cie_d_value):
#     if ref_cie_d_value == 'CCT':
#         return DEFAULT_DAYLIGHT_CCT, False
#     else:
#         return ref_cie_d_value, True



@app.callback(
    Output('spectraRef', 'children'),
    Input('refChoice', 'value')
)
def update_spectra_ref_table(ref_choice):
    if ref_choice == 'A':
        custom_spec_bb = planck(2855.542, wavelengths)
        df = interpolate_and_normalize(custom_spec_bb)
    elif ref_choice == 'D50':
        df = D50_spec
    elif ref_choice == 'D55':
        df = D55_spec
    elif ref_choice == 'D65':
        df = D65_spec
    elif ref_choice == 'D75':
        df = D75_spec
    elif ref_choice == "Custom_Blackbody":
        custom_spec_bb = planck(3200, wavelengths)
        df = interpolate_and_normalize(custom_spec_bb)
    else:
        return html.Div("Select a reference spectrum to view data.")
    
    return dash_table.DataTable(
        data=df.to_dict('records'),
        columns=[{"name": i, "id": i} for i in df.columns],
        editable=False,  # Allows users to edit the data in the table
        sort_action="native",  # Allows sorting by columns
        page_action="native",  # Enables pagination
        page_size=1000,  # Number of rows per page
        export_format="csv",
        export_headers="display",
        merge_duplicate_headers=True
    )

@app.callback(
    Output('plotRef', 'figure'),
    [Input('testChoice', 'value'), Input('refChoice', 'value')]
)
def update_plot(test_choice, ref_choice):
    fig = go.Figure()

    if test_choice == 'Warm LED':
        fig.add_trace(go.Scatter(x=warm_led_spec['wavelength'], y=warm_led_spec['intensity'], mode='lines', name='Test Spectrum'))
    elif test_choice == 'Cool LED':
        fig.add_trace(go.Scatter(x=cool_led_spec['wavelength'], y=cool_led_spec['intensity'], mode='lines', name='Test Spectrum'))
    elif test_choice == 'HMI':
        fig.add_trace(go.Scatter(x=hmi_spec['wavelength'], y=hmi_spec['intensity'], mode='lines', name='Test Spectrum'))
    elif test_choice == 'Xenon':
        fig.add_trace(go.Scatter(x=xenon_spec['wavelength'], y=xenon_spec['intensity'], mode='lines', name='Test Spectrum'))
    elif test_choice in fluorescent_specs:
        fig.add_trace(go.Scatter(x=fluorescent_specs[test_choice]['wavelength'], y=fluorescent_specs[test_choice]['intensity'], mode='lines', name='Test Spectrum'))


    if ref_choice == 'D50':
        fig.add_trace(go.Scatter(x=D50_spec['wavelength'], y=D50_spec['intensity'], mode='lines', name='Reference Spectrum'))
    elif ref_choice == 'D55':
        fig.add_trace(go.Scatter(x=D55_spec['wavelength'], y=D55_spec['intensity'], mode='lines', name='Reference Spectrum'))
    elif ref_choice == 'D65':
        fig.add_trace(go.Scatter(x=D65_spec['wavelength'], y=D65_spec['intensity'], mode='lines', name='Reference Spectrum'))
    elif ref_choice == 'D75':
        fig.add_trace(go.Scatter(x=D75_spec['wavelength'], y=D75_spec['intensity'], mode='lines', name='Reference Spectrum'))
    elif ref_choice == 'Custom_Blackbody':
        custom_spec_bb = planck(3200, wavelengths)
        custom_spec_bb_norm = interpolate_and_normalize(custom_spec_bb)
        fig.add_trace(go.Scatter(x=custom_spec_bb_norm['wavelength'], y=custom_spec_bb_norm['intensity'], mode='lines', name='Reference Spectrum'))
    elif ref_choice == 'A':
        custom_spec_bb = planck(2855.542, wavelengths)
        custom_spec_bb_norm = interpolate_and_normalize(custom_spec_bb)
        fig.add_trace(go.Scatter(x=custom_spec_bb_norm['wavelength'], y=custom_spec_bb_norm['intensity'], mode='lines', name='Reference Spectrum'))

    return fig


@app.callback(
    Output('ssiText', 'children'),
    [Input('testChoice', 'value'), Input('refChoice', 'value')]
)
def update_ssi_text(test_choice, ref_choice):

    test_data = None
    ref_data = None

    if test_choice == 'Warm LED':
        test_data = warm_led_spec
    elif test_choice == 'Cool LED':
        test_data = cool_led_spec
    elif test_choice == 'HMI':
        test_data = hmi_spec
    elif test_choice == 'Xenon':
        test_data = xenon_spec
    elif test_choice in fluorescent_specs:
        test_data = fluorescent_specs[test_choice]

    if ref_choice == 'D50':
        ref_data = D50_spec
    elif ref_choice == 'D55':
        ref_data = D55_spec
    elif ref_choice == 'D65':
        ref_data = D65_spec
    elif ref_choice == 'D75':
        ref_data = D75_spec
    elif ref_choice == 'Custom_Blackbody':
        custom_spec_bb = planck(3200, wavelengths)
        ref_data = interpolate_and_normalize(custom_spec_bb)

    elif ref_choice == 'A':
        custom_spec_bb = planck(2855.542, wavelengths)
        ref_data = interpolate_and_normalize(custom_spec_bb)

    if test_choice is None or ref_choice is None:
        return "Spectral Similarity Index: N/A"

    # Assuming your compute_ssi function expects a certain format, adapt as needed
    # Here we assume compute_ssi expects numpy arrays
    test_intensity = test_data['intensity']
    ref_intensity = ref_data['intensity']
    test_wavelengths = test_data['wavelength']
    ref_wavelengths = ref_data['wavelength']

    # Calculate the SSI using the imported function
    ssi_value = calculate_ssi(test_wavelengths, test_intensity, ref_wavelengths, ref_intensity)
    return f"{ssi_value:.2f}"

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
