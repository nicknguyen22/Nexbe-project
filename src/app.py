import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html

NEXBE_LOGO = "/assets/nexbelogo.png"

navbar = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row([
                    dbc.Col(html.Img(src=NEXBE_LOGO, height="34px")),
                    dbc.Col(dbc.NavbarBrand('Project',style={'font-size':24,
                        'padding-left':'8px','font-weight':'500'}, 
                        className='ms-2')),
                    ],
                    align='left',
                    className="g-0"
                    ),
                href='https://www.nexbe.nz',
                style={'textDecoration': 'none'},
            ),
                dbc.Row([
                    dbc.Col([
                        dbc.Nav([
                            dbc.NavItem(dbc.NavLink('Electricity Generation & Carbon Emission', 
                                href='/',active='exact'),style={'align':'left','width':'340px'}),
                            dbc.NavItem(dbc.NavLink("ICP's Electricity Usage", 
                                href='/e_usage',active='exact')),
                        ])
                    ])
                ],
                align='left',
                className="g-0"
                ),
        ]),
    sticky= 'top',
    color='#df6b2e',
    dark=True,
    style={'height':'60px'}
)

# # Define components
app = dash.Dash(__name__, use_pages=True, 
                external_stylesheets=[
                dbc.themes.UNITED, 
                dbc.icons.FONT_AWESOME
                ])
server = app.server


# Layout 
app.layout = dbc.Container([
    navbar,
    html.Br(),
    dash.page_container,
    html.Br()

],style={'backgroundColor':'#F5F5F5'})



if __name__ == "__main__":
    app.run_server(debug=False)