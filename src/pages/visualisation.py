import dash                                     
from dash import Dash, dcc, html, Input, Output, callback
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from datetime import date
import pandas as pd
import sqlite3
import os

dash.register_page(__name__,path='/')
# ------------------------------------------------------------------------------
# Variables
WKD = os.getcwd()

#Colors 
COLR1 = '#E74E1C' # Orange
COLR2 = '#00A599' # Light Teal
COLR3 = '#EFA246' # Yellow
COLR4 = '#3F9C35' # Green
COLR5 = '#005B82' # Teal
COLR6 = '#A5ACAF' # Gray
COLR7 = '#6E267B' # Purple
COLR8 = '#404545' # DarkGray

CWOOD = '#e74e1c'
CCOAL = '#eb7030'
CDIESEL = '#ee8c48'
CGAS = '#f2a663'
CGEO = '#f5be83'
CHYDRO = '#fad4a5'
CWIND = '#ffeac9'


# ------------------------------------------------------------------------------
# Data preparation

nxb = sqlite3.connect(WKD+'/database/nexbe.db')

# loading data from database
yr_ovr = pd.read_sql_query('SELECT * FROM viz_yr_ems_ov',nxb)

yr_elc = pd.read_sql_query('SELECT * FROM yearly_elec_fueltype',nxb)

yr_detail = pd.read_sql_query('SELECT * FROM viz_yr_ems_detail',nxb)

dy_elc = pd.read_sql_query('SELECT * FROM daily_elec_fueltype',nxb)

weather = pd.read_sql_query('SELECT * FROM weather',nxb)

elec_detail = pd.read_sql_query('SELECT * FROM elec_detail',nxb)
elec_detail['Trading_Date'] = pd.to_datetime(elec_detail['Trading_Date'])


ems_detail = pd.read_sql_query('SELECT * FROM daily_ems',nxb)
ems_detail['Trading_Date'] = pd.to_datetime(ems_detail['Trading_Date'])

# ------------------------------------------------------------------------------
# Functions

# Function conver hex to rgb
def hex_to_rgb(hex_color: str) -> tuple:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = hex_color * 2
    return int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)

# Function checking fuel type availability per day
def check_fuel(data):
    fuels = ['Coal','Diesel','Gas','Geo','Hydro','Wind','Wood']
    for fuel in fuels:
        if not (data['Fuel_Code']==fuel).any():
            n = [data.iloc[0,0],fuel] + ([0]*50)
            data.loc[len(data)] = n
    return data

# Function for first overview chart
def main_chart():
    print(f'This is CWD {os.getcwd()}')
    year_chosen = [2003,2022]
    df = yr_ovr[(yr_ovr['Year']>=year_chosen[0])&(yr_ovr['Year']<=year_chosen[1])]
    df2 = yr_elc[(yr_elc['Year']>=year_chosen[0])&(yr_elc['Year']<=year_chosen[1])]
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Total emission chart
    fig.add_trace(
        go.Scatter(x = df['Year'], y = df['Emission(KtCO2)'], name='Emission(KtCO2)', 
            mode='lines+markers',
            line=dict(color=COLR7),
            marker=dict(symbol="diamond",size=6,color=COLR7),
            hovertemplate = '%{y:,.2f} KtCO2'),
        secondary_y=True
    )

    # Electricity generation chart
    
    fig.add_trace(
        go.Bar(x=df2['Year'], y=df2['Wind'], name='Wind(GWh)',
            marker=dict(color=CWIND,line = dict(width=0)),
            hovertemplate = '%{y:,.2f} GWh'),
        secondary_y=False
    )

    fig.add_trace(
        go.Bar(x=df2['Year'], y=df2['Hydro'], name='Hydro(GWh)',
            marker=dict(color=CHYDRO,line = dict(width=0)),
            hovertemplate = '%{y:,.2f} GWh'),
        secondary_y=False
    )

    fig.add_trace(
        go.Bar(x=df2['Year'], y=df2['Geo'], name='Geothermal(GWh)',
            marker=dict(color=CGEO,line = dict(width=0)),
            hovertemplate = '%{y:,.2f} GWh'),
        secondary_y=False
    )

    fig.add_trace(
        go.Bar(x=df2['Year'], y=df2['Gas'], name='Gas(GWh)',
            marker=dict(color=CGAS,line = dict(width=0)),
            hovertemplate = '%{y:,.2f} GWh'),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Bar(x=df2['Year'], y=df2['Diesel'], name='Diesel(GWh)',
            marker=dict(color=CDIESEL,line = dict(width=0)),
            hovertemplate = '%{y:,.2f} GWh'),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Bar(x=df2['Year'], y=df2['Coal'], name='Coal(GWh)',
            marker=dict(color=CCOAL,line = dict(width=0)),
            hovertemplate = '%{y:,.2f} GWh'),
        secondary_y=False
    )

    fig.add_trace(
        go.Bar(x=df2['Year'], y=df2['Wood'], name='Wood(GWh)',
            marker=dict(color=CWOOD,line = dict(width=0)),
            hovertemplate = '%{y:,.2f} GWh'),
        secondary_y=False
    )
    

    fig.add_trace(
        go.Scatter(x=df['Year'], y=df['Generation(GWh)'], name='Total(GWh)',
            marker=dict(size=1, symbol='line-ew', line=dict(width=0.5, color='#FFFFFF')),
            mode="markers",
            showlegend=False,
            hovertemplate = '%{y:,.2f} GWh'),
        secondary_y=False
    )

    fig.update_xaxes(range = [2002.5,2022.5])

    # Add figure layout
    fig.update_layout(title_text=(f'<span style="font-size: 18px;"> Total Electricity Generated and Carbon Emission {year_chosen[0]} - {year_chosen[1]}</span>'),
        hovermode='x unified',
        plot_bgcolor='#FFFFFF',
        barmode = 'stack',
        margin = dict(r=20),
        xaxis = dict(tickmode = 'linear',tick0 = 2003,dtick = 1),
        legend=dict(orientation='h',yanchor='top',y=-0.1,xanchor='left',x=0)
        )
    fig.update_yaxes(title_text='Elec.Generated(GWh)', title_font=dict(size=12, 
        color=CCOAL), secondary_y=False)
    fig.update_yaxes(title_text='Emission (KtCO2)', title_font=dict(size=12,
    color=COLR3),tickformat= ',.1s',secondary_y=True)

    return fig

# Function for detail data display
def group_charts(df2,df2_1,df2_2,clk_year,daterange):
    fig2 = make_subplots(rows=4, cols=1,
                        shared_xaxes=True,
                        subplot_titles=('Temprature (C)','Carbon Intensity (g/KWh)', 
                            'Total Emission (KtCO2)','Total Generation (GWh)'),
                        vertical_spacing=0.1,
                        row_width=[0.4,0.2,0.2,0.2])

    # Chart 1 Temperature
    fig2.add_trace(
            go.Scatter(
                name='Avg Temp(C)',
                x=df2_1['Date(NZST)'],
                y=df2_1['Tavg(C)'],
                mode='lines',
                line=dict(color=COLR4)
            ),row=1,col=1)

    fig2.add_trace(
            go.Scatter(
                name='Max Temp(C)',
                x=df2_1['Date(NZST)'],
                y=df2_1['Tmax(C)'],
                mode='lines',
                marker=dict(color="#444"),
                line=dict(width=0),
                showlegend=False
            ),row=1,col=1)

    fig2.add_trace(
            go.Scatter(
                name='Min Temp(C)',
                x=df2_1['Date(NZST)'],
                y=df2_1['Tmin(C)'],
                marker=dict(color="#444"),
                line=dict(width=0),
                mode='lines',
                fillcolor=f'rgba{(*hex_to_rgb(COLR4), 0.1)}',
                fill='tonexty',
                showlegend=False
            ),row=1,col=1
        )

    # Chart 2 Carbon Intensity
    fig2.add_trace(
            go.Scatter(x = df2['Trading_Date'], y = df2['Carbon_Intensity(g/KWh)'], 
                name='Carbon Intensity(g/KWh)',
                line = dict(color=COLR5, width=2, dash='dot'),
                hovertemplate = '%{y:,.3f} g/KWh'),
                row=2,col=1
        )

    # Chart 3 Emission
    fig2.add_trace(
        go.Scatter(x = df2['Trading_Date'], y = df2['Emission(KtCO2)'],
            fill='tozeroy', marker=dict(color=COLR7),
            fillcolor=f'rgba{(*hex_to_rgb(COLR7), 0.2)}',
            name='Emission(KtCO2)', hovertemplate = '%{y:,.2f} KtCO2'),
            row=3,col=1
        )

    # Chart 4 Generation

    fig2.add_trace(
        go.Bar(x=df2_2['Trading_Date'], y=df2_2['Wind'], name='Wind(GWh)',
            marker=dict(color=CWIND,line = dict(width=0)),showlegend=False,
            hovertemplate = '%{y:,.2f} GWh'),
        row=4,col=1
    )

    fig2.add_trace(
        go.Bar(x=df2_2['Trading_Date'], y=df2_2['Hydro'], name='Hydro(GWh)',
            marker=dict(color=CHYDRO,line = dict(width=0)),showlegend=False,
            hovertemplate = '%{y:,.2f} GWh'),
        row=4,col=1
    )

    fig2.add_trace(
        go.Bar(x=df2_2['Trading_Date'], y=df2_2['Geo'], name='Geothermal(GWh)',
            marker=dict(color=CGEO,line = dict(width=0)),showlegend=False,
            hovertemplate = '%{y:,.2f} GWh'),
        row=4,col=1
    )

    fig2.add_trace(
        go.Bar(x=df2_2['Trading_Date'], y=df2_2['Gas'], name='Gas(GWh)',
            marker=dict(color=CGAS,line = dict(width=0)),showlegend=False,
            hovertemplate = '%{y:,.2f} GWh'),
        row=4,col=1
    )
    
    fig2.add_trace(
        go.Bar(x=df2_2['Trading_Date'], y=df2_2['Diesel'], name='Diesel(GWh)',
            marker=dict(color=CDIESEL,line = dict(width=0)),showlegend=False,
            hovertemplate = '%{y:,.2f} GWh'),
        row=4,col=1
    )
    
    fig2.add_trace(
        go.Bar(x=df2_2['Trading_Date'], y=df2_2['Coal'], name='Coal(GWh)',
            marker=dict(color=CCOAL,line = dict(width=0)),showlegend=False,
            hovertemplate = '%{y:,.2f} GWh'),
        row=4,col=1
    )

    fig2.add_trace(
        go.Bar(x=df2_2['Trading_Date'], y=df2_2['Wood'], name='Wood(GWh)',
            marker=dict(color=CWOOD,line = dict(width=0)),showlegend=False,
            hovertemplate = '%{y:,.2f} GWh'),
        row=4,col=1
    )

    fig2.add_trace(
        go.Scatter(x=df2['Trading_Date'], y=df2['Generation(GWh)'], name='Total(GWh)',
            marker=dict(size=1, symbol='line-ew', line=dict(width=0.5, 
            color='rgba(135, 206, 250, 0)',)),
            mode="markers",
            showlegend=False,
            hovertemplate = '%{y:,.2f} GWh'),
        row=4,col=1
    )

    # Layout setting

    fig2.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
    fig2.update_traces(xaxis='x4')
    fig2.update_xaxes(row=4, col=1, range = daterange)

    # Subplots title setting
    for n in range (4):
        fig2.layout.annotations[n].update(x=0,font_size=12,xanchor ='left') 

    fig2.update_layout( height=820,
        plot_bgcolor = '#FFFFFF', paper_bgcolor = '#FFFFFF',
        title_text=(f'<span style="font-size: 18px; ">Detail Data of year {clk_year}</span> <span style="font-size: 12px; font-style: italic"> (Click on the bar chart above to view different year)</span>'),
        title_x= 0.035,
        hovermode='x unified',
        margin = dict(r=30,b=50),
        barmode = 'stack',
        xaxis4 = dict(tickangle = 0, tickfont =dict(size=10),showticklabels=True, 
            dtick ='M1', range = daterange, type="date"),
        xaxis4_rangeslider_visible=True, xaxis4_rangeslider_thickness=0.05,
        legend=dict(orientation='h',yanchor='bottom',y=1.02,xanchor='right',x=1)
                    )
    return fig2

# Function for pie chart
def pie_chart(df3,hov_year):

    # Data preparation 
    df3=pd.melt(df3,id_vars='Year',var_name='Fuel_Type',value_name='Generation(GWh)',
    value_vars=['Coal','Diesel','Gas','Geo','Hydro','Wind','Wood'])
    lbls = df3['Fuel_Type'].tolist()
    vals = df3['Generation(GWh)'].tolist()

    # Plot pie chart
    fig3 = go.Figure(data=[go.Pie(labels=lbls, values=vals,hole=.4, sort=False)])
    fig3.update_traces(marker=dict(colors=[CCOAL,CDIESEL,CGAS,CGEO,CHYDRO,CWIND,
        CWOOD],line=dict(width=0)))
    fig3.update_layout(title_text=(f'<span style="font-size: 18px; > Electricity Generated by Sources Year {hov_year}</span>'),
        margin = dict(l=0,r=0),
        legend=dict(orientation='h',yanchor='top',y=-0.1,xanchor='left',x=0)
        )

    return fig3

# Function for indicator cards
def indicator_cards(gen_df,ci_df,ems_df):
    indicators = dbc.Col([
        dbc.Card([
            dbc.CardBody([
                html.H6('Total Generation',style={'textAlign':'center','font-size':15}),
                html.H4(f"{(gen_df['Generation(MWh)'].sum()/1000):,.3f} GWh",style={'textAlign':'center',
                    'color':COLR1})
                ])
        ]),

        html.Br(),

        dbc.Card([
            dbc.CardBody([
                html.H6('Avg. Carbon Intensity',style={'textAlign':'center','font-size':15}),
                html.H4(f"{ci_df['Carbon_Intensity(g/KWh)'].mean():.3f} g/KWh",style={'textAlign':'center',
                    'color':COLR5})
                ])
        ]),

        html.Br(),

        dbc.Card([
            dbc.CardBody([
                html.H6('Total Emission',style={'textAlign':'center','font-size':15}),
                html.H4(f"{(ems_df['Emission(tCO2)'].sum())/1000:.3f} KtCO2",style={'textAlign':'center',
                    'color':COLR7})
                ])
        ])
    ])
    return indicators

# Function for group daily detail charts
def daily_charts(df3,df4,clk_date):
    # Data preparation
    # Generation dataframe
    df3_0 = pd.melt(df3,id_vars = ['Trading_Date','Fuel_Code'], var_name='Trading_Period',
        value_vars= df3.columns[df3.columns.str.startswith('TP')].to_list(),
        value_name= 'Generation(MWh)')
    df3_1 = pd.pivot(df3_0, index=['Trading_Date','Trading_Period'], columns='Fuel_Code',
        values='Generation(MWh)').reset_index().rename_axis(None,axis=1)
    
    # Carbon intensity dataframe
    df4_1 = pd.melt(df4,id_vars='Trading_Date',var_name= 'Trading_Period' ,
        value_name='Carbon_Intensity(g/KWh)', 
        value_vars=df4.columns[df4.columns.str.startswith('c_int')].to_list())
    df4_1['Trading_Period'] = df4_1['Trading_Period'].str.replace('c_int','TP')

    # Emission dataframe
    df4_2 = pd.melt(df4,id_vars='Trading_Date',var_name= 'Trading_Period' ,
        value_name='Emission(tCO2)', 
        value_vars=df4.columns[df4.columns.str.startswith('eTP')].to_list())
    df4_2['Trading_Period'] = df4_2['Trading_Period'].str.replace('e','')

    # Create summary cards
    cards = indicator_cards(df3_0,df4_1,df4_2)

    # Plot group charts
    fig4 = make_subplots(rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=('Carbon Intensity (g/KWh)','Total Emission (tCO2)',
            'Total Generation (MWh)'),
        vertical_spacing=0.1,
        row_width=[0.6,0.2,0.2])
    
    # Carbon intensity
    fig4.add_trace(
        go.Scatter(x = df4_1['Trading_Period'], y = df4_1['Carbon_Intensity(g/KWh)'], 
            name='Carbon Intensity(g/KWh)',
            line = dict(color=COLR5, width=2, dash='dot'),
            hovertemplate = '%{y:,.3f} g/KWh'),
            row=1,col=1
        )

    # Emission
    fig4.add_trace(
        go.Scatter(x = df4_2['Trading_Period'], y = df4_2['Emission(tCO2)'],
            fill='tozeroy', marker=dict(color=COLR7),
            fillcolor=f'rgba{(*hex_to_rgb(COLR7), 0.2)}',
            name='Emission(tCO2)', hovertemplate = '%{y:,.2f} tCO2'),
            row=2,col=1
        )

    # Generation
    fig4.add_trace(
        go.Bar(x=df3_1['Trading_Period'], y=df3_1['Wind'], name='Wind(MWh)',
            marker=dict(color=CWIND,line = dict(width=0)),showlegend=False,
            hovertemplate = '%{y:,.2f} MWh'),
        row=3,col=1
    )

    fig4.add_trace(
        go.Bar(x=df3_1['Trading_Period'], y=df3_1['Hydro'], name='Hydro(MWh)',
            marker=dict(color=CHYDRO,line = dict(width=0)),showlegend=False,
            hovertemplate = '%{y:,.2f} MWh'),
        row=3,col=1
    )

    fig4.add_trace(
        go.Bar(x=df3_1['Trading_Period'], y=df3_1['Geo'], name='Geothermal(MWh)',
            marker=dict(color=CGEO,line = dict(width=0)),showlegend=False,
            hovertemplate = '%{y:,.2f} MWh'),
        row=3,col=1
    )

    fig4.add_trace(
        go.Bar(x=df3_1['Trading_Period'], y=df3_1['Gas'], name='Gas(MWh)',
            marker=dict(color=CGAS,line = dict(width=0)),showlegend=False,
            hovertemplate = '%{y:,.2f} MWh'),
        row=3,col=1
    )
    
    fig4.add_trace(
        go.Bar(x=df3_1['Trading_Period'], y=df3_1['Diesel'], name='Diesel(MWh)',
            marker=dict(color=CDIESEL,line = dict(width=0)),showlegend=False,
            hovertemplate = '%{y:,.2f} MWh'),
        row=3,col=1
    )
    
    fig4.add_trace(
        go.Bar(x=df3_1['Trading_Period'], y=df3_1['Coal'], name='Coal(MWh)',
            marker=dict(color=CCOAL,line = dict(width=0)),showlegend=False,
            hovertemplate = '%{y:,.2f} MWh'),
        row=3,col=1
    )

    fig4.add_trace(
        go.Bar(x=df3_1['Trading_Period'], y=df3_1['Wood'], name='Wood(MWh)',
            marker=dict(color=CWOOD,line = dict(width=0)),showlegend=False,
            hovertemplate = '%{y:,.2f} MWh'),
        row=3,col=1
    )

    # Layout setting

    fig4.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
    fig4.update_traces(xaxis='x3')
    fig4.update_xaxes(row=3, col=1, type='category', categoryorder='category ascending')

    # Subplots title setting
    for n in range (3):
        fig4.layout.annotations[n].update(x=0,font_size=12,xanchor ='left')
    sel_date = str(clk_date).split(' ',1)[0]
    fig4.update_layout( height=622,
        plot_bgcolor = '#FFFFFF', paper_bgcolor = '#FFFFFF',
        title_text=(f'<span style="font-size: 18px; ">Detail Data of {sel_date}</span> <span style="font-size: 12px; font-style: italic"> (Click on the bar chart above to view different day)</span>'),
        yaxis2 = dict(range=[(df4_2.loc[df4_2['Emission(tCO2)']>0,'Emission(tCO2)'].min())-5,
            df4_2['Emission(tCO2)'].max()]),
        title_x= 0.035,
        hovermode='x unified',
        margin = dict(r=30,b=50),
        barmode = 'stack',
        xaxis3 = dict(tickangle = -45, tickfont =dict(size=10),showticklabels=True),
        legend=dict(orientation='h',yanchor='bottom',y=1.02,xanchor='right',x=1))

    return fig4, cards

# ------------------------------------------------------------------------------
# Layout
layout = dbc.Container([
    html.H4("Electricity Generation and Carbon Footprint in New Zealand", style={'textAlign':'center'}),
    html.Br(),html.Br(),
    dbc.Row([
        dbc.Col(dbc.Card(dcc.Graph(id='overview-graph', figure=main_chart())),
            width=8),
        dbc.Col(dbc.Card(
            dcc.Loading(id='pie_load',children = [
                dcc.Graph(id='pie-graph', figure = {})]
                )
            ),
            width=4)
    ]),
    html.Br(),
    dbc.Row([  
        dcc.Loading(id='detail_loading',children=
            [dbc.Card(dcc.Graph(id='detail-graph', figure={}, clickData=None, 
                hoverData=None)
            )],
            type = 'default',
        )
    ]),
    html.Br(),
    dbc.Row([  
        dbc.Col([
            dbc.Card([
                html.Br(),
                html.H6('Date selection method',style={'font-size':15}),
                dbc.RadioItems(
                    options=[
                        {"label": "Click on the Detail Year chart above", "value": 1},
                        {"label": "Select date using Datapicker below", "value": 2},
                    ],
                    value=1,
                    id ='method',
                    switch=True
                ,style={'font-size':15}),
            ],style={'padding':'10px'}),

            html.Br(),

            dbc.Card([
                html.H6('Select the date',style={'font-size':15}),
                html.Div(id='datepicker',children=[])
            ],style={'padding':'10px'}),
            
            html.Br(),

            dcc.Loading(id='side_loading',children=[
                html.Div(id='side_cards')
            ],type = 'default'),
        ],width=2),
        
        dbc.Col(
            dcc.Loading(id='detail_loading2',children=
                [dbc.Card(dcc.Graph(id='detail-graph2', figure={}, clickData=None, 
                hoverData=None)
                )],
            type = 'default',),
            width=10),
        
    ])
],)

# ------------------------------------------------------------------------------
# Callback

# Pie chart
@callback(
    Output(component_id='pie-graph', component_property='figure'),
    Input(component_id='overview-graph', component_property='hoverData')
)
def update_pie_graph(hov_data):
    if hov_data is None:
        # Assign default hover data
        hov_year = 2022
        df3 = yr_elc[yr_elc['Year'] == hov_year]
  
        fig3 = pie_chart(df3,hov_year)

        return fig3

    else:
        # Get hover data
        hov_year = hov_data['points'][0]['x']
        df3 = yr_elc[yr_elc['Year'] == hov_year]

        fig3 = pie_chart(df3,hov_year)

        return fig3        

# Year detail charts
@callback(
    Output(component_id='detail-graph', component_property='figure'),
    Input(component_id='overview-graph', component_property='clickData')
)
def update_group_charts(clk_data):
    if clk_data is None:
        clk_year = 2022
        df2 = yr_detail[yr_detail['Year'] == clk_year]
        df2_1 = weather[weather['Year'] == clk_year]
        df2_2 = dy_elc[dy_elc['Year'] == clk_year]
        daterange = [df2_2['Trading_Date'].min(),df2_2['Trading_Date'].max()]

        fig2 = group_charts(df2,df2_1,df2_2,clk_year,daterange)

        return fig2
    else:

        clk_year = clk_data['points'][0]['x']
        df2 = yr_detail[yr_detail['Year'] == clk_year]
        df2_1 = weather[weather['Year'] == clk_year]
        df2_2 = dy_elc[dy_elc['Year'] == clk_year]
        daterange = [df2_2['Trading_Date'].min(),df2_2['Trading_Date'].max()]

        fig2 = group_charts(df2,df2_1,df2_2,clk_year,daterange)

        return fig2        

# Date picker panel
@callback(
    Output(component_id='datepicker', component_property='children'),
    Input(component_id='method', component_property='value'),
    Input(component_id='overview-graph', component_property='clickData'),
)
def dt_picker(method_val,clk_data):
    if method_val == 1:
        return html.Div(dcc.DatePickerSingle(
                id='dpicker',
                month_format='MMMM Y',
                placeholder='MMMM Y',
                min_date_allowed=date(2003, 1, 1),
                max_date_allowed=date(2022, 12, 31),
                initial_visible_month=date(2022, 1, 1),
                date=date(2022, 1, 1),
                disabled= True))
    elif (method_val==2) & (clk_data is None):
        year = 2022
        return html.Div(dcc.DatePickerSingle(
                id='dpicker',
                month_format='MMMM Y',
                placeholder='MMMM Y',
                min_date_allowed=date(2003, 1, 1),
                max_date_allowed=date(2022, 12, 31),
                initial_visible_month=date(year, 1, 1),
                date=date(year, 1, 1),
                disabled= False))
    elif (method_val==2) & (clk_data is not None):
        year = clk_data['points'][0]['x']
        return html.Div(dcc.DatePickerSingle(
                id='dpicker',
                month_format='MMMM Y',
                placeholder='MMMM Y',
                min_date_allowed=date(2003, 1, 1),
                max_date_allowed=date(2022, 12, 31),
                initial_visible_month=date(year, 1, 1),
                date=date(year, 1, 1),
                disabled= False))

# Daily detail charts and cards
@callback(
    Output(component_id='detail-graph2', component_property='figure'),
    Output(component_id='side_cards', component_property='children'),
    Input(component_id='method', component_property='value'),
    Input(component_id='dpicker', component_property='date'),
    Input(component_id='detail-graph', component_property='clickData')
)
def update_daily_charts(method_val,dp_date,clk_data):
    def chart_cards(clk_date):
        df3 = elec_detail[elec_detail['Trading_Date'] == clk_date]
        df3 = check_fuel(df3)
        df4 = ems_detail[ems_detail['Trading_Date'] == clk_date]
        fig4,cards = daily_charts(df3,df4,clk_date)
        return fig4,cards

    if (method_val ==1) & (clk_data is None):
        default = '2022-01-01'
        clk_date = pd.to_datetime(default)
        return chart_cards(clk_date)

    elif (method_val ==1) & (clk_data is not None):
        clk_date = clk_data['points'][0]['x']
        return chart_cards(clk_date)
    
    elif (method_val ==2):
        clk_date = dp_date
        return chart_cards(clk_date)




