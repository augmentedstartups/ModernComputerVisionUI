# General Imports
import cv2
import sys
import time
import torch
import numpy as np
import pandas as pd
from collections import deque
import dash_dangerously_set_inner_html
from dash import html
# Flask Imports
from flask import Flask, Response

# Plotly-Dash Imports 
import dash
from dash import Dash, html, dcc, Input, Output, State
import plotly.graph_objects as go
import plotly.express as px

from flask import Flask
import dash_bootstrap_components as dbc

from mainTracker import Tracker, vis_track, draw_lines, lines
from flask_cloudflared import run_with_cloudflared

from dash_iconify import DashIconify

import plotly.io as pio

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "11rem",
    "padding": "2rem 1rem",
    "background-color": "#F7F9FA",
}

CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [

        html.Img(src="/assets/aslogo.png", style={
            'max-width': '100%',
            'height': 'auto',
            'display': 'block',
            'padding-bottom': '20px',
        }),
        html.Hr([], className="horizontal dark mt-0"),
        dbc.Nav(
            [
                dbc.NavLink("Dashboard", href="/", active="exact", className="nav-link-text ms-1"),
                dbc.NavLink("Page 1", href="/page-1", active="exact", className="nav-link-text ms-1"),
                dbc.NavLink("Page 2", href="/page-2", active="exact", className="nav-link-text ms-1"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

# Variables
fps_prev = 0
fps_delta = '0';
vehiclestotal_prev = 0
vehiclestotal_delta = '0';

ValueMoney = 59000
iconz = DashIconify(icon="ic:twotone-directions-car", width=47, color="white")
Traffic_icon = DashIconify(icon="carbon:traffic-event", width=47, color="white")
FPS_icon = DashIconify(icon="ic:baseline-speed", width=47, color="white")
cctv_icon = DashIconify(icon="bxs:cctv", width=47, color="white")



dark = True
if dark:
    pio.templates.default = "plotly_dark"

# Init Flask Server
server = Flask(__name__)
run_with_cloudflared(server)
# Init Dash App
# app = Dash(__name__, server = server, external_stylesheets=[dbc.themes.MORPH, dbc.icons.BOOTSTRAP,'https://fonts.googleapis.com/css2?family=Montserrat'])
app = Dash(__name__, server=server, external_stylesheets=["assets/soft-ui-dashboard.css"])
# Init Tracker
tracker = Tracker(filter_classes=None, model='yolox-s', ckpt='weights/yolox_s.pth')

Main = []


# Sunburst Data Function
def build_hierarchical_dataframe(df, levels, value_column):
    df_all_trees = pd.DataFrame(columns=['id', 'parent', 'value'])
    for i, level in enumerate(levels):
        df_tree = pd.DataFrame(columns=['id', 'parent', 'value'])
        dfg = df.groupby(levels[i:]).sum()
        dfg = dfg.reset_index()
        df_tree['id'] = dfg[level].copy()
        if i < len(levels) - 1:
            df_tree['parent'] = dfg[levels[i + 1]].copy()
        else:
            df_tree['parent'] = 'total'
        df_tree['value'] = dfg[value_column]
        df_all_trees = df_all_trees.append(df_tree, ignore_index=True)
    total = pd.Series(dict(id='total', parent='',
                           value=df[value_column].sum(),
                           ))
    df_all_trees = df_all_trees.append(total, ignore_index=True)
    return df_all_trees


def update_layout1(figure, title, margin):
    figure.update_layout(
        title=title,
        xaxis=dict(automargin=False, showgrid=False, zeroline=False, showline=True, linecolor='white',
                   showticklabels=True, gridwidth=1, zerolinecolor='white',
                   tickfont=dict(family='Arial Black', size=14, color='#C2C6CC')),
        yaxis=dict(showgrid=True, automargin=False, zeroline=False, showline=True, gridcolor='#E5E6E5', gridwidth=0.5,
                   zerolinecolor='gray', zerolinewidth=1, linecolor='white', linewidth=1,
                   titlefont=dict(family='Arial, sans-serif', size=18, color='lightgrey'), showticklabels=True,
                   tickangle=0, tickfont=dict(family='Arial Black', size=14, color='#C2C6CC'),
                   tickmode='linear', tick0=0.0, dtick=1,
                   ),
        font_family="Arial Black",
        font_color="#201D4D",
        showlegend=False,
        paper_bgcolor='rgba(255,0,0 ,0)',
        plot_bgcolor='rgba(255,0,0,0)',
        width=600,
        height=338,
        autosize=False,
        margin=dict(
            l=50,
            r=50,
            b=40,
            t=50,
            pad=4, )

    )
    return figure


def update_layout2(figure, title, margin):
    figure.update_layout(
        title=title,
        xaxis=dict(automargin=False, showgrid=False, zeroline=False, showline=True, linecolor='white',
                   showticklabels=True, gridwidth=1, zerolinecolor='white',
                   tickfont=dict(family='Arial Black', size=14, color='#C2C6CC')),
        yaxis=dict(showgrid=True, automargin=False, zeroline=False, showline=True, gridcolor='#E5E6E5', gridwidth=0.5,
                   zerolinecolor='gray', zerolinewidth=1, linecolor='white', linewidth=1,
                   titlefont=dict(family='Arial, sans-serif', size=18, color='lightgrey'), showticklabels=True,
                   tickangle=0, tickfont=dict(family='Arial Black', size=14, color='#C2C6CC'),
                   tickmode='linear', tick0=0.0, dtick=1,
                   ),
        font_family="Arial Black",
        font_color="#201D4D",
        showlegend=False,
        paper_bgcolor='rgba(255,0,0 ,0)',
        plot_bgcolor='rgba(255,0,0,0)',
        autosize=False,
        margin=dict(
            l=50,
            r=50,
            b=40,
            t=50,
            pad=4, )

    )
    return figure


def update_layout3(figure, title, margin, GraphTick):
    figure.update_layout(
        title=title,
        xaxis=dict(automargin=False, showgrid=False, zeroline=False, showline=True, linecolor='white',
                   showticklabels=True, gridwidth=1, zerolinecolor='white',
                   tickfont=dict(family='Arial Black', size=14, color='#C2C6CC')),
        yaxis=dict(showgrid=True, automargin=False, zeroline=False, showline=True, gridcolor='#E5E6E5', gridwidth=0.5,
                   zerolinecolor='gray', zerolinewidth=1, linecolor='white', linewidth=1,
                   titlefont=dict(family='Arial, sans-serif', size=18, color='lightgrey'), showticklabels=True,
                   tickangle=0, tickfont=dict(family='Arial Black', size=14, color='#C2C6CC'),
                   tickmode='linear', tick0=0.0, dtick=GraphTick,
                   ),
        font_family="Arial Black",
        font_color="#201D4D",
        showlegend=False,
        paper_bgcolor='rgba(255,0,0 ,0)',
        plot_bgcolor='rgba(255,0,0,0)',
        autosize=False,
        margin=dict(
            l=50,
            r=50,
            b=40,
            t=50,
            pad=4, )

    )
    return figure


# -------------------------------------------------Getting Video Feeds ------------------------------#

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


class VideoCamera(object):
    def __init__(self):
        global res;
        self.video = cv2.VideoCapture(sys.argv[1])
        res = f"{int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))} x {int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))}"

    def __del__(self):
        self.video.release()
        cv2.destroyAllWindows()

    def get_frame(self):
        global fps,fps_prev,fps_delta
        success, image = self.video.read()
        if success:
            fps_prev = fps
            t1 = time_synchronized()
            image = draw_lines(lines, image)
            image, bbox, data = tracker.update(image, logger_=False)
            image = vis_track(image, bbox)
            Main.extend(data)
            
            fps = (1. / (time_synchronized() - t1))
            try:
                fps_delta = ((fps_prev -fps)/fps_prev)*100
            except ZeroDivisionError:
                fps_delta =  0
            ret, jpeg = cv2.imencode('.jpg', image)
            return jpeg.tobytes()
        else:
            return "Video is Completed !!!"


def gen(camera):
    fps = 0.0
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@server.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ---------------------------------------------------------------------------------------------------#
# Card Compnent


def create_card(Header, Value,Second_Value, cardcolor,icon_thumb):
    card = html.Div(
        html.Div([
            html.Div([
                html.Div([
                    html.Div([
                        html.Div([
                            html.P(Header,
                                   className="text-sm mb-0 text-capitalize font-weight-bold"),
                            html.H5(Value, className="font-weight-bolder mb-0")
                        ], className="numbers"),
                        html.Span([Second_Value], className="text-red text-sm font-weight-bolder"),
                    ], className="col-8"),
                    html.Div([
                        html.Div([
                            icon_thumb
                        ], className="icon icon-shape bg-gradient-primary shadow text-center border-radius-md")
                    ], className="col-4 text-end")
                ], className="row")
            ], className="card-body p-3")
        ], className="card mb-4")
    )
    return card




# Video Feed Component

videofeeds = html.Div([
    html.Div([
        html.Div([
            html.Img(src="/video_feed", style={
                # 'max-width': '100%',
                'height': 'auto',
                'display': 'block',
            }),
        ], className="row")
    ], className="card-body p-3")
], className="card mb-4")

# Header Component
header = dbc.Col(width=10,
                 children=[
                     html.Header(style={
                         'padding': '10px',
                         'text-align': 'left',
                         'background': '#1abc9c;',
                         'color': 'white;'

                     }, children=[html.H1(["Traffic Analytics Dashboard"],
                                          style={'textAlign': 'left', 'padding-bottom': '20px',
                                                 'padding-top': '20px'},
                                          )],
                     )]
                 )

figure1 = html.Div([
    html.Div([
        html.Div([
            dbc.Col([dcc.Graph(id="live-graph1")], style={'margin-top': '-10px'}),
        ], className="row")
    ], className="card-body p-3")
], className="card mb-4")

figure2 = html.Div([
    html.Div([
        html.Div([
            dbc.Col([dcc.Graph(id="live-graph2")], style={'margin-top': '-10px'}),
        ], className="row")
    ], className="card-body p-3")
], className="card mb-4")

piefig = html.Div([
    html.Div([
        html.Div([
            dbc.Col([dcc.Graph(id="piefig")], style={'margin-top': '-10px'}),
        ], className="row")
    ], className="card-body p-3")
], className="card mb-4")

# Grpahical Components

dirfig = html.Div([
    html.Div([
        html.Div([
            dbc.Col([dcc.Graph(id="dirfig")], style={'margin-top': '-10px'}),
        ], className="row")
    ], className="card-body p-3")
], className="card mb-4")

sunfig = html.Div([
    html.Div([
        html.Div([
            dbc.Col([dcc.Graph(id="sunfig")], style={'margin-top': '-10px'}),
        ], className="row")
    ], className="card-body p-3")
], className="card mb-4")

speedfig = html.Div([
    html.Div([
        dbc.Col([dcc.Graph(id="speedfig")], style={'margin-top': '-10px'}),
    ], className="card-body p-3")
], className="card mb-4")

infig = html.Div([
    html.Div([
        html.Div([
            dbc.Col([dcc.Graph(id="infig")], style={'margin-top': '-10px'}),
        ], className="row")
    ], className="card-body p-3")
], className="card mb-4")

fps = 0
res = "-"
stream = "Stream 1"
average_speed = 0
previous_av_speed = 0

# ----------------------------------------Off Canvas Form -----------------------------------------------------------#

dropdown = dbc.Form(
    [
        html.H6("Detection Model Selected :: YOLOX S", id="model-dropdown-head"),
        dbc.DropdownMenu(
            label="YOLOX S",
            id='model-dropdown',
            menu_variant="dark",
            children=[
                dbc.DropdownMenuItem("YOLOX S", id="yolox_s"),
                dbc.DropdownMenuItem(divider=True),
                dbc.DropdownMenuItem("YOLOX M", id="yolox_m"),
                dbc.DropdownMenuItem(divider=True),
                dbc.DropdownMenuItem("YOLOX L", id="yolox_l"),

            ],
        )
    ]
)

slider = dbc.Form(
    [
        dbc.Label("Confidence", html_for="slider"),
        dcc.Slider(id="slider", min=0, max=1, step=0.05, value=3, tooltip={"placement": "top", "always_visible": True},
                   className="sl"),
    ], style={'padding-top': '40px'}
)

form = dbc.Form([dropdown, dbc.DropdownMenuItem(divider=True), slider, dbc.DropdownMenuItem(divider=True),
                 dbc.Col(html.A(dbc.Button("run", id="run", color="primary")))])

# ----------------------------------------Off Canvas Menu -----------------------------------------------------------#
offcanvas = html.Div(children=[dbc.Button([html.I(className="bi bi-list"), ""],
                                          id="open-offcanvas-scrollable",
                                          n_clicks=0,
                                          color="danger",
                                          outline=True,
                                          size="lg"
                                          ),

                               dbc.Offcanvas(
                                   children=[
                                       html.H2("Configuration Menu", style={'padding-bottom': "60px"}),
                                       form,
                                       html.Div(id='update_tracker')
                                   ],
                                   id="offcanvas-scrollable",
                                   scrollable=True,
                                   placement="end",
                                   close_button=False,
                                   is_open=False,
                                   keyboard=True,
                                   style={
                                       'background-color': 'rgba(20,20,20,0.9)',
                                       'width': '550px',
                                       'padding': "20px 40px 20px 40px"

                                   }
                               )

                               ])


@app.callback(
    Output('offcanvas-scrollable', "is_open"),
    Input('open-offcanvas-scrollable', "n_clicks"),
    State("offcanvas-scrollable", "is_open")
)
def toggle_offcavas_scrollable(n1, is_open):
    if n1:
        return not is_open
    return is_open


@app.callback(
    Output("model-dropdown", "label"),
    Output("model-dropdown-head", "children"),
    [Input("yolox_s", "n_clicks"), Input("yolox_m", "n_clicks"), Input("yolox_l", "n_clicks")],
)


def update_label(n1, n2, n3):
    id_lookup = {"yolox_s": "YOLOX S", "yolox_m": "YOLOX M", "yolox_l": "YOLOX L"}

    ctx = dash.callback_context
    if (n1 is None and n2 is None and n3 is None) or not ctx.triggered:
        return "YOLOX S", "Detection Model Selected :: " + "YOLOX S"

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    # instantiate Tracker
    return id_lookup[button_id], "Detection Model Selected :: " + id_lookup[button_id]


modelmapping = {
    'YOLOX S': {'Name': 'yolox-s', 'path': 'weights/yolox_s.pth'},
    'YOLOX M': {'Name': 'yolox-m', 'path': 'weights/yolox_m.pth'},
    'YOLOX L': {'Name': 'yolox-l', 'path': 'weights/yolox_l.pth'},
}


@app.callback(output=[Output("update_tracker", "children")],
              inputs=[Input('run', "n_clicks")],
              state=[State("model-dropdown", "label")]

              )
def retrack(n_clicks, model_name):
    global tracker;
    global Main;
    if n_clicks:
        tracker = Tracker(filter_classes=None, model=modelmapping[model_name]['Name'],
                          ckpt=modelmapping[model_name]['path'])
        Main = []
    return None


"""
This Function Takes the input as n_interval and will execute by itself after a certain time
It outputs the figures 

"""


@app.callback([
    Output('live-graph1', 'figure'),
    Output('live-graph2', 'figure'),
    Output('cards', 'children'),
    Output('piefig', 'figure'),
    Output('dirfig', 'figure'),
    Output('sunfig', 'figure'),
    Output('speedfig', 'figure'),
    Output('infig', 'figure'),

],
    [
        Input('visual-update', 'n_intervals')
    ]
)
def update_visuals(n):
    global average_speed, previous_av_speed
    global vehiclestotal,vehiclestotal_prev,vehiclestotal_delta
    fig1 = go.FigureWidget()
    fig2 = go.FigureWidget()
    piefig = go.FigureWidget()
    dirfig = go.FigureWidget()
    sunfig = go.FigureWidget()
    speedfig = go.FigureWidget()
    infig = go.FigureWidget()

    # Dataset Creation a
    vehicleslastminute_prev = 0
    vehicleslastminute = 0
    vehicleslastminute_delta = 0
    vehiclestotal = 0
    df = pd.DataFrame(Main)

    if len(df) != 0:
        df1 = df.copy()
        df1['count'] = 1
        average_speed = int(df["Speed"].mean())
        # Database Transformations
        df = df.pivot_table(index=['Time'], columns='Category', aggfunc={'Category': "count"}).fillna(0)
        df.columns = df.columns.droplevel(0)
        df = df.reset_index()
        df.Time = pd.to_datetime(df.Time)
        columns = list(df.columns)
        columns.remove('Time')

        # Direction Datset
        dirdf = df1.groupby(['direction']).agg({"Speed": np.mean}).reset_index()

        # Sunburst Dataset
        df_all_trees = build_hierarchical_dataframe(df=df1, levels=["Category", 'direction'], value_column="count")

        # Speed Dataset
        df1 = df1.pivot_table(index=['Time'], columns='Category', aggfunc={'Speed': np.mean}).fillna(0)
        df1.columns = df1.columns.droplevel(0)
        df1 = df1.reset_index()
        df1.Time = pd.to_datetime(df1.Time)
        columns1 = list(df1.columns)
        columns1.remove('Time')

        # Speed Fig Add Scatter 
        for col in columns1:
            speedfig.add_scatter(name=col, x=df1['Time'], y=df1[col], fill="tonexty", line_shape="spline",
                                 line=dict(shape='linear', color='#3A416F', width=5))

        # Looping for adding scatter for each category
        values_sum = []
        for col in columns:
            vehiclestotal_prev = vehiclestotal
            fig1.add_scatter(name=col, x=df['Time'], y=df[col], fill='tonexty', showlegend=True, line_shape='spline',
                             line=dict(shape='linear', color='#3A416F', width=5))
            fig2.add_scatter(name=col, x=df['Time'], y=df[col].cumsum(), fill='tonexty',
                             showlegend=True,
                             line_shape='spline', line=dict(shape='linear', color='#CB0C9F', width=5))
            vehicleslastminute_prev = vehicleslastminute
            vehicleslastminute += df[col].values[-1]
            vehicleslastminute_delta = vehicleslastminute
            vehiclestotal += df[col].cumsum().values[-1]
            if vehiclestotal >=1:
                try:
                    #vehiclestotal_delta = 500
                    vehiclestotal_delta = ((vehiclestotal-vehiclestotal_prev)/vehiclestotal)*100
                except ZeroDivisionError:
                    vehiclestotal_delta  =  0

            values_sum.append(df[col].sum())

        piefig = px.pie(
            labels=columns, names=columns, values=values_sum, hole=0.5,
            title="Traffic Distribution - Vehicle Type",
            color_discrete_sequence=px.colors.sequential.Agsunset, opacity=0.85
        )

        dirfig = px.bar(dirdf, x="Speed", color="direction", orientation="h", hover_name="direction",
                        color_discrete_map={
                            "North": "rgba(188,75,128,0.8)",
                            "South": 'rgba(26,150,65,0.5)',
                            "East": 'rgba(64,167,216,0.8)',
                            "West": "rgba(218,165,32,0.8)"},
                        title="Average Speed Direction Flow"

                        )

        sunfig = go.FigureWidget(go.Sunburst(
            labels=df_all_trees['id'],
            parents=df_all_trees['parent'],
            values=df_all_trees['value'],
            branchvalues='total',
            textinfo='label+percent entry',
            opacity=0.85
        ))

    cards = [
        dbc.Col(create_card(Header="Vehicles Rate", Value=vehicleslastminute,Second_Value=0, cardcolor="primary",icon_thumb=iconz)),
        dbc.Col(create_card(Header="Total Vehicles", Value=vehiclestotal,Second_Value=f"{int(vehiclestotal_delta)}"+"%", cardcolor="info",icon_thumb=Traffic_icon)),
        dbc.Col(create_card(Header="FPS", Value=f"{int(fps)}", cardcolor="secondary",Second_Value=f"{int(fps_delta)}"+"%",icon_thumb=FPS_icon)),
        dbc.Col(create_card(Header="Resolution", Value=res, cardcolor="warning",Second_Value=stream,icon_thumb=cctv_icon)),

    ]

    stack = [dbc.Col(piefig), dbc.Col(sunfig), dbc.Col(dirfig)],  # stack of data

    infig = go.FigureWidget(
        go.Indicator(
            domain={'x': [0, 1], 'y': [0, 1]},
            value=average_speed,
            mode="gauge+number+delta",
            title={'text': ""},
            delta={'reference': previous_av_speed, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            gauge={'axis': {'range': [None, 50]},
                   'bar': {'color': "#CB0C9F"},
                   'bordercolor': "white",
                   'steps': [
                       {'range': [0, 15], 'color': 'rgba(0,0,0,0)'},
                       {'range': [15, 50], 'color': 'rgba(0,0,0,0)'}],
                   'threshold': {'line': {'color': "#624B9D", 'width': 4}, 'thickness': 0.75, 'value': 45}}

        )
    )

    # Updating the layout
    fig1 = update_layout1(figure=fig1, title='Traffic per Minute', margin=dict(t=0, b=00, r=00, l=0))
    fig2 = update_layout3(figure=fig2, title='Cumulative Traffic', GraphTick=1, margin=dict(t=20, b=20, r=20, l=20))
    speedfig = update_layout3(figure=speedfig, title='Average Speed Flow by Vehicle Type', GraphTick=20,
                              margin=dict(t=20, b=20, r=20, l=20))
    dirfig = update_layout2(figure=dirfig, title="Average Speed Direction Flow", margin=dict(t=40, b=10, r=10, l=10))
    sunfig = update_layout2(figure=sunfig, title="Traffic Direction Flow", margin=dict(t=30, b=10, r=60, l=10))
    infig = update_layout2(figure=infig, title="Average Speed Km/h", margin=dict(t=40, b=10, r=10, l=10))
    piefig = update_layout2(figure=piefig, title="Traffic Distribution - Vehicle Type",
                            margin=dict(t=30, b=10, r=60, l=10))

    return fig1, fig2, cards, piefig, dirfig, sunfig, speedfig, infig


# ===============================


progress = dbc.Progress(value=25)

app.layout = html.Div([sidebar,
                       # Input for all the updating visuals
                       dcc.Interval(id='visual-update', interval=1000, n_intervals=0),
                       dbc.Container([
                           dbc.Row([header, dbc.Col(children=[offcanvas])]),  # Header
                           dbc.Row(id="cards"),
                           dbc.Row([dbc.Col(videofeeds), dbc.Col(figure1)]),
                           dbc.Row([dbc.Col(figure2)]),  # VideoFeed and 2 Graphs
                           dbc.Row([dbc.Col(piefig), dbc.Col(sunfig), dbc.Col(infig)]),  # Header
                           dbc.Row([dbc.Col(speedfig), dbc.Col(dirfig)]),  # Header
                           dash_dangerously_set_inner_html.DangerouslySetInnerHTML('''
              <h1>Header</h1>
          '''),
                       ]),

                       dbc.Container(
                           progress
                       ),

                       ])

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)