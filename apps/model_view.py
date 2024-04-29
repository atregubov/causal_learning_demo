from dash import dcc
from dash import html

import pandas as pd
from data_processing.dag_processing import get_tree

import dash_ag_grid as dag
from dash import Dash, Input, Output, html, dcc, callback
import plotly.express as px
import json

#import dash_mantine_components as dmc
#import dash_iconify


import dash_bootstrap_components as dbc



def model_div(app, data, hidden=True):
    """
    Results section/page div block layout.
    :param app: DASH/Flask webapp object
    :param data: dataframe with simulation outputs/inputs
    :param hidden: True/False if block is visible
    :return: div block
    """

    rules_data_table = [{"Name": "sleep_hours_rule",
                         "Description": "if sleep_hours in to_ban_range -> ban and if sleep_hours in not_to_ban_range -> no ban",
                         "Local": True,
                         "Shared": True,
                         "graph": "",
                         "Add": "Add"
                         },
                        {"Name": "total_number_of_posts",
                         "Description": "if total_number_of_posts in to_ban_range -> ban and if total_number_of_posts in not_to_ban_range -> no ban",
                         "Local": True,
                         "Shared": False,
                         "graph": "",
                         "Add": "Add"
                         },
                        ]

    rules_df = pd.DataFrame.from_records(rules_data_table)
    for i, r in rules_df.iterrows():
        fig = get_tree() if r["Name"] == "sleep_hours_rule" else get_tree(2)
        rules_df.at[i, "graph"] = fig

    columnDefs = [
        {
            "field": "Name",
        },
        {
            "field": "Description",
            "headerName": "Description",
        },
        {
            "field": "Local",
            "headerName": "Local",
        },
        {
            "field": "Shared",
            "headerName": "Shared",
        },
        {
            "field": "graph",
            "cellRenderer": "DCC_GraphClickData",
            "headerName": "DAG",
            "maxWidth": 900,
            "minWidth": 500,
        },
        {
            "field": "Add",
            "cellRenderer": "DBC_Button_Simple",
            "cellRendererParams": {"color": "success"},
        },
    ]

    table_div = html.Div(
        [
            dcc.Markdown("Causal Rules"),
            dag.AgGrid(
                id="custom-component-graph-grid",
                rowData=rules_df.to_dict("records"),
                columnSize="sizeToFit",
                columnDefs=columnDefs,
                defaultColDef={"filter": True, "minWidth": 60},
                dashGridOptions={"rowHeight": 200, "animateRows": False},
                style={"height": 600, 'display': 'none'} if hidden else {"height": 600, 'display': 'inline-block'}
            ),
            #html.Div(id="custom-component-graph-output"),
        ]
    )


    # nodes_and_edges_cache = get_nodes_and_edges(data)
    res_div = (html.Div(id="results_div",
                        children=[html.Div(id="local_div",
                                           children=[html.H2("Local causal DAG and rules:"),
                                                     html.I('Rule:'),
                                                     dcc.Input(id='local_rule', type='string',
                                                               value='narrative_ratio > 0.1', debounce=True),
                                                     dcc.Graph(id='local_dag',
                                                               figure=get_tree()),
                                            ],
                                           style={'width': '40%', 'padding': '10px 10px 20px 20px',
                                                  'display': 'none'} if hidden else {'width': '40%',
                                                                                     'padding': '10px 10px 20px 20px', 'display': 'inline-block'}
                                           ),
                                  html.Div(id="shared_div",
                                           children=[html.H2("Shared causal DAG and rules:"),
                                                     html.I('Rule:'),
                                                     dcc.Input(id='shared_rule', type='string',
                                                               value='narrative_ratio > 0.1', debounce=True),
                                                     dcc.Graph(id='shared_dag',
                                                               figure=get_tree()), #user_graph(data, 4, nodes_and_edges_cache)),
                                                     ],
                                           style={'width': '40%', 'padding': '10px 10px 20px 20px',
                                                  'display': 'none'} if hidden else {'width': '40%',
                                                                                     'padding': '10px 10px 20px 20px', 'display': 'inline-block'}
                                           ),
                                  table_div,
                                  ],
                       )
               )

    return res_div
