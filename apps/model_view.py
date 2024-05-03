from dash import dcc
from dash import html

import pandas as pd
from data_processing.dag_processing import get_DAG_fig

import dash_ag_grid as dag
from dash import Dash, Input, Output, html, dcc, callback

import dash_bootstrap_components as dbc


def model_div(app, data, hidden=True):
    """
    Results section/page div block layout.
    :param app: DASH/Flask webapp object
    :param data: dataframe with simulation outputs/inputs
    :param hidden: True/False if block is visible
    :return: div block
    """

    rules_data_table = [{"Name": rule.name,
                         "Description": rule.rule_str,
                         "Local": rule.local,
                         "Shared": not(rule.local),
                         "graph": rule.get_DAG(),
                         "Add": "Add to \npolicy editor"
                         }
                        for rule in data["rules"]]

    rules_df = pd.DataFrame.from_records(rules_data_table)
    for i, r in rules_df.iterrows():
        fig = get_DAG_fig(data["rules"][i].get_DAG())
        rules_df.at[i, "graph"] = fig

    # local
    local_rules_data_table = [{"Name": rule.name,
                         "Description": rule.rule_str,
                         "graph": rule.get_DAG(),
                         "Add": "Add to \npolicy editor"
                         }
                        for rule in data["local"][0].rules]

    local_rules_df = pd.DataFrame.from_records(local_rules_data_table)
    for i, r in local_rules_df.iterrows():
        fig = get_DAG_fig(r["graph"])
        local_rules_df.at[i, "graph"] = fig

    # shared
    shared_rules_data_table = [{"Name": rule.name,
                         "Description": rule.rule_str,
                         "graph": rule.get_DAG(),
                         "Add": "Add to \npolicy editor"
                         }
                        for rule in data["shared"][0].rules]

    shared_rules_df = pd.DataFrame.from_records(shared_rules_data_table)
    for i, r in shared_rules_df.iterrows():
        fig = get_DAG_fig(r["graph"])
        shared_rules_df.at[i, "graph"] = fig


    rules_table_columns = [
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

    short_set_columns = [
        {
            "field": "Name",
        },
        {
            "field": "Description",
            "headerName": "Description",
        },
        {
            "field": "graph",
            "cellRenderer": "DCC_GraphClickData",
            "headerName": "DAG",
            "maxWidth": 900,
            "minWidth": 500,
        }
    ]
    table_div = html.Div(
        [
            dcc.Markdown("Causal Rules"),
            dag.AgGrid(
                id="custom-component-graph-grid",
                rowData=rules_df.to_dict("records"),
                columnSize="sizeToFit",
                columnDefs=rules_table_columns,
                defaultColDef={"filter": True, "minWidth": 60},
                dashGridOptions={"rowHeight": 200, "animateRows": False},
                style={"height": 600, 'display': 'none'} if hidden else {"height": 600, 'display': 'inline-block'}
            ),
        ]
    )

    local_table_div = html.Div(
        [
            dag.AgGrid(
                id="local_table_div",
                rowData=local_rules_df.to_dict("records"),
                columnSize="sizeToFit",
                columnDefs=short_set_columns,
                defaultColDef={"filter": True, "minWidth": 60},
                dashGridOptions={"rowHeight": 150, "animateRows": False},
                style={"height": 400, 'display': 'none'} if hidden else {"height": 400, 'display': 'inline-block'}
            ),
        ]
    )

    shared_table_div = html.Div(
        [
            dag.AgGrid(
                id="shared_table_div",
                rowData=shared_rules_df.to_dict("records"),
                columnSize="sizeToFit",
                columnDefs=short_set_columns,
                defaultColDef={"filter": True, "minWidth": 60},
                dashGridOptions={"rowHeight": 150, "animateRows": False},
                style={"height": 400, 'display': 'none'} if hidden else {"height": 400, 'display': 'inline-block'}
            ),
        ]
    )


    # nodes_and_edges_cache = get_nodes_and_edges(data)
    res_div = (html.Div(id="results_div",
                        children=[html.Div(id="local_div",
                                           children=[html.H2("Local causal DAG and rules:"),
                                                     dcc.Dropdown([r.name for r in data["local"]], 'Local sleep policy', id='local-dropdown'),
                                                     html.Div(id='local-output-container'),
                                                     html.I('Policy name: '),
                                                     dcc.Input(id='local_rule', type='string',
                                                               value=data["local"][0].name, debounce=True),
                                                     html.Br(),
                                                     html.I('Rules: '),
                                                     local_table_div,
                                                     html.Br(),
                                                     html.I('DAG:'),
                                                     dcc.Graph(id='local_dag',
                                                               figure=get_DAG_fig(data["local"][0].get_DAG())),
                                            ],
                                           style={'width': '50%', 'padding': '10px 10px 20px 20px',
                                                  'display': 'none'} if hidden else {'width': '50%',
                                                                                     'padding': '10px 10px 20px 20px', 'display': 'inline-block'}
                                           ),
                                  html.Div(id="shared_div",
                                           children=[html.H2("Shared causal DAG and rules:"),
                                                     dcc.Dropdown([r.name for r in data["shared"]], 'Shared policy on number of posts',
                                                                  id='shared-dropdown'),
                                                     html.Div(id='dd-output-container_shared'),
                                                     html.I('Policy name: '),
                                                     dcc.Input(id='shared_rule', type='string',
                                                               value=data["shared"][0].name, debounce=True),
                                                     html.Br(),
                                                     html.I('Rules: '),
                                                     shared_table_div,
                                                     html.Br(),
                                                     html.I('DAG:'),
                                                     dcc.Graph(id='shared_dag',
                                                               figure=get_DAG_fig(data["shared"][0].get_DAG())),
                                                     ],
                                           style={'width': '50%', 'padding': '10px 10px 20px 20px',
                                                  'display': 'none'} if hidden else {'width': '50%',
                                                                                     'padding': '10px 10px 20px 20px', 'display': 'inline-block'}
                                           ),
                                  table_div,
                                  ],
                       )
               )

    return res_div
