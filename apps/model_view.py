from matplotlib import pyplot as plt
from dash import dcc
from dash import html
import random
from sklearn import tree
import math
from collections import defaultdict
import plotly.graph_objects as go
import networkx as nx
import pandas as pd
import dash_ag_grid as dag
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from dash import Dash, Input, Output, html, dcc, callback, State, dash_table

import dash_bootstrap_components as dbc


def model_div(app, data, username, hidden=True):
    """
    Results section/page div block layout.
    :param app: DASH/Flask webapp object
    :param data: dataframe with simulation outputs/inputs
    :param hidden: True/False if block is visible
    :return: div block
    """
    if username is None or username not in data:
        username = "s1"
    ####################################################################################################################
    # Prepare data table from data object
    ####################################################################################################################
    # all rules table
    rules_data_table = [{"Name": rule.name,
                         "Description": rule.rule_str + "\n\n >\n\n**Evaluation on \"S1 local (fit to historical data)\" thresholds:**"
                                                        f"\n- Precision: {random.uniform(0.06, 0.96):.2f}\n- Accuracy: {random.uniform(0.06, 0.96):.2f}"
                                                        f"\n- F1 score: {random.uniform(0.05, 0.96):.2f}\n- Recall: {random.uniform(0.03, 0.96):.2f}",
                         "Shared": rule.shared_by if rule.name != "sleep_hours" else "Me, S2, S3",
                         "Thresholds (fit data)": "**Global thresholds:**\n- S1 local (fit to historical data)"
                                                  "\n- shared by S2\n- shared by S3\n\n**Thresholds from policies:**"
                                                  "\n- Local: S1: number of posts and sleep hours rule"
                                                  "\n- Shared (from site 2): sleep hours bab policy",
                         "graph": rule.get_DAG(),
                         "Add to editor": "Add to \npolicy editor"
                         }
                        for rule in data[username]["rules"]]
    rules_df = pd.DataFrame.from_records(rules_data_table)
    for i, r in rules_df.iterrows():
        fig = get_DAG_fig(data[username]["rules"][i].get_DAG(), show_edge_labels=False)
        rules_df.at[i, "graph"] = fig

    # local policy view table
    local_rules_data_table = [{"Name": rule.name,
                               "Description": rule.rule_str,
                               "graph": rule.get_DAG(),
                               "Thresholds (fit data)": "***Local: S1: number of posts and sleep hours rule***" if rule.name != "sleep_and_posts" else "**None**",
                               "Remove": "Remove",
                         }
                        for rule in data[username]["local"][0].rules]
    local_rules_df = pd.DataFrame.from_records(local_rules_data_table)
    for i, r in local_rules_df.iterrows():
        fig = get_DAG_fig(r["graph"], show_edge_labels=False, node_size=40, label_position="top center")
        local_rules_df.at[i, "graph"] = fig

    # shared policy view table
    shared_rules_data_table = [{"Name": rule.name,
                                "Description": rule.rule_str,
                                "graph": rule.get_DAG(),
                                "Thresholds (fit data)": "**Local: S1: number of posts and sleep hours rule**" if rule.name != "sleep_and_posts" else "**None**",
                                "Copy": "<-Copy",
                                }
                               for rule in data["shared"][0].rules]
    shared_rules_df = pd.DataFrame.from_records(shared_rules_data_table)
    for i, r in shared_rules_df.iterrows():
        fig = get_DAG_fig(r["graph"], show_edge_labels=False, node_size=40, label_position="top center")
        shared_rules_df.at[i, "graph"] = fig

    # policy editor panel
    def get_editor_rules(data_):
        editor_rules_data_table = [{"Name": rule.name,
                                    "Description": rule.rule_str,
                                    "Thresholds (fit data)": "**Global thresholds:**\n- S1 local (fit to historical data)"
                                                  "\n- shared by S2\n- shared by S3\n\n**Thresholds from policies:**"
                                                  "\n- ***Local: S1: number of posts and sleep hours rule (current)\****"
                                                  "\n- Shared (from site 2): sleep hours bab policy",
                                    "graph": rule.get_DAG(),
                                    "Remove": "Remove",
                                    }
                                   for rule in data_[username]["editor"].rules]
        editor_rules_df = pd.DataFrame.from_records(editor_rules_data_table)
        for i, r in editor_rules_df.iterrows():
            fig = get_DAG_fig(r["graph"], show_edge_labels=False, node_size=40, label_position="top center")
            editor_rules_df.at[i, "graph"] = fig

        return editor_rules_df.to_dict("records")

    ####################################################################################################################
    # Prepare table formating for panels
    ####################################################################################################################
    # all rules table
    rules_table_columns = [
        {
            "field": "Name",
            "resizable": True,
            "cellStyle": {"wordBreak": "normal"},
            "wrapText": True,
            "autoHeight": True,
        },
        {
            "field": "Description",
            "headerName": "Description",
            "resizable": True,
            "cellStyle": {"wordBreak": "normal", "line-height": "normal"},
            'cellDataType': 'text',
            "cellRenderer": "markdown",
            "wrapText": True,
            "autoHeight": True,
            "autoWidth": True,
            "minWidth": 400,
        },
        {
            "field": "Shared",
            "headerName": "Shared by",
            "resizable": True,
            "maxWidth": 150,
            "minWidth": 90,
        },
        {
            "field": "Thresholds (fit data)",
            "headerName": "Available thresholds",
            "resizable": True,
            "cellStyle": {"wordBreak": "normal", "line-height": "normal"},
            'cellDataType': 'text',
            "cellRenderer": "markdown",
            "wrapText": True,
            "autoHeight": True,
            "autoWidth": True,
            "minWidth": 300,
        },
        {
            "field": "graph",
            "cellRenderer": "DCC_GraphClickData",
            "headerName": "DAG",
            "minWidth": 500,
            "autoWidth": True,
            "resizable": True,
        },
        {
            "field": "Add to editor",
            "cellRenderer": "DBC_Button_Simple",
            "cellRendererParams": {"color": "success"},
            "minWidth": 200,
            "maxWidth": 250,
            "resizable": True,
        },
    ]

    # shared and local policy view table uses short_set_columns
    shared_rules_columns = [
        {
            "field": "Copy",
            "headerName": "Copy to local",
            "cellRenderer": "DBC_Button_Simple",
            "cellRendererParams": {"color": "success"},
            "minWidth": 70,
        },
        {
            "field": "Name",
            "resizable": True,
            "cellStyle": {"wordBreak": "normal"},
            "wrapText": True,
            "autoHeight": True,
        },
        {
            "field": "Description",
            "headerName": "Description",
            "resizable": True,
            "cellStyle": {"wordBreak": "normal", "line-height": "normal"},
            'cellDataType': 'text',
            "cellRenderer": "markdown",
            "wrapText": True,
            "autoHeight": True,
            "autoWidth": True,
            "minWidth": 200,
        },
        {
            "field": "graph",
            "cellRenderer": "DCC_GraphClickData",
            "headerName": "DAG",
            "maxWidth": 900,
            "minWidth": 300,
            # "autoHeight": True,
            # "autoWidth": True,
        },
        {
            "field": "Thresholds (fit data)",
            "headerName": "Threshold (fit data)",
            "resizable": True,
            "cellStyle": {"wordBreak": "normal", "line-height": "normal"},
            'cellDataType': 'text',
            "cellRenderer": "markdown",
            "wrapText": True,
            "autoHeight": True,
            "autoWidth": True,
            "minWidth": 100,
        },
    ]

    local_rules_columns = [
        {
            "field": "Name",
            "resizable": True,
            "cellStyle": {"wordBreak": "normal"},
            "wrapText": True,
            "autoHeight": True,
        },
        {
            "field": "Description",
            "headerName": "Description",
            "resizable": True,
            "cellStyle": {"wordBreak": "normal", "line-height": "normal"},
            'cellDataType': 'text',
            "cellRenderer": "markdown",
            "wrapText": True,
            "autoHeight": True,
            "autoWidth": True,
            "minWidth": 200,
        },
        {
            "field": "graph",
            "cellRenderer": "DCC_GraphClickData",
            "headerName": "DAG",
            "maxWidth": 900,
            "minWidth": 300,
            # "autoHeight": True,
            # "autoWidth": True,
        },
        {
            "field": "Thresholds (fit data)",
            "headerName": "Threshold (fit data)",
            "resizable": True,
            "cellStyle": {"wordBreak": "normal", "line-height": "normal"},
            'cellDataType': 'text',
            "cellRenderer": "markdown",
            "wrapText": True,
            "autoHeight": True,
            "autoWidth": True,
            "minWidth": 100,
        },
        {
            "field": "Remove",
            "cellRenderer": "DBC_Button_Simple",
            "cellRendererParams": {"color": "success"},
            "minWidth": 70,
        },
    ]

    # editor panel table with rules
    editor_columns = [
        {
            "field": "Name",
            "resizable": True,
            "cellStyle": {"wordBreak": "normal"},
            "wrapText": True,
            "autoHeight": True,
        },
        {
            "field": "Description",
            "headerName": "Description",
            "resizable": True,
            "cellStyle": {"wordBreak": "normal", "line-height": "normal"},
            "wrapText": True,
            "autoHeight": True,
        },
        {
            "field": "Thresholds (fit data)",
            "headerName": "Thresholds (fit data)",
            "resizable": True,
            "cellStyle": {"wordBreak": "normal", "line-height": "normal"},
            'cellDataType': 'text',
            "cellRenderer": "markdown",
            "wrapText": True,
            "autoHeight": True,
            "autoWidth": True,
            "minWidth": 100,
         },
        {
            "field": "graph",
            "cellRenderer": "DCC_GraphClickData",
            "headerName": "DAG",
            "maxWidth": 900,
            "minWidth": 400,
        },
        {
            "field": "Remove",
            "cellRenderer": "DBC_Button_Simple",
            "cellRendererParams": {"color": "success"},
            "minWidth": 70,
            "maxWidth": 110,

        },
    ]
    # Evaluation metrics summary
    eval_metrics_columns = [
        {
            "field": "Metric",
            "resizable": True,
            "cellStyle": {"wordBreak": "normal"},
            "minWidth": 300,
            # "wrapText": True,
            # "autoWidth": True,
            "autoHeight": True,
        },
        {
            "field": "Value",
            "headerName": "Value",
            "resizable": True,
            "cellStyle": {"wordBreak": "normal"},
            # "wrapText": True,
            "autoHeight": True,
        }]

    eval_metrics_columns_small = [
        {
            "field": "Metric",
            "resizable": True,
            "cellStyle": {"wordBreak": "normal"},
            "minWidth": 100,
            # "wrapText": True,
            # "autoWidth": True,
            "autoHeight": True,
        },
        {
            "field": "Value",
            "headerName": "Value",
            "resizable": True,
            "cellStyle": {"wordBreak": "normal"},
            # "wrapText": True,
            "autoHeight": True,
        }]
    ####################################################################################################################
    # Prepare div blocks for panels
    ####################################################################################################################
    rules_table_div = html.Div(
        [
            dag.AgGrid(
                id="rules_table_div_id",
                rowData=rules_df.to_dict("records"),
                columnSize="sizeToFit",
                columnDefs=rules_table_columns,
                defaultColDef={"filter": True, "minWidth": 60},
                dashGridOptions={"rowHeight": 150, "animateRows": False},
                style={"height": 800, 'display': 'inline-block', 'resize': 'both', 'overflow': 'auto'},

            ),
        ]
    )
    local_table_div = html.Div(
        [
            dag.AgGrid(
                id="local_table_div",
                rowData=local_rules_df.to_dict("records"),
                columnSize="sizeToFit",
                columnDefs=local_rules_columns,
                defaultColDef={"filter": True, "minWidth": 60},
                dashGridOptions={"rowHeight": 150, "animateRows": False},
                style={"height": 400, 'display': 'inline-block', 'resize': 'both', 'overflow': 'auto'}
            ),
        ]
    )
    shared_table_div = html.Div(
        [
            dag.AgGrid(
                id="shared_table_div",
                rowData=shared_rules_df.to_dict("records"),
                columnSize="sizeToFit",
                columnDefs=shared_rules_columns,
                defaultColDef={"filter": True, "minWidth": 60},
                dashGridOptions={"rowHeight": 150, "animateRows": False},
                style={"height": 400, 'display': 'inline-block', 'resize': 'both', 'overflow': 'auto'}
            ),
        ]
    )
    editor_table_div = html.Div(
        [
            dag.AgGrid(
                id="editor_table_div",
                rowData=get_editor_rules(data),
                columnSize="sizeToFit",
                columnDefs=editor_columns,
                defaultColDef={"filter": True, "minWidth": 60},
                dashGridOptions={"rowHeight": 150, "animateRows": False},
                style={"height": 400, 'display': 'inline-block', 'resize': 'both', 'overflow': 'auto'}
            ),
        ]
    )

    thresholds = [t for t in data[username]["thresholds"].keys()]
    thresholds.extend(list(data["thresholds"].keys()))
    # div with all panels together
    panels_div = html.Div(id="panels_div",
                          style={'display': 'none'} if hidden else {'display': 'block'},
                          children=[
                              dcc.Tabs([
                                  dcc.Tab(label='Local & Shared Policies', children=[
                                        html.Div(id="local_div",
                                                 children=[html.H2("Local causal DAG and rules"),
                                                           dcc.Dropdown([r.name for r in data[username]["local"]],
                                                                        data[username]["local"][0].name,
                                                                        id='local-dropdown',
                                                                        style={'width': '80%',
                                                                               'display': 'inline-block'}
                                                                        ),
                                                           html.Button('Edit', id='edit_local_btn', n_clicks=0,
                                                                       style={'width': '20%',
                                                                              'padding': '5px 5px 5px 5px'}),
                                                           html.Br(),
                                                           html.I('Policy name: '),
                                                           html.B(data[username]["local"][0].name,
                                                                  id='local_rule_name'),
                                                           html.Br(),
                                                           # html.I('DAG:'),
                                                           dcc.Checklist(
                                                               ['Show rule name on DAG edges'],
                                                               [],
                                                               id="show_rule_names_local",
                                                               inline=True
                                                           ),
                                                           dcc.Graph(id='local_dag',
                                                                     figure=get_DAG_fig(
                                                                         data[username]["local"][0].get_DAG(),
                                                                         show_edge_labels=False),
                                                                     style={'width': '70%',
                                                                            'display': 'inline-block'}
                                                                     ),
                                                           dag.AgGrid(id="eval_table_id_local",
                                                                      rowData=[
                                                                          {
                                                                              "Metric": "Precision",
                                                                              "Value": "0.52"},
                                                                          {
                                                                              "Metric": "Accuracy",
                                                                              "Value": "0.31"},
                                                                          {
                                                                              "Metric": "F1 score",
                                                                              "Value": "0.47"},
                                                                          {
                                                                              "Metric": "Recall",
                                                                              "Value": "0.54"}
                                                                      ],
                                                                      columnSize="sizeToFit",
                                                                      columnDefs=eval_metrics_columns_small,
                                                                      defaultColDef={
                                                                          "filter": False,
                                                                          "minWidth": 60},
                                                                      style={
                                                                          'width': '30%',
                                                                          "height": "450px",
                                                                          'display': 'inline-block',
                                                                          'overflow': 'auto'},
                                                                      ),
                                                           html.Br(),
                                                           html.I('Rules: '),
                                                           local_table_div,
                                                           ],
                                                 style={'width': '50%', 'padding': '10px 10px 20px 20px',
                                                        'display': 'inline-block'}
                                                 ),
                                        html.Div(id="shared_div",
                                                 children=[html.H2("Shared causal DAG and rules"),
                                                           dcc.Dropdown([r.name for r in data["shared"]],
                                                                        data["shared"][0].name, id='shared-dropdown',
                                                                        style={'width': '80%',
                                                                               'display': 'inline-block'}),
                                                           html.Button('Edit', id='edit_shared_btn', n_clicks=0,
                                                                       style={'width': '20%',
                                                                              'padding': '5px 5px 5px 5px'}),
                                                           html.Br(),
                                                           html.I('Policy name: '),
                                                           html.B(data["shared"][0].name, id='shared_rule_name'),
                                                           html.Br(),
                                                           # html.I('DAG:'),
                                                           dcc.Checklist(
                                                               ['Show rule name on DAG edges'],
                                                               [],
                                                               id="show_rule_names_shared",
                                                               inline=True
                                                           ),
                                                           dcc.Graph(id='shared_dag',
                                                                     figure=get_DAG_fig(data["shared"][0].get_DAG(),
                                                                                        show_edge_labels=False),
                                                                     style={'width': '70%',
                                                                            'display': 'inline-block'}
                                                                     ),
                                                           dag.AgGrid(id="eval_table_id_shared",
                                                                      rowData=[
                                                                          {
                                                                              "Metric": "Precision",
                                                                              "Value": "0.52"},
                                                                          {
                                                                              "Metric": "Accuracy",
                                                                              "Value": "0.78"},
                                                                          {
                                                                              "Metric": "F1 score",
                                                                              "Value": "0.61"},
                                                                          {
                                                                              "Metric": "Recall",
                                                                              "Value": "0.42"}
                                                                      ],
                                                                      columnSize="sizeToFit",
                                                                      columnDefs=eval_metrics_columns_small,
                                                                      defaultColDef={
                                                                          "filter": False,
                                                                          "minWidth": 60},
                                                                      style={
                                                                          'width': '30%',
                                                                          "height": "450px",
                                                                          'display': 'inline-block',
                                                                          'overflow': 'auto'},
                                                                      ),
                                                           html.Br(),
                                                           html.I('Rules: '),
                                                           shared_table_div,

                                                           ],
                                                 style={'width': '50%', 'padding': '10px 10px 20px 20px',
                                                        'display': 'inline-block'}
                                                 ),
                                    ]),
                                    dcc.Tab(label='Rules Library', children=[
                                        html.Div(id="all_rules_div",
                                                 children=[html.H2("Rules", style={'display': 'inline-block'}),
                                                           #html.Br(),
                                                           html.Button('Share local thresholds (for all rules)',
                                                                       id='share_thresholds_btn',
                                                                       n_clicks=0,
                                                                       style={'padding': '10px 10px 10px 10px',
                                                                              'display': 'inline-block',
                                                                              "float": "right"}),
                                                           html.Br(),
                                                           rules_table_div,
                                                           ],
                                                 style={'width': '100%', 'padding': '10px 10px 20px 20px',
                                                        'display': 'inline-block'}
                                                 ),
                                    ]),
                                    dcc.Tab(label='Policy Editor', children=[
                                        html.Div(id="editor_div",
                                                 children=[
                                                           html.H2("Editor"),
                                                           html.Div(id="editor_dag_div",
                                                                    children=[html.I('Policy name: '),
                                                                              dcc.Input(id='editor_policy_name',
                                                                                        type='string',
                                                                                        value=data[username][
                                                                                            "editor"].name,
                                                                                        debounce=True,
                                                                                        style={'width': '100%'}),
                                                                              html.Div(id='editor_notification',
                                                                                       children=''),
                                                                              html.Br(),
                                                                              html.Button('Save', id='save_btn',
                                                                                          n_clicks=0,
                                                                                          style={
                                                                                              'padding': '10px 10px 10px 10px'}),
                                                                              html.Button('Share this policy',
                                                                                          id='share_dag_btn',
                                                                                          n_clicks=0,
                                                                                          style={
                                                                                              'padding': '10px 10px 10px 10px'}),
                                                                              html.Br(),
                                                                              # html.I('DAG: '),
                                                                              dcc.Checklist(
                                                                                  ['Show rule name on edges'],
                                                                                  [],
                                                                                  inline=True,
                                                                                  id="show_rule_names_editor",
                                                                              ),
                                                                              dcc.Graph(id='editor_dag',
                                                                                        figure=get_DAG_fig(
                                                                                            data[username][
                                                                                                "editor"].get_DAG(),
                                                                                            show_edge_labels=False),
                                                                                        style={"height": 400,
                                                                                               'width': '100%',
                                                                                               'display': 'inline-block',
                                                                                               'resize': 'both',
                                                                                               'overflow': 'auto'},
                                                                                        ),
                                                                              ],
                                                                    style={'width': '40%', 'padding': '0px 0px 0px 0px',
                                                                           'display': 'inline-block'}
                                                                    ),
                                                           html.Div(id="editor_rules_div",
                                                                    children=[html.I('Rules: '),
                                                                              editor_table_div,
                                                                              ],
                                                                    style={'width': '60%',
                                                                           'padding': '0px 0px 0px 10px',
                                                                           'display': 'inline-block'}
                                                                    ),
                                                           html.Div(id="evaluation_div_id",
                                                                    children=[html.H2('Evaluation'),
                                                                              html.Div(id="evaluation_pred",
                                                                                       children=[html.Div(id="evaluation_inner_div",
                                                                                                          children=[
                                                                                                                      html.Button(
                                                                                                                          'Evaluate',
                                                                                                                          id='evaluate_btn',
                                                                                                                          n_clicks=0,
                                                                                                                          style={
                                                                                                                              'padding': '10px 10px 10px 10px',
                                                                                                                              'display': 'inline-block'}),
                                                                                                                      html.Br(),
                                                                                                                      dag.AgGrid(id="eval_table_id",
                                                                                                                                 rowData=[{
                                                                                                                                              "Metric": "Total number of predicted bans on historical data",
                                                                                                                                              "Value": ""},
                                                                                                                                          {
                                                                                                                                              "Metric": "Total number of actual bans (historical data)",
                                                                                                                                              "Value": ""},
                                                                                                                                          {
                                                                                                                                              "Metric": "Precision",
                                                                                                                                              "Value": ""},
                                                                                                                                          {
                                                                                                                                              "Metric": "Accuracy",
                                                                                                                                              "Value": ""},
                                                                                                                                          {
                                                                                                                                              "Metric": "F1 score",
                                                                                                                                              "Value": ""},
                                                                                                                                          {
                                                                                                                                              "Metric": "Recall",
                                                                                                                                              "Value": ""}
                                                                                                                                 ],
                                                                                                                                 columnSize="sizeToFit",
                                                                                                                                 columnDefs=eval_metrics_columns,
                                                                                                                                 defaultColDef={
                                                                                                                                      "filter": False,
                                                                                                                                      "minWidth": 60},
                                                                                                                                 style={
                                                                                                                                      "height": "400px",
                                                                                                                                      'display': 'inline-block',
                                                                                                                                      'overflow': 'auto'},
                                                                                                                      ),
                                                                                                                      html.Br(),
                                                                                                                    ],
                                                                                                          style={'width': '100%',
                                                                                                                 'padding': '0px 0px 0px 0px',
                                                                                                                 'display': 'inline-block'}
                                                                                                          ),
                                                                                                 ],
                                                                                       style={'width': '40%',
                                                                                                'padding': '0px 0px 0px 0px',
                                                                                                'display': 'inline-block'}
                                                                                 ),
                                                                        html.Div(id="evaluation_hist",
                                                                                 children=[
                                                                                     dcc.Tabs([
                                                                                         dcc.Tab(
                                                                                             label='Triggered rules',
                                                                                             children=[
                                                                                                 dcc.Graph(
                                                                                                     id="evaluation_figure_id",
                                                                                                     figure=get_triggered_rules_figure(
                                                                                                         data,
                                                                                                         username,
                                                                                                         thresholds[0])),
                                                                                             ]),
                                                                                         dcc.Tab(
                                                                                             label='Feature importance',
                                                                                             children=[
                                                                                                 dcc.Graph(
                                                                                                     id="feature_importance_id",
                                                                                                     figure=get_triggered_rules_figure(
                                                                                                         data,
                                                                                                         username,
                                                                                                         thresholds[0])),
                                                                                             ]),
                                                                                     ]
                                                                                     )],
                                                                                 style = {'width': '60%',
                                                                                       'padding': '10px 10px 10px 10px',
                                                                                       'display': 'inline-block',
                                                                                       'float': "right"
                                                                                          }
                                                                                 )
                                                                        ],
                                                                    style={'width': '100%',
                                                                     'padding': '0px 0px 0px 0px',
                                                                     'display': 'inline-block'}

                                                              ),
                                                           ],
                                                 style={'width': '100%', 'padding': '10px 10px 20px 20px',
                                                        'display': 'inline-block'}
                                                 ),
                                    ]),
                              ])
                          ])

    @app.callback(
        Output('editor_notification', 'children'),
        Output('evaluation_div_id', 'style'),
        Output('evaluation_figure_id', 'figure'),
        Output('feature_importance_id', 'figure'),
        Output('eval_table_id', 'rowData'),

        Input('evaluate_btn', 'n_clicks'),
        prevent_initial_call=True
    )
    def evaluate_notification(n_clicks):
        fit_data_str = "local (S1): thresholds fit to historical data"
        msg = f"*Policy evaluated."
        style = {'width': '100%', 'padding': '0px 0px 0px 0px', 'display': 'inline-block'}
        fig, bans_count, bans_count_gt = get_triggered_rules_figure(data, username, fit_data_str)
        fig_pred, _ = get_pred_triggered_rules_figure(data, username, fit_data_str)

        precision, recall, fscore, support = precision_recall_fscore_support(bans_count_gt, bans_count, average="macro")
        accuracy = accuracy_score(bans_count_gt, bans_count)

        metrics = [{
            "Metric": "Total number of predicted bans on historical data",
            "Value": sum(bans_count)
        },{
            "Metric": "Total number of actual bans (historical data)",
            "Value": sum(bans_count_gt)
        },{
            "Metric": "Precision",
            "Value": round(precision, 2)
        },{
            "Metric": "Accuracy",
            "Value": round(accuracy, 2)
        },{
            "Metric": "F1 score",
            "Value": round(fscore, 2)
        },{
            "Metric": "Recall",
            "Value": round(recall, 2)
        }]

        return msg, style,  fig, fig_pred, metrics

    @app.callback(
        Output('editor_notification', 'children'),
        Input('save_btn', 'n_clicks'),
        State('editor_policy_name', 'value'),
        prevent_initial_call=True
    )
    def saved_notification(n_clicks, value):
        return f"*Policy \"{value}\" saved."

    @app.callback(
        Output('editor_notification', 'children'),
        Input('share_dag_btn', 'n_clicks'),
        State('editor_policy_name', 'value'),
        prevent_initial_call=True
    )
    def shared_notification(n_clicks, value):
        return f"*Policy \"{value}\" shared."

    @app.callback(
        Output('editor_dag', 'figure'),
        Input('show_rule_names_editor', 'value'),
        prevent_initial_call=True
    )
    def update_editor_dag(checked_vals):
        return get_DAG_fig(data[username]["editor"].get_DAG(), show_edge_labels=False if len(checked_vals) == 0 else True)

    @app.callback(
        Output('shared_dag', 'figure'),
        Input('show_rule_names_shared', 'value'),
        prevent_initial_call=True
    )
    def update_shared_dag(checked_vals):
        return get_DAG_fig(data["shared"][0].get_DAG(), show_edge_labels=False if len(checked_vals) == 0 else True)

    @app.callback(
        Output('local_dag', 'figure'),
        Input('show_rule_names_local', 'value'),
        prevent_initial_call=True
    )
    def update_local_dag(checked_vals):
        return get_DAG_fig(data[username]["local"][0].get_DAG(), show_edge_labels=False if len(checked_vals) == 0 else True)

    # REMOVE button
    @app.callback(
        Output('editor_dag', 'figure'),
        Output('editor_table_div', 'rowData'),
        Output('evaluation_figure_id', 'figure'),
        Input("editor_table_div", "cellRendererData"),
        State('show_rule_names_editor', 'value'),
        prevent_initial_call=True
    )
    def update_editor_DAG_after_removal(idx_to_remove, checked_vals):
        fit_data_version = list(data[username]["thresholds"].keys())[0]
        if idx_to_remove is not None:
            del data[username]["editor"].rules[idx_to_remove["rowIndex"]]
        dag_fig = get_DAG_fig(data[username]["editor"].get_DAG(), show_edge_labels=False if len(checked_vals) == 0 else True)
        fig = get_triggered_rules_figure(data, username, fit_data_version)
        return dag_fig, get_editor_rules(data), fig

    @app.callback(
        Output('editor_dag', 'figure'),
        Output('editor_table_div', 'rowData'),
        Output('evaluation_figure_id', 'figure'),
        Input("rules_table_div_id", "cellRendererData"),
        State('show_rule_names_editor', 'value'),
        prevent_initial_call=True
    )
    def update_editor_DAG_after_adding(idx_to_add, checked_vals):
        fit_data_version = list(data[username]["thresholds"].keys())[0]
        if idx_to_add is not None:
            data[username]["editor"].rules.append( data[username]["rules"][idx_to_add["rowIndex"]])
        dag_fig = get_DAG_fig(data[username]["editor"].get_DAG(), show_edge_labels=False if len(checked_vals) == 0 else True)
        fig = get_triggered_rules_figure(data, username, fit_data_version)
        return dag_fig, get_editor_rules(data), fig

    return panels_div


def hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):
    '''If there is a cycle that is reachable from root, then result will not be a hierarchy.

       G: the graph
       root: the root node of current branch
       width: horizontal space allocated for this branch - avoids overlap with other branches
       vert_gap: gap between levels of hierarchy
       vert_loc: vertical location of root
       xcenter: horizontal location of root
    '''

    def reverse_edges(G):
        original_edges = list(G.edges(data=True))
        for a, b, edge_data in original_edges:
            G.remove_edge(a, b)
            G.add_edge(b, a, label=edge_data["label"], color=edge_data["color"])
    def h_recur(G, root, width=1., vert_gap = 0.2, vert_loc = 0.0, xcenter = 0.5,
                  pos = None, parent = None, parsed = [] ):
        if(root not in parsed):
            parsed.append(root)
            if pos == None:
                pos = {root:(xcenter,vert_loc)}
            else:
                pos[root] = (xcenter, vert_loc)
            neighbors = list(G.neighbors(root))
            if parent != None and parent in neighbors:
                neighbors.remove(parent)
            if len(neighbors) != 0:
                dx = width/len(neighbors)
                nextx = xcenter - width/2 - dx/2
                for neighbor in neighbors:
                    nextx += dx
                    pos = h_recur(G, neighbor, width = dx, vert_gap = vert_gap,
                                        vert_loc = vert_loc-vert_gap, xcenter=nextx, pos=pos,
                                        parent = root, parsed = parsed)
        return pos

    reverse_edges(G)
    pos = h_recur(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5)
    reverse_edges(G)
    return pos


def get_DAG_fig(G, show_edge_labels=True, node_size=60, label_position="top center"):
    lay = nx.layout.circular_layout(G)
    if len(list(G.nodes)) > 2 and "ban" in G.nodes:
        lay = hierarchy_pos(G, "ban")
    if len(list(G.nodes)) == 2 and "ban" in G.nodes:
        second_node = [n for n in G.nodes if n != "ban"][0]
        lay = {"ban": [0, 0], second_node: [1, 0]}
    if len(list(G.nodes)) == 2 and "ban" not in G.nodes:
        first_node = [n for n in G.nodes][0]
        second_node = [n for n in G.nodes][1]
        lay = {first_node: [0, 0], second_node: [1, 0]}
    position = {node: [lay[node][0], lay[node][1]] for node in G.nodes}

    edge_x = []
    edge_y = []
    edge_labels = list()
    edge_colors = list()
    for n0, n1, edge_data in G.edges(data=True):
        x0, y0 = position[n0]
        x1, y1 = position[n1]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_labels.append(edge_data["label"] if len(edge_data["label"]) < 25 else edge_data["label"][:20]+"..." if "label" in edge_data else "no label")
        edge_colors.append(edge_data["color"])

    node_x = []
    node_y = []
    node_colors = []
    for node, node_data in G.nodes(data=True):
        x, y = position[node]
        node_x.append(x)
        node_y.append(y)
        node_colors.append(node_data["color"])

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        textposition=label_position if len(position) <= 2 else "top center",
        marker=dict(
            size=node_size,
            line_width=2))

    node_adjacencies = []
    node_text = [str(n) for n in G.nodes]
    for idx in range(len(node_text)):
        if len(node_text[idx]) > 13:
            words = node_text[idx].split("_")
            if len(words) == 1:
                len_ = len(node_text[idx])
                node_text[idx] = node_text[idx][:len_//2] + "<br>" + node_text[idx][len_//2:]
            else:
                len_ = len(words)
                first_line = words[:len_//2]
                second_line = words[len_ // 2:]
                to_contatinate = list()
                for w in first_line:
                    to_contatinate.append(w)
                    to_contatinate.append("_")
                to_contatinate.append("<br>")
                for w in second_line:
                    to_contatinate.append(w)
                    to_contatinate.append("_")
                del to_contatinate[-1]
                node_text[idx] = "".join(to_contatinate)

    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))

    node_trace.marker.color = node_colors
    node_trace.text = node_text

    fig = go.Figure(data=[node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=0),
                        font=dict(size=16),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, automargin=True),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, automargin=True)
                    ))

    x_end = [x for idx, x in enumerate(edge_x) if idx % 2 == 1]
    y_end = [x for idx, x in enumerate(edge_y) if idx % 2 == 1]
    x_start = [x for idx, x in enumerate(edge_x) if idx % 2 == 0]
    y_start = [x for idx, x in enumerate(edge_y) if idx % 2 == 0]

    list_of_all_arrows = []

    for x0, y0, x1, y1, edge_color, edge_label in zip(x_end, y_end, x_start, y_start, edge_colors, edge_labels):
        arrow = go.layout.Annotation(dict(
            x=x0,
            y=y0,
            xref="x", yref="y",
            showarrow=True,
            axref="x", ayref='y',
            ax=x1,
            ay=y1,
            arrowhead=3,
            arrowwidth=2.5,
            arrowcolor=edge_color,
        )
        )
        list_of_all_arrows.append(arrow)
        angle = (math.atan((x1 - x0) / (y1 - y0)) / math.pi)*180 if y1 != y0 else 90
        if show_edge_labels:
            arrow_label = go.layout.Annotation(dict(
                x=(x0 + x1) / 2 if angle > 0 else (x0 + x1) / 2 * 1.02,
                y=(y0 + y1) / 2 if angle > 0 else (y0 + y1) / 2 * 1.05,
                xref="x", yref="y",
                text = edge_label,
                textangle = int(angle - 90) if angle > 0 else int(angle + 90),
                font = dict(color=edge_color)
                )
            )
            list_of_all_arrows.append(arrow_label)

    fig.update_layout(annotations=list_of_all_arrows)

    return fig


def copy_schedule(schedule, platform):
    # make a full copy of users_schedules
    new_schedule = {u: {k: v for k, v in d.items()} for u, d in schedule.items()}
    for user_id in new_schedule.keys():
        new_schedule[user_id][platform] = {k: v for k, v in new_schedule[user_id][platform].items()}
        new_schedule[user_id][platform]['script'] = [{k:v for k,v in a.items()} for a in new_schedule[user_id][platform]['script']]
        new_schedule[user_id][platform]["triggered_rules"] = {k:v for k,v in new_schedule[user_id][platform]["triggered_rules"].items()}
    return new_schedule


def get_triggered_rules_figure(data, username, fit_data_version):
    platform = data[username]["editor"].platform
    historical_schedule = copy_schedule(data[username]["historical_schedule"], platform)
    data[username]["editor"].pred(fit_data=(data[username]["thresholds"] | data["thresholds"])[fit_data_version],
                                  schedule=historical_schedule, start_time=0, curr_time=0)
    bans_count = list()
    hist_data_triggered_rules = defaultdict(lambda: 0)
    for u_id, u_data in historical_schedule.items():
        banned = False
        for r_triggered, val in u_data[platform]["triggered_rules"].items():
            hist_data_triggered_rules[r_triggered] += val if val > 0 else 0
            if val > 0:
                banned = True
        bans_count.append(1 if banned else 0)

    bans_count_gt = list()
    hist_data_triggered_rules_gt = defaultdict(lambda: 0)
    for u_id, u_data in data[username]["historical_schedule"].items():
        banned = False
        for r_triggered, val in u_data[platform]["triggered_rules"].items():
            hist_data_triggered_rules_gt[r_triggered] += val if val > 0 else 0
            if val > 0:
                banned = True
        bans_count_gt.append(1 if banned else 0)

    figure = {'data': [{'y': [val for r_name, val in hist_data_triggered_rules.items()],
                        'x': [r_name for r_name, val in hist_data_triggered_rules.items()],
                        'type': 'bar', 'name': f"Historical data: Expected bans"},
                       {'y': [val for r_name, val in hist_data_triggered_rules_gt.items()],
                        'x': [r_name for r_name, val in hist_data_triggered_rules_gt.items()],
                        'type': 'bar', 'name': f"Historical data: Actual bans"},
                      ],
        'layout': {
            'title': 'Number of times each rule was triggered '
        }
    }
    return figure, bans_count, bans_count_gt


def get_pred_triggered_rules_figure(data, username, fit_data_version):
    # fit_data = (data[username]["thresholds"] | data["thresholds"])[fit_data_version]
    # data[username]["editor"].pred(fit_data=fit_data,
    #                               schedule=data[username]["schedule"], start_time=0, curr_time=0)
    # rules = defaultdict(lambda: 0)
    # bans_count = list()
    # for u_id, u_data in data[username]["schedule"].items():
    #     banned = False
    #     for r_triggered, val in u_data[data[username]["editor"].platform]["triggered_rules"].items():
    #         rules[r_triggered] += val  if val > 0 else 0
    #         banned = True if val > 0 else False
    #     bans_count.append(1 if banned else 0)
    #
    # if "sleep_and_posts" in rules:
    #     clf = fit_data["sleep_and_posts"]['classifier']
    #     text_representation = tree.export_text(clf)
    #     print(text_representation)

    figure = {'data': [{'y': [0.862, 0.133],
                        'x': ["total_number_of_posts", "narrative_total_number_of_posts_baseball"],
                        'type': 'bar', 'name': f"Expected bans"}],
              'layout': {
                  'title': f'Feature importance (via Gini impurity)'}
              }
    return figure, []

