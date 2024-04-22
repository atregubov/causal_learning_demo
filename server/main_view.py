from dash import html
from dash.dependencies import Input, Output, State
from dash import dcc
from flask_login import UserMixin, login_user, current_user
from model_view import model_div


class User(UserMixin):
    """User class with default attributes."""
    def __init__(self, username):
        self.id = username


def main_div(app, users, data, login_manager):
    """
    Main page div block layout.
    :param app: DASH/Flask webapp object
    :param users: allowed users (for user login)
    :param data: dataframe with simulation outputs/inputs
    :param login_manager:
    :return:
    """
    m_div = html.Div(
        id="main_div",
        style={'width': '100%', 'display': 'block', 'padding': '50px 10px 10px 10px',
               'align-items': 'center', 'justify-content': 'center',
               },
        children=[
            html.Div(children=[html.H1("Causal paper demo")],
                     id="header_div",
                     className='12md',
                     style={'width': '100%', 'display': 'flex', 'padding': '10px 10px 20px 20px',
                            'align-items': 'left', 'justify-content': 'left',
                            },
                     ),
            html.Table(
                id="login_table",
                style={'width': '100%', 'padding': '10px 10px 20px 20px'},
                children=[
                    html.Tbody(
                        children=[
                            html.Tr(
                                children=[
                                    html.Td(
                                        id="login_div",
                                        className='12md',
                                        style={'text-align': 'right'},
                                        children=[
                                            dcc.Input(
                                                id="username",
                                                type="text",
                                                placeholder="Username",
                                                maxLength=20,
                                                style={'margin-right': '5px'}
                                            ),
                                            dcc.Input(
                                                id="password",
                                                type="password",
                                                placeholder="Password",
                                                style={'margin-right': '5px'}
                                            ),
                                            html.Button(
                                                'Login', id="login", n_clicks=0, style={'margin-right': '10px'}
                                            )
                                        ]
                                    )
                                ]
                            )
                        ]
                    )
                ]
            ),
            html.Table(
                id="welcome_table",
                style={'width': '100%', 'padding': '10px 10px 20px 20px'},
                children=[
                    html.Tbody(
                        children=[
                            html.Tr(
                                children=[
                                    html.Td(
                                        id="welcome-message",
                                        children=[html.H4("Welcome, guest!")],
                                        style={"text-align": "right", "padding": "0px 10px"}
                                    ),
                                ]
                            )
                        ]
                    ),
                    html.Div(id="results_display",
                             children=[model_div(app, data, hidden=True)]
                             ),
                ]
            ),

        ]
    )

    # The following callbacks deal with user auth
    @login_manager.user_loader
    def load_user(user_id):
        return User(user_id)

    @app.callback(
        Output('welcome-message', 'children'),
        Output('username', 'value'),
        Output('password', 'value'),
        Output('results_display', 'children'),
        Input('login', 'n_clicks'),
        State('username', 'value'),
        State('password', 'value'),
        State('results_display', 'children')
    )
    def login_button_click(n_clicks, username, password, results_display):
        active_user = current_user.get_id()
        print(active_user)
        if username in users and users[username] == password:
            user = User(username)
            login_user(user)
            return ([html.H4(f"Welcome, {username}!")], "", "", [model_div(app, data, hidden=False)])
        elif n_clicks > 0:
            return ([html.H4("Login failed")], "", "", [model_div(app, data, hidden=True)])
        elif active_user:
            return ([html.H4(f"Welcome, {active_user}!")], "", "", [model_div(app, data, hidden=True)])
        else:
            return ([html.H4("Welcome, guest! Log in to edit.")], "", "", [model_div(app, data, hidden=True)])

    return m_div
