import argparse
import os
import sys
from pathlib import Path

import flask
from flask_login import LoginManager
import configparser
import dash_bootstrap_components as dbc
from dash_extensions.enrich import MultiplexerTransform, DashProxy
from utils import load_users, load_data
from main_view import main_div

DATA_PATH = Path("../data/initial_scenario.json")
USERS_PATH = Path("../data/users.json")

login_manager = LoginManager()
config = configparser.ConfigParser()


def parse_args(args):
    """
    Parameters of main() functions.
    :param args:
    :return:
    """
    parser = argparse.ArgumentParser(description="DASH demo")
    parser.add_argument("--data", "-d",
                        type=str,
                        default=str(DATA_PATH),
                        required=False,
                        help="Path to json file with default data.")

    return parser.parse_args(args)


def main(data_path=DATA_PATH, user_path=USERS_PATH):
    """
    Causal knowledge sharing demo.
    :param data_path: path to data file with DAGs for each site
    :param user_path: path to Json file with allowed usernames.
    :return: 
    """
    users = load_users(user_path)
    data = load_data(data_path)

    # star webserver
    server = flask.Flask(__name__)
    server.config.update(SECRET_KEY=os.urandom(15))
    app = DashProxy(
        transforms=[MultiplexerTransform()],
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        server=server
    )

    login_manager.init_app(app.server)
    app.layout = main_div(app, users, data, login_manager)

    app.run_server()


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    print(args)
    main(args.data)
