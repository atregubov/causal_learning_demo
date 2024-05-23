import json
from pathlib import Path
import pandas as pd
from model.rules import *


def load_users(users_path: Path):
    """
    Load users from json
    :param users_path: path to user data
    :return: dict

    """
    # Load users
    if users_path.exists():
        with open(users_path, 'r', encoding='utf-8') as ud_in:
            users = json.load(ud_in)
    else:
        users = {}

    return users


def load_data(data_path: Path):
    """
    Load data in DASH simulation format.
    :param data_path: path to data file
    :return: list of events

    """
    #data = load_json_into_df(data_path, True, False)
    data = {"s1": {"rules": [r for r in RULES.values()],
                   "local": [AggregateRule("S1: Local sleep hours and number of posts policy", "Trigger ban if constituent rules are triggered.",
                                           "ban", [SleepHoursRule(),
                                                   TotalNumberOfPostsRule(),
                                                   NarrativeRatioRule(),
                                                   TotalNumberOfPostsCauseSleepHoursRule(), SleepAndPostsRule()]),
                             AggregateRule("S1: Local sleep policy", "Trigger ban if constituent rule triggered",
                                    "ban", [SleepHoursRule()]),
                             ],
                   "editor": AggregateRule("Shared policy on number of posts", "Trigger ban if constituent rule triggered",
                                     "ban",
                                     [#SleepHoursRule(),
                                      #TotalNumberOfPostsRule(),
                                      NarrativeRatioRule(narrative="un"),
                                      SleepAndPostsRule(),
                                      TotalLinesOfPostsRule(),
                                      TotalNumberOfPostsCauseSleepHoursRule(),
                                      TotalNumberOfLinesCauseSleepHoursRule()]
                                     )
                   },
            "s2": {"rules": [r for r in RULES.values()],
                   "local": [AggregateRule("S2: Local narrative and number of posts policy", "Trigger ban if constituent rules are triggered.",
                                           "ban", [TotalNumberOfPostsRule(),
                                                   NarrativeRatioRule()]),
                             ],
                   "editor": AggregateRule("New policy", "Trigger ban if constituent rule triggered",
                                           "ban", [SleepHoursRule()])
                   },
            "s3": {"rules": [r for r in RULES.values()],
                   "local": [AggregateRule("S3: Local sleep hours, narrative and number of lines policy", "Trigger ban if constituent rules are triggered.",
                                           "ban", [SleepHoursRule(),
                                                   TotalLinesOfPostsRule(),
                                                   NarrativeRatioRule(),
                                                   TotalNumberOfLinesCauseSleepHoursRule()]),
                             ],
                   "editor": AggregateRule("New policy", "Trigger ban if constituent rule triggered",
                                           "ban", [SleepHoursRule()])
                   },
            "shared": [AggregateRule("Shared policy on number of posts", "Trigger ban if constituent rule triggered",
                                     "ban",
                                     [SleepHoursRule(),
                                      TotalNumberOfPostsRule(),
                                      NarrativeRatioRule(narrative="un"),
                                      TotalLinesOfPostsRule(),
                                      TotalNumberOfPostsCauseSleepHoursRule(),
                                      TotalNumberOfLinesCauseSleepHoursRule()],
                                     )],
            }
    return data


def load_json_into_df(filepath, ignore_first_line, verbose, aliases=None, narratives=None):
    """
    Loads a dataset from a json file.
    :param filepath: The filepath to the submission file.
    :param ignore_first_line: A True/False value. If True the first line is skipped.
    :param verbose: True/False value for debug output
    :param aliases: column names aliases
    :param narratives: narratives filter
    :return: The loaded dataframe (pandas dataframe) object.
    """

    dataset = []

    if verbose:
        print('Loading dataset at ' + filepath)

    with open(filepath, 'r') as file:
        for line_number, line in enumerate(file):

            if (line_number == 0 and ignore_first_line) or line == "" or line is None or line == "\n":
                continue

            if verbose:
                print(line_number)
                print('\r')

            line_data = json.loads(line)
            if line_data['nodeUserID'] is not None and \
                    ((narratives is not None and line_data['informationID'] in narratives) or (narratives is None)):
                line_data.pop('domain_linked', None)
                line_data.pop('has_URL', None)
                line_data.pop('links_to_external', None)
                line_data.pop('urls_linked', None)
                if aliases is not None:
                    line_data = {k: v for k, v in line_data.items() if k in aliases.keys()}
                    for k, alias in aliases.items():
                        k_val = line_data[k]
                        line_data.pop(k)
                        line_data[alias] = k_val
                dataset.append(line_data)

    if verbose:
        print(' ' * 100)
        print('\r')
        print(line_number)

    dataset = pd.DataFrame(dataset)

    if aliases is not None:
        inverse = {v:k for k, v in aliases.items()}
        new_columns = dataset.columns.values
        new_columns = [inverse[c] for c in new_columns]
        dataset.columns = new_columns

    return dataset
