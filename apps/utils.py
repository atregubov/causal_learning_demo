import json
from pathlib import Path
import pandas as pd
from model.rules import *
from model.schedules import *


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

    gt_historical_thresholds = get_random_thresholds(narrative="baseball")
    data = {"s1": {"rules": [r for r in RULES.values()],
                   "local": [
                             # AggregateRule("S1: Local sleep hours and number of posts policy", "Trigger ban if constituent rules are triggered.",
                             #               "ban", [
                             #                       # SleepHoursRule(),
                             #                       # TotalNumberOfPostsRule(),
                             #                       SleepAndPostsRule(),
                             #                       NarrativeRatioRule()]),
                             AggregateRule("S1: number of posts policy", "number_of_posts > 100 -> ban",
                                     "ban", [TotalNumberOfPostsRule()]),
                             AggregateRule("S1: Local sleep policy", "Trigger ban if constituent rule triggered",
                                    "ban", [SleepHoursRule()]),
                             ],
                   "schedule": generate_site_schedule(site_idx=0),
                   "historical_schedule": generate_historical_data(schedule=generate_site_schedule(site_idx=0),
                                                                   thresholds=gt_historical_thresholds,
                                                                   policy=AggregateRule("S1: number of posts policy",
                                                                                        "number_of_posts > 100 -> ban",
                                                                                        "ban",
                                                                                        [TotalNumberOfPostsRule(),
                                                                                         SleepHoursRule()])),
                   "thresholds": {"local:S1 initial": get_random_thresholds("baseball"),
                                  "local:S1 historical data fit": get_random_thresholds("baseball") },
                   "editor": AggregateRule("S1+S1: number of posts and sleep hours policy", "number_of_posts > <threshold> -> ban "
                                                                         "\nor number_of_sleep_hours < <threshold> -> ban",
                                                              "ban", [TotalLinesOfPostsRule(),
                                                                                         SleepHoursRule()]),
                   # "editor": AggregateRule("Shared policy on number of posts", "Trigger ban if constituent rule triggered",
                   #                   "ban",
                   #                   [#SleepHoursRule(),
                   #                    #TotalNumberOfPostsRule(),
                   #                    NarrativeRatioRule(narrative="baseball"),
                   #                    SleepAndPostsRule(),
                   #                    TotalLinesOfPostsRule(),
                   #                    TotalNumberOfPostsCauseSleepHoursRule(),
                   #                    TotalNumberOfLinesCauseSleepHoursRule()]
                   #                   )
                   },
            "s2": {"rules": [r for r in RULES.values()],
                   "local": [AggregateRule("S2: sleep hours rule", "If number of sleep hours < 9 -> ban",
                                           "ban", [SleepHoursRule()]),
                             AggregateRule("S2: number of posts policy on baseball", "number of posts on baseball > 50 -> ban",
                                           "ban", [NarrativeNumberOfPostsRule(narrative="baseball")]),
                             AggregateRule("S2: Local narrative and number of posts policy", "Trigger ban if constituent rules are triggered.",
                                           "ban", [TotalNumberOfPostsRule(),
                                                   NarrativeRatioRule()]),
                             ],
                   "schedule": generate_site_schedule(site_idx=1),
                   "historical_schedule": generate_historical_data(schedule=generate_site_schedule(site_idx=1),
                                                                   thresholds=gt_historical_thresholds,
                                                                   policy=AggregateRule("S1: number of posts policy",
                                                                                        "number_of_posts > 100 -> ban",
                                                                                        "ban",
                                                                                        [TotalNumberOfPostsRule(),
                                                                                         SleepHoursRule()])),
                   "thresholds": {"local:S2 initial": get_random_thresholds("baseball"),
                                  "local:S2 historical data fit": get_random_thresholds("baseball")},
                   # "editor": AggregateRule("S2: number of posts policy", "number_of_posts > 100 -> ban",
                   #                         "ban", [TotalNumberOfPostsRule()]),
                   "editor": AggregateRule("S2: number of posts policy  on baseball", "number of posts on baseball > 50 -> ban",
                                                              "ban", [NarrativeNumberOfPostsRule(narrative="baseball")])
                   },
            "s3": {"rules": [r for r in RULES.values()],
                   "local": [AggregateRule("S3: Local sleep hours, narrative and number of lines policy", "Trigger ban if constituent rules are triggered.",
                                           "ban", [SleepHoursRule(),
                                                   TotalLinesOfPostsRule(),
                                                   NarrativeRatioRule(),
                                                   TotalNumberOfLinesCauseSleepHoursRule()]),
                             ],
                   "schedule": generate_site_schedule(site_idx=2),
                   "historical_schedule": generate_historical_data(schedule=generate_site_schedule(site_idx=2),
                                                                   thresholds=gt_historical_thresholds,
                                                                   policy=AggregateRule("S1: number of posts policy",
                                                                                        "number_of_posts > 100 -> ban",
                                                                                        "ban",
                                                                                        [TotalNumberOfPostsRule(),
                                                                                         SleepHoursRule()])),
                   "thresholds": {"local:S3 initial": get_random_thresholds("baseball"),
                                  "local:S3 historical data fit": get_random_thresholds("baseball")},
                   "editor": AggregateRule("New policy", "Trigger ban if constituent rule triggered",
                                           "ban", [SleepHoursRule()])
                   },
            "shared": [
                        AggregateRule("Shared (from site 2): sleep hours bab policy", "If number of sleep hours < 9 -> ban",
                              "ban", [SleepHoursRule()]),
                        AggregateRule("Shared (from site 2): Policy on number of posts", "Trigger ban if constituent rule triggered",
                                     "ban",
                                     [SleepHoursRule(),
                                      TotalNumberOfPostsRule(),
                                      NarrativeRatioRule(narrative="baseball"),
                                      TotalLinesOfPostsRule(),
                                      TotalNumberOfPostsCauseSleepHoursRule(),
                                      TotalNumberOfLinesCauseSleepHoursRule()],
                                     ),
                AggregateRule("Shared (from site 1): number of posts policy", "number_of_posts > 100 -> ban",
                              "ban", [TotalNumberOfPostsRule()]),
            ],
            "thresholds": {"shared:S1": get_random_thresholds("baseball"),
                           "shared:S2": get_random_thresholds("baseball"),
                           "shared:S3": get_random_thresholds("baseball")},
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
