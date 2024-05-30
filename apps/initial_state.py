import json
from pathlib import Path
import pandas as pd
from model.rules import *
from model.schedules import *


def create_initial_state_data() -> dict:
    """
    Creates initial state of the models/policies, thresholds for 3 sites in the CCD Demo
    :return: dictionary object with site and shared data.

    """
    gt_historical_thresholds = get_random_thresholds(narrative="baseball")
    s1_historical_schedule = generate_historical_data(schedule=generate_site_schedule(site_idx=0),
                                                      thresholds=gt_historical_thresholds,
                                                      policy=AggregateRule("S1+S2: number of posts and sleep hours policy",
                                                                           "number_of_posts > <threshold> -> ban "
                                                                           "\nor number_of_sleep_hours < <threshold> -> ban",
                                                              "ban",
                                                                           [TotalNumberOfPostsRule(),
                                                                            TotalLinesOfPostsRule(),
                                                                            SleepHoursRule()]))
    s1_historical_data_fit = get_local_data_fit_for_all_rules(s1_historical_schedule)

    data = {"s1": {"rules": [r for r in RULES.values()],
                   "local": [AggregateRule("S1: number of posts and sleep hours rule",
                                           "number_of_posts > 100 -> ban",
                                           "ban",
                                           shared_by="Me",
                                           rules=[TotalLinesOfPostsRule(),
                                                  SleepHoursRule()]),
                             AggregateRule("S1: Local sleep policy",
                                           "Trigger ban if constituent rule triggered",
                                           "ban",
                                           shared_by="Me",
                                           rules=[SleepHoursRule()]),
                             ],
                   "schedule": generate_site_schedule(site_idx=0),
                   "historical_schedule": s1_historical_schedule,
                   "thresholds": {"local (S1): thresholds fit to historical data": s1_historical_data_fit
                                  },
                   "editor": AggregateRule("S1+S2 combined policy: number of posts and sleep hours policy",
                                           "number_of_posts > <threshold> -> ban "
                                           "\nor number_of_sleep_hours < <threshold> -> ban",
                                           "ban",
                                           [
                                            # TotalLinesOfPostsRule(),
                                            # SleepHoursRule().
                                            SleepAndPostsRule(),
                                            NarrativeRatioRule(narrative="baseball"),
                                            TotalLinesOfPostsRule(),
                                            TotalNumberOfPostsCauseSleepHoursRule(),
                                            TotalNumberOfLinesCauseSleepHoursRule()
                                            ]),
                   },
            "s2": {"rules": [r for r in RULES.values()],
                   "local": [AggregateRule("S2: sleep hours rule",
                                           "If number of sleep hours < 9 -> ban",
                                           "ban",
                                           [SleepHoursRule()]),
                             AggregateRule("S2: number of posts policy on baseball",
                                           "number of posts on baseball > 50 -> ban",
                                           "ban",
                                           [NarrativeNumberOfPostsRule(narrative="baseball")]),
                             AggregateRule("S2: Local narrative and number of posts policy",
                                           "Trigger ban if constituent rules are triggered.",
                                           "ban",
                                           [TotalNumberOfPostsRule(),
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
                   "thresholds": {"local:S2 historical data fit": get_random_thresholds("baseball")},
                   "editor": AggregateRule("S2: number of posts policy  on baseball",
                                           "number of posts on baseball > 50 -> ban",
                                           "ban",
                                           [NarrativeNumberOfPostsRule(narrative="baseball")])
                   },
            "s3": {"rules": [r for r in RULES.values()],
                   "local": [AggregateRule("S3: Local sleep hours, narrative and number of lines policy",
                                           "Trigger ban if constituent rules are triggered.",
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
                   "thresholds": {"local:S3 historical data fit": get_random_thresholds("baseball"),
                                  },
                   "editor": AggregateRule("New policy",
                                           "Trigger ban if constituent rule triggered",
                                           "ban",
                                           [SleepHoursRule()])
                   },
            "shared": [AggregateRule("Shared (from site 2): Policy on number of posts",
                                     "Trigger ban if constituent rule triggered",
                                     "ban",
                                     shared_by="S2",
                                     rules=[SleepAndPostsRule(),
                                            NarrativeRatioRule(narrative="baseball"),
                                            TotalLinesOfPostsRule(),
                                            TotalNumberOfPostsCauseSleepHoursRule(),
                                            TotalNumberOfLinesCauseSleepHoursRule()],
                                     ),
                       AggregateRule("Shared (from site 2): sleep hours ban policy",
                                     "If number of sleep hours < 9 -> ban",
                                     "ban",
                                     shared_by="S2",
                                     rules=[SleepHoursRule()]),

                       AggregateRule("Shared (from site 1): number of posts policy",
                                     "number_of_posts > 100 -> ban",
                                     "ban",
                                     shared_by="S1",
                                     rules=[TotalNumberOfPostsRule()]),
                       ],
            "thresholds": {"shared:S1": get_random_thresholds("baseball"),
                           "shared:S2": get_random_thresholds("baseball"),
                           "shared:S3": get_random_thresholds("baseball")
                           },
            }

    return data

