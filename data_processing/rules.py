import pandas as pd
import networkx as nx
from abc import ABC, abstractmethod


class Feature(ABC):
    """
    Abstract Feature class defines how feature _name_ is computed. Feature class has a name and value() methods.
    """
    def __init__(self, name: str, platform: str = "reddit"):
        self.name = name
        self.platform = platform

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Feature: {self.name}"

    @abstractmethod
    def _value_from_actions(self, actions_script: list, start_time: int = 0, curr_time: int = None):
        return None

    def value(self, schedule: dict, start_time: int = 0, curr_time: int = None):
        """
        Compute feature value. Feature value is added to the schedule
        ( in schedule[user_id][self.platform]["features"][feature_name] ).
        The schedule must have the following structure:
        schedule[user_id][self.platform]
                                        [ "script"             | "ban"      | "triggered_rules" | "features"]
                                        [[list_of_actions]     | 1_0_-1_ban | rule_name         | feature_name ]

        """
        for user_id, u_schedule in schedule.items():
            if "features" not in u_schedule[self.platform]:
                u_schedule[self.platform]["features"] = {self.name: None}
            actions_script = u_schedule[self.platform]["script"]
            u_schedule[self.platform]["features"][self.name] = (
                self._value_from_actions(actions_script, start_time, curr_time))


class TotalNumberOfPostsFeature(Feature):
    """
    Feature total_number_of_posts, defined for each user.
    """
    def __init__(self, platform: str = "reddit"):
        super().__init__("total_number_of_posts", platform)

    def _value_from_actions(self, actions_script: list, start_time: int = 0, curr_time: int = None):
        return len(actions_script)


class TotalLinesFeature(Feature):
    """
    Feature total_number_of_posts, defined for each user.
    """
    def __init__(self, platform: str = "reddit"):
        super().__init__("total_lines_of_posts", platform)

    def _value_from_actions(self, actions_script: list, start_time: int = 0, curr_time: int = None):
        total_lines = sum([v['n_lines'] for v in actions_script])
        return total_lines


class SleepHoursFeature(Feature):
    """
    Feature sleep_hours is a list of hours when user sleeps,  defined for each user.
    """
    def __init__(self, platform: str ="reddit"):
        super().__init__("sleep_hours", platform)

    def _value_from_actions(self, actions_script: list, start_time: int = 0, curr_time: int = None):
        hours_of_sleep_dist = [0 for _ in range(24)]
        for action in actions_script:
            hours_of_sleep_dist[int(((action['nodeTime'] - start_time) % (24 * 3600)) / 3600)] += 1
        return hours_of_sleep_dist


class NarrativeRatioFeature(Feature):
    """
    Feature narrative_ratio, defined for each user.
    """
    def __init__(self, platform: str = "reddit"):
        super().__init__("narrative_ratio", platform)

    def _value_from_actions(self, actions_script: list, start_time: int = 0, curr_time: int = None):
        script_df = pd.DataFrame(actions_script)
        narrative_counts = script_df.groupby(["informationID"]).size().to_dict()
        narrative_ratios = {k: v/len(actions_script) for k, v in narrative_counts.items()}
        return narrative_ratios


class Rule(ABC):
    """
    Rule class defines causal relationship between ban node (outcome) causing factors node (features).
    The schedule must have the following structure:
    schedule[user_id][self.platform]
                                    [ "script"             | "ban"      | "triggered_rules" | "features"]
                                        [[list_of_actions] | 1_0_-1_ban | rule_name         | feature_name ]
    """

    def __init__(self, name: str, rule_str: str, outcome_name: str, rules: list, depends_on: list,
                 features: list, platform: str ="reddit"):
        self.name = name
        self.platform = platform
        self.rule_str = rule_str
        self.fit_data = None
        self.local = True

        self.outcome_name = outcome_name
        self.rules = rules
        self.depends_on = depends_on
        self.features = features

    def rule_str(self) -> str:
        return self.rule_str

    def __str__(self):
        return self.rule_str

    def __repr__(self):
        return f"Rule: {self.rule_str}"

    @abstractmethod
    def get_DAG(self) -> nx.MultiDiGraph:
        """
        Creates an NX graph.
        """
        return None

    @abstractmethod
    def pred(self, schedule: dict, start_time: int = 0, curr_time: int = None):
        """
        Predict outcome value (e.g. ban), predicted outcome is added to the schedule
        ( in schedule[user_id][self.platform]["triggered_rules"][self.rule_name] ).
        """
        return None

    @abstractmethod
    def fit(self, schedule: dict, start_time: int = 0, curr_time: int = None):
        """
        Updates self.fit_data (e.g. thresholds, weights, and other rule parameters) to fit observations.
        The schedule must have ban labels in schedule[user_id][self.platform]["ban"]
        """
        return None


class AggregateRule(Rule):
    """
    Rule class defines causal relationship between ban node (outcome) causing factors node (features).
    The schedule must have the following structure:
    schedule[user_id][self.platform]
                                    [ "script"             | "ban"      | "triggered_rules" | "features"]
                                        [[list_of_actions] | 1_0_-1_ban | rule_name         | feature_name ]
    """

    def __init__(self, name: str, rule_str: str, outcome_name: str, rules: list, platform="reddit"):
        super().__init__(name, rule_str, outcome_name, rules, [], [], platform)
        self.fit_data = None

    def get_DAG(self) -> nx.MultiDiGraph:
        """
        Creates an NX graph with features nodes pointing at the ban node.
        """
        G = nx.MultiDiGraph()
        for rule in self.rules:
            rule_G = rule.get_DAG()
            G.add_nodes_from(list(rule_G.nodes))
            G.add_edges_from(list(rule_G.edges))

        return G

    def pred(self, schedule: dict, start_time: int = 0, curr_time: int = None):
        """
        Predict outcome value (e.g. ban), predicted outcome is added to the schedule
        ( in schedule[user_id][self.platform]["triggered_rules"][self.rule_name] ).
        """
        pred_vals = list()
        for rule in self.rules:
            pred_vals.append(rule.pred(schedule, start_time, curr_time))
        return max(pred_vals)

    def fit(self, schedule: dict, start_time: int = 0, curr_time: int = None):
        """
        Updates self.fit_data (e.g. thresholds, weights, and other rule parameters) to fit observations.
        The schedule must have ban labels in schedule[user_id][self.platform]["ban"]
        """
        for rule in self.rules:
            rule.fit(schedule, start_time, curr_time)


class BasicRule(Rule):
    """
    Rule class defines causal relationship between ban node (outcome) causing factors node (features).
    The schedule must have the following structure:
    schedule[user_id][self.platform]
                                    [ "script"             | "ban"      | "triggered_rules" | "features"]
                                        [[list_of_actions] | 1_0_-1_ban | rule_name         | feature_name ]
    """

    def __init__(self, name: str, rule_str: str, features: list, outcome_name: str = "ban", platform="reddit"):
        super().__init__(name, rule_str, outcome_name, [], [], features, platform)
        self.fit_data = None

    def get_DAG(self) -> nx.MultiDiGraph:
        """
        Creates an NX graph with features nodes pointing at the ban node.
        """
        G = nx.MultiDiGraph()
        feature_names = [f.name for f in self.features]
        G.add_nodes_from(feature_names)
        G.add_nodes_from([self.outcome_name])
        G.add_edges_from([(f, self.outcome_name) for f in feature_names])

        return G

    @abstractmethod
    def pred(self, schedule: dict, start_time: int = 0, curr_time: int = None):
        """
        Predict outcome value (e.g. ban), predicted outcome is added to the schedule
        ( in schedule[user_id][self.platform]["triggered_rules"][self.rule_name] ).
        """
        return None

    def feature_values(self, schedule: dict, start_time: int = 0, curr_time: int = None):
        """
        Computes features needed for the rule, feature values are added in the schedule
        ( in schedule[user_id][self.platform]["feaures"][feature_name] ).
        """
        for feature in self.features:
            feature.value(schedule, start_time, curr_time)

    @abstractmethod
    def fit(self, schedule: dict, start_time: int = 0, curr_time: int = None):
        """
        Updates self.fit_data (e.g. thresholds, weights, and other rule parameters) to fit observations.
        The schedule must have ban labels in schedule[user_id][self.platform]["ban"]
        """
        return None


class AbstractOneFeatureThresholdRule(BasicRule):
    def pred(self, schedule: dict, start_time: int = 0, curr_time: int = None):
        if self.fit_data[self.name] is None:
            raise AttributeError("fit_data is None. Run fit() first or set fit_data attribute.")

        # update feature values
        self.feature_values(schedule, start_time, curr_time)


        # predict/evaluate ban
        for user, u_schedule in schedule.items():
            if "triggered_rules" not in u_schedule[self.platform]:
                u_schedule[self.platform]["triggered_rules"] = dict()
            u_schedule[self.platform]["triggered_rules"][self.name] = 0
            feature_value = u_schedule[self.platform]["features"][self.features[0].name]

            if isinstance(self.fit_data[self.name], dict):
                banned = False
                not_banned = False
                ban_unknown = False

                if (feature_value > self.fit_data[self.name]['banned']['min'] and
                        feature_value < self.fit_data[self.name]['banned']['max']):
                    banned = True
                elif (feature_value > self.fit_data[self.name]['notbanned']['min'] and
                      feature_value < self.fit_data[self.name]['notbanned']['max']):
                    not_banned = True
                else:
                    ban_unknown = True

                u_schedule[self.platform]["triggered_rules"][self.name] = 1 if banned else 0 if not_banned else -1

            else:
                u_schedule[self.platform]["triggered_rules"][self.name] = 1 \
                    if feature_value > self.fit_data[self.features[0].name] else 0

            if u_schedule[self.platform]["triggered_rules"][self.name] == 1:
                u_schedule[self.platform]["ban"] = 1
            else:
                u_schedule[self.platform]["ban"] = 0

    def fit(self, schedule: dict, start_time: int = 0, curr_time: int = None):
        data = [{"ban": u_data[self.platform]["ban"],
                 self.features[0].name: u_data[self.platform]["features"][self.features[0].name]
                 } for u_id, u_data in schedule.items()]
        self.fit_data[self.name] = {"banned": {"min": None, "max": None}, "notbanned": {"min": None, "max": None}}

        f_data = self.fit_data[self.name]
        bans_df = pd.DataFrame.from_records(data)
        f_data['banned']['min'] = bans_df[bans_df['ban'] == 1][self.features[0].name].min()
        f_data['banned']['max'] = bans_df[bans_df['ban'] == 1][self.features[0].name].max()
        f_data['notbanned']['min'] = bans_df[bans_df['ban'] == 0][self.features[0].name].min()
        f_data['notbanned']['max'] = bans_df[bans_df['ban'] == 0][self.features[0].name].max()


class TotalNumberOfPostsRule(AbstractOneFeatureThresholdRule):
    def __init__(self):
        super().__init__("total_number_of_posts",
                         "If the total number of all user posts \nis in the \"to_ban\" range -> ban.\n"
                         "Also If the total number of all user posts \nis in the \"no_to_ban\" range -> no ban.\n"
                         "Otherwise no ban.",
                         [TotalNumberOfPostsFeature()], "ban")


class TotalLinesOfPostsRule(AbstractOneFeatureThresholdRule):
    def __init__(self):
        super().__init__("total_lines_of_posts",
                         "If the total number of lines \nin all user posts is in the \"to_ban\" range -> ban.\n"
                         "Also If the total number of lines \nin all user posts is in the \"no_ban\" range -> no ban.\n"
                         "Otherwise no ban.",
                         [TotalLinesFeature()], "ban")


class SleepHoursRule(BasicRule):
    def __init__(self):
        super().__init__("sleep_hours",
                         "If hours when user is inactive (sleep_hours)\n are in the ban range -> ban.\n"
                         "If sleep_hours are in the \"no_ban_range\" -> no ban",
                         [SleepHoursFeature()], "ban")

    def pred(self, schedule: dict, start_time: int = 0, curr_time: int = None):
        if self.fit_data[self.name] is None:
            raise AttributeError("fit_data is None. Run fit() first or set fit_data attribute.")

        # update feature values
        self.feature_values(schedule, start_time, curr_time)

        # predict/evaluate ban
        for user, u_schedule in schedule.items():
            if "triggered_rules" not in u_schedule[self.platform]:
                u_schedule[self.platform]["triggered_rules"] = dict()
            u_schedule[self.platform]["triggered_rules"][self.name] = 0
            hours_of_sleep_dist = u_schedule[self.platform]["features"][self.features[0].name]
            banned_by_min_sleep_hours = False
            ban_status_unknown = False
            for index, h in enumerate(hours_of_sleep_dist):
                if h != 0:
                    if isinstance(self.fit_data[self.name], dict):
                        # this assumes that first k-elements in hours_of_sleep_dist are sleep hours:
                        if (self.fit_data[self.name]['banned']['min'] <= index <=
                                self.fit_data[self.name]['banned']['max']):
                            banned_by_min_sleep_hours = True
                        elif (self.fit_data[self.name]['notbanned']['min'] <= index <=
                                self.fit_data[self.name]['notbanned']['max']):
                            ban_status_unknown = True
                        break
                    else:
                        # this assumes that first k-elements in hours_of_sleep_dist are sleep hours:
                        if index < self.fit_data[self.name]:
                            banned_by_min_sleep_hours = True
                        break

            if isinstance(self.fit_data[self.name], dict):
                u_schedule[self.platform]["triggered_rules"][self.name] = 1 \
                    if banned_by_min_sleep_hours else -1 if ban_status_unknown else 0
            else:
                u_schedule[self.platform]["triggered_rules"][self.name] = 1 \
                    if banned_by_min_sleep_hours else 0

            if u_schedule[self.platform]["triggered_rules"][self.name] == 1:
                u_schedule[self.platform]["ban"] = 1
            else:
                u_schedule[self.platform]["ban"] = 0

    def fit(self, schedule: dict, start_time: int = 0, curr_time: int = None):
        data = [{"ban": u_data[self.platform]["ban"],
                 "sleep_bgn": min([hour_idx for hour_idx, val
                                   in enumerate(u_data[self.platform]["features"][self.features[0].name]) if val > 0]),
                 "sleep_end": max([hour_idx for hour_idx, val
                                   in enumerate(u_data[self.platform]["features"][self.features[0].name]) if val > 0])
                 }
                for u_id, u_data in schedule.items()]
        self.fit_data[self.name] = {"banned": {"min": None, "max": None}, "notbanned": {"min": None, "max": None}}

        bans_df = pd.DataFrame.from_records(data)
        f_data = self.fit_data[self.name]
        f_data['banned']['min'] = bans_df[bans_df['ban'] == 1]["sleep_bgn"].min()
        f_data['banned']['max'] = bans_df[bans_df['ban'] == 1]["sleep_end"].max()
        f_data['notbanned']['min'] = bans_df[bans_df['ban'] == 0]["sleep_bgn"].min()
        f_data['notbanned']['max'] = bans_df[bans_df['ban'] == 0]["sleep_end"].max()


class NarrativeRatioRule(BasicRule):
    def __init__(self, narrative: str = None):
        super().__init__("narrative_ratio",
                         "If narrative ratio is in the ban range -> ban.\n "
                         "If narrative ratio is in no ban range -> no ban.",
                         [NarrativeRatioFeature()], "ban")
        self.narrative = narrative

    def pred(self, schedule: dict, start_time: int = 0, curr_time: int = None):
        if self.fit_data[self.name] is None:
            raise AttributeError("fit_data is None. Run fit() first or set fit_data attribute.")

        # update feature values
        self.feature_values(schedule, start_time, curr_time)

        # predict/evaluate ban
        all_narratives = set()
        for user, u_schedule in schedule.items():
            all_narratives.update(pd.DataFrame(u_schedule[self.platform]["script"])["informationID"].unique())
        narratives = [self.narrative] if self.narrative is not None else list(all_narratives)

        for user, u_schedule in schedule.items():
            if "triggered_rules" not in u_schedule[self.platform]:
                u_schedule[self.platform]["triggered_rules"] = dict()
            u_schedule[self.platform]["triggered_rules"][self.name] = 0

            for narrative in narratives:
                if narrative in u_schedule[self.platform]["features"][self.features[0].name]:
                    feature_value = u_schedule[self.platform]["features"][self.features[0].name][narrative]
                    if isinstance(self.fit_data[self.name], dict):
                        banned = False
                        not_banned = False
                        ban_unknown = False

                        if (feature_value > self.fit_data[self.name]['banned']['min'] and
                                feature_value < self.fit_data[self.name]['banned']['max']):
                            banned = True
                        elif (feature_value > self.fit_data[self.name]['notbanned']['min'] and
                              feature_value < self.fit_data[self.name]['notbanned']['max']):
                            not_banned = True
                        else:
                            ban_unknown = True

                        u_schedule[self.platform]["triggered_rules"][self.name] = max(1 if banned else 0
                        if not_banned else -1, u_schedule[self.platform]["triggered_rules"][self.name])

                    else:
                        u_schedule[self.platform]["triggered_rules"][self.name] = \
                            max(1 if feature_value > self.fit_data[self.features[0].name] else 0,
                                u_schedule[self.platform]["triggered_rules"][self.name])

            if u_schedule[self.platform]["triggered_rules"][self.name] == 1:
                u_schedule[self.platform]["ban"] = 1
            else:
                u_schedule[self.platform]["ban"] = 0

    def fit(self, schedule: dict, start_time: int = 0, curr_time: int = None):
        data = []
        for u_id, u_data in schedule.items():
            data_itm = {"ban": u_data[self.platform]["ban"],
                        self.features[0].name: 0}
            u_narrative_ratios = u_data[self.platform]["features"][self.features[0].name]
            if self.narrative is not None:
                if self.narrative in u_narrative_ratios:
                    data_itm[self.features[0].name] = u_narrative_ratios[self.narrative]
                else:
                    data_itm[self.features[0].name] = 0
            else:
                data_itm[self.features[0].name] = max(list(u_narrative_ratios.values()))
            data.append(data_itm)
        self.fit_data[self.name] = {"banned": {"min": None, "max": None}, "notbanned": {"min": None, "max": None}}

        bans_df = pd.DataFrame.from_records(data)
        f_data = self.fit_data[self.name]
        f_data['banned']['min'] = bans_df[bans_df['ban'] == 1][self.features[0].name].min()
        f_data['banned']['max'] = bans_df[bans_df['ban'] == 1][self.features[0].name].max()
        f_data['notbanned']['min'] = bans_df[bans_df['ban'] == 0][self.features[0].name].min()
        f_data['notbanned']['max'] = bans_df[bans_df['ban'] == 0][self.features[0].name].max()


class FeaturesRelationshipRule(Rule):
    """
    This class describes relationships between Features. Features can be computed directly from schedules, but still
    can have causal relationships among themselves. In such cases to add this relationship to DAG use this class
    FeaturesRelationshipRule.
    """
    def __init__(self, name: str, rule_str: str, features: list, outcome_feature: Feature):
        super().__init__(name, rule_str, outcome_feature.name, [],[], features)
        self.outcome_feature = outcome_feature

    def pred(self, schedule: dict, start_time: int = 0, curr_time: int = None):
        for feature in [self.features[0], self.outcome_feature]:
            feature.value(schedule, start_time, curr_time)

    def fit(self, schedule: dict, start_time: int = 0, curr_time: int = None):
        return None

    def get_DAG(self) -> nx.MultiDiGraph:
        """
        Creates an NX graph.
        """
        G = nx.MultiDiGraph()
        feature_names = [f.name for f in self.features]
        G.add_nodes_from(feature_names)
        G.add_nodes_from([self.outcome_name])
        G.add_edges_from([(f, self.outcome_name) for f in feature_names])

        return G


class TotalNumberOfPostsCauseSleepHoursRule(FeaturesRelationshipRule):
    def __init__(self):
        super().__init__("total_number_of_posts_cause_sleep_hours",
                         "The total number of posts affects sleep hours of a user.",
                         [TotalNumberOfPostsFeature()], SleepHoursFeature())


class TotalNumberOfLinesCauseSleepHoursRule(FeaturesRelationshipRule):
    def __init__(self):
        super().__init__("total_number_of_lines_cause_sleep_hours",
                         "The total number of lines in posts affects sleep hours of a user.",
                         [TotalLinesFeature()], SleepHoursFeature())


RULES = {r.name: r for r in [SleepHoursRule(),
                             TotalNumberOfPostsRule(),
                             NarrativeRatioRule(narrative="un"),
                             TotalLinesOfPostsRule(),
                             TotalNumberOfPostsCauseSleepHoursRule(),
                             TotalNumberOfLinesCauseSleepHoursRule()]}

FEATURES = {f.name: f for f in [SleepHoursFeature(),
                                TotalNumberOfPostsRule(),
                                TotalLinesFeature(),
                                NarrativeRatioFeature()]}

