import pandas as pd
import networkx as nx
from abc import ABC, abstractmethod


class Feature(ABC):
    """
    Feature class defines how feature _name_ is computed. Feature class has a name and value() methods.
    """
    def __init__(self, name: str, platform: str = "reddit"):
        self.name = name
        self.platform = platform

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Feature: {self.name}"

    @abstractmethod
    def value(self, schedule: dict, start_time: int = 0, curr_time: int = None):
        """
        Compute feature value.
        """
        return None


class SleepHoursFeature(Feature):
    """
    Feature class defines how feature _name_ is computed. Feature class has a name and value() methods.
    """
    def __init__(self, platform: str ="reddit"):
        super().__init__("sleep_hours", platform)

    def _get_sleep_hours(self, actions_script: list, start_time: int = 0):
        hours_of_sleep_dist = [0 for _ in range(24)]
        for action in actions_script:
            hours_of_sleep_dist[int(((action['nodeTime'] - start_time) % (24 * 3600)) / 3600)] += 1

        return hours_of_sleep_dist

    def value(self, schedule: dict, start_time: int = 0, curr_time: int = None):
        """
        Compute feature value. Feature value is added to the schedule
        ( in schedule[user_id][self.platform]["features"][feature_name] ).
        """
        for user_id, u_schedule in schedule.items():
            if "features" not in u_schedule[self.platform]:
                u_schedule[self.platform]["features"] = {self.name: None}
            actions_script = u_schedule[self.platform]["script"]
            u_schedule[self.platform]["features"][self.name] = self._get_sleep_hours(actions_script, start_time)


class TotalNumberOfPostsFeature(Feature):
    """
    Feature class defines how feature _name_ is computed. Feature class has a name and value() methods.
    """

    def __init__(self, platform: str = "reddit"):
        super().__init__("total_number_of_posts", platform)

    def _get_total_number_of_posts(self, actions_script: list):
        return len(actions_script)

    def value(self, schedule: dict, start_time: int = 0, curr_time: int = None):
        """
        Compute feature value. Feature value is added to the schedule
        ( in schedule[user_id][self.platform]["features"][feature_name] ).
        """
        for user_id, u_schedule in schedule.items():
            if "features" not in u_schedule[self.platform]:
                u_schedule[self.platform]["features"] = {self.name: None}
            actions_script = u_schedule[self.platform]["script"]
            u_schedule[self.platform]["features"][self.name] = self._get_total_number_of_posts(actions_script)


class Rule(ABC):
    """
    Rule class defines causal relationship between ban node (outcome) causing factors node (features).
    The schedule must have the following structure:
    schedule[user_id][self.platform]
                                    [ "script"             | "ban"      | "triggered_rules" | "features"]
                                        [[list_of_actions] | 1_0_-1_ban | rule_name         | feature_name ]
    """

    def __init__(self, name: str, rule_str: str, outcome_name: str, rules: list, depends_on: list = [],
                 features: list = [], platform: str ="reddit"):
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


class SleepHoursRule(BasicRule):
    def __init__(self):
        super().__init__("sleep_hours_rule",
                         "if sleep_hours in ban_range -> ban and if sleep_hours in not_to_ban_range -> no ban",
                         [SleepHoursFeature()], "ban")

    def pred(self, schedule: dict, start_time: int = 0, curr_time: int = None):
        # update feature values
        self.feature_values(schedule, start_time, curr_time)

        # predict/evaluate ban
        for user, u_schedule in schedule.items():
            if "triggered_rules" not in u_schedule[self.platform]:
                u_schedule[self.platform]["triggered_rules"] = dict()
            u_schedule[self.platform]["triggered_rules"][self.name] = 0
            hours_of_sleep_dist = u_schedule[self.platform]["features"]["sleep_hours"]
            banned_by_min_sleep_hours = False
            ban_status_unknown = False
            for index, h in enumerate(hours_of_sleep_dist):
                if h != 0:
                    if isinstance(self.fit_data['sleep_hours'], dict):
                        # this assumes that first k-elements in hours_of_sleep_dist are sleep hours:
                        if (self.fit_data['sleep_hours']['banned']['min'] <= index <=
                                self.fit_data['sleep_hours']['banned']['max']):
                            banned_by_min_sleep_hours = True
                        elif (self.fit_data['sleep_hours']['notbanned']['min'] <= index <=
                                self.fit_data['sleep_hours']['notbanned']['max']):
                            ban_status_unknown = True
                        break
                    else:
                        # this assumes that first k-elements in hours_of_sleep_dist are sleep hours:
                        if index < self.fit_data['sleep_hours']:
                            banned_by_min_sleep_hours = True
                        break

            if isinstance(self.fit_data['sleep_hours'], dict):
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
                                   in enumerate(u_data[self.platform]["features"]["sleep_hours"]) if val > 0]),
                 "sleep_end": max([hour_idx for hour_idx, val
                                   in enumerate(u_data[self.platform]["features"]["sleep_hours"]) if val > 0])
                 }
                for u_id, u_data in schedule.items()]
        bans_df = pd.DataFrame.from_records(data)

        for feature_name in [f.name for f in self.features]:
            if feature_name not in self.fit_data:
                self.fit_data[feature_name] = {"sleep_hours": {"banned": {"min": None, "max": None},
                                                               "notbanned": {"min": None, "max": None}}}
            f_data = self.fit_data[feature_name]
            f_data['banned']['min'] = bans_df[bans_df['ban'] == 1]["sleep_bgn"].min()
            f_data['banned']['max'] = bans_df[bans_df['ban'] == 1]["sleep_end"].max()
            f_data['notbanned']['min'] = bans_df[bans_df['ban'] == 0]["sleep_bgn"].min()
            f_data['notbanned']['max'] = bans_df[bans_df['ban'] == 0]["sleep_end"].max()


class TotalNumberOfPostsRule(BasicRule):
    def __init__(self):
        super().__init__("total_number_of_posts",
                         "if total_number_of_posts in to_ban_range -> "
                         "ban and if total_number_of_posts in not_to_ban_range -> no ban",
                         [TotalNumberOfPostsFeature()], "ban")

    def pred(self, schedule: dict, start_time: int = 0, curr_time: int = None):
        # update feature values
        self.feature_values(schedule, start_time, curr_time)

        # predict/evaluate ban
        for user, u_schedule in schedule.items():
            if "triggered_rules" not in u_schedule[self.platform]:
                u_schedule[self.platform]["triggered_rules"] = dict()
            u_schedule[self.platform]["triggered_rules"][self.name] = 0
            n_posts = u_schedule[self.platform]["features"]["total_number_of_posts"]

            if isinstance(self.fit_data['total_number_of_posts'], dict):
                banned = False
                not_banned = False
                ban_unknown = False

                if (n_posts > self.fit_data["total_number_of_posts"]['banned']['min'] and
                        n_posts < self.fit_data["total_number_of_posts"]['banned']['max']):
                    banned = True
                elif (n_posts > self.fit_data["total_number_of_posts"]['notbanned']['min'] and
                      n_posts < self.fit_data["total_number_of_posts"]['notbanned']['max']):
                    not_banned = True
                else:
                    ban_unknown = True

                u_schedule[self.platform]["triggered_rules"][self.name] = 1 if banned else 0 if not_banned else -1

            else:
                u_schedule[self.platform]["triggered_rules"][self.name] = 1 if n_posts > self.fit_data[
                    "total_number_of_posts"] else 0

            if u_schedule[self.platform]["triggered_rules"][self.name] == 1:
                u_schedule[self.platform]["ban"] = 1
            else:
                u_schedule[self.platform]["ban"] = 0

    def fit(self, schedule: dict, start_time: int = 0, curr_time: int = None):
        data = [{"ban": u_data[self.platform]["ban"],
                 "n_posts": u_data[self.platform]["features"]["total_number_of_posts"]
                 } for u_id, u_data in schedule.items()]
        bans_df = pd.DataFrame.from_records(data)

        for feature_name in [f.name for f in self.features]:
            if feature_name not in self.fit_data:
                self.fit_data[feature_name] = {"total_number_of_posts": {"banned": {"min": None, "max": None},
                                                                         "notbanned": {"min": None, "max": None}}}
            f_data = self.fit_data[feature_name]
            f_data['banned']['min'] = bans_df[bans_df['ban'] == 1]["n_posts"].min()
            f_data['banned']['max'] = bans_df[bans_df['ban'] == 1]["n_posts"].max()
            f_data['notbanned']['min'] = bans_df[bans_df['ban'] == 0]["n_posts"].min()
            f_data['notbanned']['max'] = bans_df[bans_df['ban'] == 0]["n_posts"].max()

