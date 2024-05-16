import unittest
from schedules import *
from rules import *


class TestRules(unittest.TestCase):

    def test_complex_rule(self):
        # Ground Truth policy and thresholds
        gt_DAG = """
        dag {
        bb="-2.474,-3.873,2.319,2.622"
        ban [pos="0.717,-2.587"]
        narrative_ratio [pos="-0.565,-2.824"]
        sleep_hours [pos="0.086,-1.307"]
        total_lines_of_posts [pos="-0.990,-1.251"]
        total_number_of_posts [pos="0.113,0.560"]
        narrative_ratio -> ban
        sleep_hours -> ban
        total_lines_of_posts -> ban
        total_lines_of_posts -> sleep_hours
        total_number_of_posts -> ban
        total_number_of_posts -> sleep_hours
        }
        """
        gt_DAG = to_nx_DAG(gt_DAG)

        gt_feature_thresholds = {'sleep_hours': 6,
                                 'narrative_ratio': 0.7,
                                 'narrative': "un",
                                 'total_number_of_posts': 23,
                                 'total_lines_of_posts': 45
                                 }

        # keep ranges of the thresholds for each site
        activity_dist = {0: [0, 0, 0, 0, 0, 1, 10, 5, 10, 9, 1, 1, 10, 10, 10, 10, 5, 4, 4, 4, 10, 6, 1, 1],
                         1: [0, 0, 0, 1, 1, 1, 1, 5, 1, 9, 1, 1, 10, 10, 1, 1, 5, 4, 4, 4, 1, 6, 1, 1],
                         2: [0, 0, 0, 0, 0, 1, 1, 5, 1, 9, 1, 1, 10, 10, 1, 1, 5, 4, 4, 4, 1, 6, 1, 1],
                         }  # site-0
        # site-1
        # site-2
        narrative_dist = {0: {'un': 60, 'travel': 20, 'mistreatment': 15, 'prejudice': 5},
                          1: {'un': 40, 'anti': 20, 'travel': 10, 'pro': 15, 'infrastructure': 15},
                          2: {'environmentalism': 20, 'covid': 20, 'debt': 20, 'pro': 25, 'un': 10}
                          }  # %
        number_of_actions_per_agent = {0: [2, 25],
                                       1: [3, 25],
                                       2: [3, 25]
                                       }
        number_of_lines_per_post = {0: [1, 3],
                                    1: [1, 4],
                                    2: [1, 2]
                                    }
        # Initial schedule before rescheduling
        schedule_before_rescheduling = generate_script(n_agents=100, n_sites=3, start_time=0,
                                                       n_actions=number_of_actions_per_agent,
                                                       activity_dist=activity_dist,
                                                       narrative_dist=narrative_dist,
                                                       n_lines_per_post=number_of_lines_per_post)

        # schedule_before_rescheduling["puppet_0_0"]["reddit"]

        sleep_h_rule = SleepHoursRule()
        sleep_h_rule.fit_data = {sleep_h_rule.features[0].name: {"banned": {"min": 0, "max": 6}, "notbanned": {"min": 8, "max": 22}}}
        sleep_h_rule.pred(schedule_before_rescheduling, 0)
        sleep_h_rule.fit(schedule_before_rescheduling, 0)

        sleep_h_rule2 = SleepHoursRule()
        sleep_h_rule2.fit_data = {sleep_h_rule2.features[0].name: {"banned": {"min": 0, "max": 6}, "notbanned": {"min": 8, "max": 22}}}
        sleep_h_rule2.pred(schedule_before_rescheduling, 0)
        sleep_h_rule2.fit(schedule_before_rescheduling, 0)

        narrative_ratio_rule = NarrativeRatioRule(narrative="un")
        narrative_ratio_rule.fit_data = {narrative_ratio_rule.features[0].name: {"banned": {"min": 0.5, "max": 1.0}, "notbanned": {"min": 0, "max": 0.3}}}
        narrative_ratio_rule.pred(schedule_before_rescheduling, 0)
        narrative_ratio_rule.fit(schedule_before_rescheduling, 0)


        total_lines_rule = TotalLinesOfPostsRule()
        total_lines_rule.fit_data = {total_lines_rule.features[0].name: {"banned": {"min": 18, "max": 100}, "notbanned": {"min": 0, "max": 5}}}
        total_lines_rule.pred(schedule_before_rescheduling, 0)
        total_lines_rule.fit(schedule_before_rescheduling, 0)

        complex_rule = AggregateRule("Complex Rule", "Two sleep rules, narrative_ratio_rule and total_lines_rule", "ban",
                                     [sleep_h_rule, sleep_h_rule2, narrative_ratio_rule, total_lines_rule, TotalNumberOfPostsCauseSleepHoursRule()])

        dag = complex_rule.get_DAG()
        print(dag)
        # print(schedule_before_rescheduling)

        self.assertIsNotNone(schedule_before_rescheduling)
        self.assertIsNotNone(dag)


if __name__ == '__main__':
    unittest.main()
