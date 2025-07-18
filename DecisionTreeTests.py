import unittest
from typing import List

from AnswerGroupFile import AnswerGroup
from ConditionFile import NumericCondition
from DecisionTree import DecisionTree, MAX_DIVISIONS_PER_RANGE


class MyTestCase(unittest.TestCase):

    def setUp(self):
        self.decision_tree = DecisionTree()

    def test_a_build_conditions_1(self):
        print("Starting test_a.")
        condition_list = self.decision_tree.build_conditions_for_range([200, 500, 400, 600])

        x_conditions:List[NumericCondition] = []

        y_conditions:List[NumericCondition] = []
        for condition in condition_list:
            if condition.attribute_name == "x":
                x_conditions.append(condition)
            if condition.attribute_name == "y":
                y_conditions.append(condition)

        # testing x conditions.
        # I am assuming you've added the conditions in order.
        x_diff = x_conditions[0].threshold_value - 200
        self.assertGreater(x_diff, 0, "The first division should be after the left margin.")
        for i in range(1,len(x_conditions)):
            self.assertLessEqual(abs(x_conditions[i].threshold_value - x_conditions[i-1].threshold_value - x_diff),
                                 1,
                                 "Conditions should be evenly spaced in x.")
        self.assertLess(x_conditions[-1].threshold_value, 400, "last condition in x should be less than right margin.")
        print ("  passed equal spacing in x.")

        self.assertEqual(len(x_conditions),
                         MAX_DIVISIONS_PER_RANGE,
                         f"There should be {MAX_DIVISIONS_PER_RANGE} conditions for x.")
        print("  passed x count.")

        # testing y conditions.
        # I am assuming you've added the conditions in order.
        y_diff = y_conditions[0].threshold_value - 500
        self.assertGreater(y_diff, 0, "The first division should be after the left margin.")
        for i in range(1,len(y_conditions)):
            self.assertLessEqual(abs(y_conditions[i].threshold_value - y_conditions[i-1].threshold_value-y_diff),
                                 1,
                                 "Conditions should be evenly spaced in y.")
        self.assertLess(y_conditions[-1].threshold_value, 600, "last condition in y should be less than right margin.")
        print ("  passed equal spacing in y.")
        self.assertEqual(len(y_conditions),
                         MAX_DIVISIONS_PER_RANGE,
                         f"There should be {MAX_DIVISIONS_PER_RANGE} conditions for y.")
        print("  passed y count.")
        print("Test_a completed.")

    def test_b_build_conditions_2(self):
        print("Starting test_b.")
        condition_list = self.decision_tree.build_conditions_for_range([200, 100, 205, 500])

        x_conditions: List[NumericCondition] = []
        y_conditions: List[NumericCondition] = []

        for condition in condition_list:
            if condition.attribute_name == "x":
                x_conditions.append(condition)
            if condition.attribute_name == "y":
                y_conditions.append(condition)

        # testing x conditions.
        # I am assuming you've added the conditions in order.

        self.assertEqual(x_conditions[0].threshold_value,
                         201,
                         f"The first division should be 201 but it was {x_conditions[0].threshold_value}.")
        for i in range(1, len(x_conditions)):
            self.assertEqual(
                abs(x_conditions[i].threshold_value - x_conditions[i - 1].threshold_value),
                1,
                "Conditions should be evenly spaced in x.")
        self.assertLess(x_conditions[-1].threshold_value, 205,
                        "last condition in x should be less than right margin.")
        print("  passed equal spacing in x.")

        self.assertEqual(len(x_conditions),
                          4,
                         f"There should be 4 conditions for x.")
        print("  passed x count.")

        # testing y conditions.
        # I am assuming you've added the conditions in order.
        y_diff = y_conditions[0].threshold_value - 100
        self.assertGreater(y_diff, 0, "The first division should be after the left margin.")
        for i in range(1, len(y_conditions)):
            self.assertLessEqual(
                abs(y_conditions[i].threshold_value - y_conditions[i - 1].threshold_value - y_diff),
                1,
                "Conditions should be evenly spaced in y.")
        self.assertLess(y_conditions[-1].threshold_value, 500,
                        "last condition in y should be less than right margin.")
        print("  passed equal spacing in y.")
        self.assertEqual( len(y_conditions),
                          MAX_DIVISIONS_PER_RANGE,
                          "There should be {MAX_DIVISIONS_PER_RANGE} conditions for y.")
        print("  passed y count.")
        print("Test_b completed.")

    def test_c_split_answerGroups(self):
        print("Starting test_c.")
        names = ["x","y"]
        answerGroupList = [AnswerGroup(question_name_list=names, answer_list=[235,347]),
                           AnswerGroup(question_name_list=names, answer_list=[264,316]),
                           AnswerGroup(question_name_list=names, answer_list=[222,322]),
                           AnswerGroup(question_name_list=names, answer_list=[285,374]),
                           AnswerGroup(question_name_list=names, answer_list=[271,395]),
                           AnswerGroup(question_name_list=names, answer_list=[236,381]),
                           AnswerGroup(question_name_list=names, answer_list=[201,355]),
                           AnswerGroup(question_name_list=names, answer_list=[219,348])]
        condition0 = NumericCondition("x",245)
        condition1 = NumericCondition("y",350)
        condition2 = NumericCondition("x",200)
        condition3 = NumericCondition("y",375)

        no_list, yes_list = self.decision_tree.split_answer_groups_by_condition(groups=answerGroupList,
                                                                                condition = condition0)
        self.assertEqual(len(no_list), 5, "For condition 0, there should be 5 no's.")
        self.assertEqual(len(yes_list), 3, "For condition 0, there should be 3 yes's.")
        print("  condition0 passed.")

        no_list, yes_list = self.decision_tree.split_answer_groups_by_condition(groups=answerGroupList,
                                                                                condition = condition1)
        self.assertEqual(len(no_list), 4, "For condition 0, there should be 5 no's.")
        self.assertEqual(len(yes_list), 4, "For condition 0, there should be 3 yes's.")
        print("  condition1 passed.")

        no_list, yes_list = self.decision_tree.split_answer_groups_by_condition(groups=answerGroupList,
                                                                                condition = condition2)
        self.assertEqual(len(no_list), 0, "For condition 0, there should be 5 no's.")
        self.assertEqual(len(yes_list), 8, "For condition 0, there should be 3 yes's.")
        print("  condition2 passed.")

        no_list, yes_list = self.decision_tree.split_answer_groups_by_condition(groups=answerGroupList,
                                                                                condition = condition3)
        self.assertEqual(len(no_list), 6, "For condition 0, there should be 5 no's.")
        self.assertEqual(len(yes_list), 2, "For condition 0, there should be 3 yes's.")
        print("  condition3 passed.")
        print("Test_c completed.")

    def test_d_counts_per_label(self):
        print("Starting test_d.")
        names = ["x", "y"]
        answerGroupList = [AnswerGroup(question_name_list=names, answer_list=[235, 347], label="land"),
                           AnswerGroup(question_name_list=names, answer_list=[264, 316], label="water"),
                           AnswerGroup(question_name_list=names, answer_list=[222, 322], label="land"),
                           AnswerGroup(question_name_list=names, answer_list=[285, 374], label="water"),
                           AnswerGroup(question_name_list=names, answer_list=[271, 395], label="water"),
                           AnswerGroup(question_name_list=names, answer_list=[236, 381], label="land"),
                           AnswerGroup(question_name_list=names, answer_list=[201, 355], label="water"),
                           AnswerGroup(question_name_list=names, answer_list=[219, 348], label="water")]

        answerGroupList2 = [AnswerGroup(question_name_list=names, answer_list=[235, 347], label="land"),
                           AnswerGroup(question_name_list=names, answer_list=[264, 316], label="land"),
                           AnswerGroup(question_name_list=names, answer_list=[222, 322], label="land"),
                           AnswerGroup(question_name_list=names, answer_list=[285, 374], label="land"),
                           AnswerGroup(question_name_list=names, answer_list=[271, 395], label="land"),
                           AnswerGroup(question_name_list=names, answer_list=[236, 381], label="land"),
                           AnswerGroup(question_name_list=names, answer_list=[201, 355], label="land"),
                           AnswerGroup(question_name_list=names, answer_list=[219, 348], label="land")]

        answerGroupList3 = []

        self.assertEqual(self.decision_tree.counts_per_label(answerGroupList),
                         {"land":3, "water":5})
        print("  assertion 0 passed.")
        self.assertEqual(self.decision_tree.counts_per_label(answerGroupList2),
                         {"land": 8, "water": 0})
        print("  assertion 1 passed.")
        self.assertEqual(self.decision_tree.counts_per_label(answerGroupList3),
                         {"land": 0, "water": 0})
        print("  assertion 2 passed.")
        print("Test_d completed.")

    def test_e_all_labels_match(self):
        print("Starting test_e.")
        names = ["x", "y"]
        answerGroupList = [AnswerGroup(question_name_list=names, answer_list=[235, 347], label="land"),
                           AnswerGroup(question_name_list=names, answer_list=[264, 316], label="water"),
                           AnswerGroup(question_name_list=names, answer_list=[222, 322], label="land"),
                           AnswerGroup(question_name_list=names, answer_list=[285, 374], label="water"),
                           AnswerGroup(question_name_list=names, answer_list=[271, 395], label="water"),
                           AnswerGroup(question_name_list=names, answer_list=[236, 381], label="land"),
                           AnswerGroup(question_name_list=names, answer_list=[201, 355], label="water"),
                           AnswerGroup(question_name_list=names, answer_list=[219, 348], label="water")]

        answerGroupList2 = [AnswerGroup(question_name_list=names, answer_list=[235, 347], label="land"),
                           AnswerGroup(question_name_list=names, answer_list=[264, 316], label="land"),
                           AnswerGroup(question_name_list=names, answer_list=[222, 322], label="land"),
                           AnswerGroup(question_name_list=names, answer_list=[285, 374], label="land"),
                           AnswerGroup(question_name_list=names, answer_list=[271, 395], label="land"),
                           AnswerGroup(question_name_list=names, answer_list=[236, 381], label="land"),
                           AnswerGroup(question_name_list=names, answer_list=[201, 355], label="land"),
                           AnswerGroup(question_name_list=names, answer_list=[219, 348], label="land")]

        answerGroupList3 = [AnswerGroup(question_name_list=names, answer_list=[235, 347], label="water"),
                           AnswerGroup(question_name_list=names, answer_list=[264, 316], label="water"),
                           AnswerGroup(question_name_list=names, answer_list=[222, 322], label="water")]

        self.assertFalse(self.decision_tree.all_labels_in_group_match(answerGroupList))
        print("  assertion 0 passed.")
        self.assertTrue(self.decision_tree.all_labels_in_group_match(answerGroupList2))
        print("  assertion 1 passed.")
        self.assertTrue(self.decision_tree.all_labels_in_group_match(answerGroupList3))
        print("  assertion 2 passed.")
        print("Test_e completed.")

    def test_f_most_frequent_label(self):
        print("Starting test_f.")
        names = ["x", "y"]
        answerGroupList = [AnswerGroup(question_name_list=names, answer_list=[235, 347], label="land"),
                           AnswerGroup(question_name_list=names, answer_list=[264, 316], label="water"),
                           AnswerGroup(question_name_list=names, answer_list=[222, 322], label="land"),
                           AnswerGroup(question_name_list=names, answer_list=[285, 374], label="water"),
                           AnswerGroup(question_name_list=names, answer_list=[271, 395], label="water"),
                           AnswerGroup(question_name_list=names, answer_list=[236, 381], label="land"),
                           AnswerGroup(question_name_list=names, answer_list=[201, 355], label="water"),
                           AnswerGroup(question_name_list=names, answer_list=[219, 348], label="water")]

        answerGroupList2 = [AnswerGroup(question_name_list=names, answer_list=[235, 347], label="land"),
                           AnswerGroup(question_name_list=names, answer_list=[264, 316], label="land"),
                           AnswerGroup(question_name_list=names, answer_list=[222, 322], label="land"),
                           AnswerGroup(question_name_list=names, answer_list=[285, 374], label="land"),
                           AnswerGroup(question_name_list=names, answer_list=[271, 395], label="land"),
                           AnswerGroup(question_name_list=names, answer_list=[236, 381], label="land"),
                           AnswerGroup(question_name_list=names, answer_list=[201, 355], label="land"),
                           AnswerGroup(question_name_list=names, answer_list=[219, 348], label="land")]

        answerGroupList3 = [AnswerGroup(question_name_list=names, answer_list=[235, 347], label="water"),
                           AnswerGroup(question_name_list=names, answer_list=[264, 316], label="land"),
                           AnswerGroup(question_name_list=names, answer_list=[222, 322], label="water"),
                            AnswerGroup(question_name_list=names, answer_list=[235, 355], label="land")]
        answerGroupList4 = []

        self.assertEqual(self.decision_tree.get_most_frequent_label_in_list(groups=answerGroupList), "water")
        print("  assertion 0 passed.")
        self.assertEqual(self.decision_tree.get_most_frequent_label_in_list(groups=answerGroupList2), "land")
        print("  assertion 1 passed.")
        self.assertEqual(self.decision_tree.get_most_frequent_label_in_list(groups=answerGroupList3), "land")
        print("  assertion 2 passed.")
        self.assertEqual(self.decision_tree.get_most_frequent_label_in_list(groups=answerGroupList4), "land")
        print("  assertion 3 passed.")
        print("Test_f completed.")

    def test_g_most_frequent_label(self):
        print("Starting test_g.")
        names = ["x", "y"]
        answerGroupList = [AnswerGroup(question_name_list=names, answer_list=[235, 347], label="land"),
                           AnswerGroup(question_name_list=names, answer_list=[264, 316], label="water"),
                           AnswerGroup(question_name_list=names, answer_list=[222, 322], label="land"),
                           AnswerGroup(question_name_list=names, answer_list=[285, 374], label="water"),
                           AnswerGroup(question_name_list=names, answer_list=[271, 395], label="water"),
                           AnswerGroup(question_name_list=names, answer_list=[236, 381], label="land"),
                           AnswerGroup(question_name_list=names, answer_list=[201, 355], label="water"),
                           AnswerGroup(question_name_list=names, answer_list=[219, 348], label="water")]

        answerGroupList2 = [AnswerGroup(question_name_list=names, answer_list=[235, 347], label="land"),
                           AnswerGroup(question_name_list=names, answer_list=[264, 316], label="land"),
                           AnswerGroup(question_name_list=names, answer_list=[222, 322], label="land"),
                           AnswerGroup(question_name_list=names, answer_list=[285, 374], label="land"),
                           AnswerGroup(question_name_list=names, answer_list=[271, 395], label="land"),
                           AnswerGroup(question_name_list=names, answer_list=[236, 381], label="land"),
                           AnswerGroup(question_name_list=names, answer_list=[201, 355], label="land"),
                           AnswerGroup(question_name_list=names, answer_list=[219, 348], label="land")]

        answerGroupList3 = [AnswerGroup(question_name_list=names, answer_list=[235, 347], label="water"),
                           AnswerGroup(question_name_list=names, answer_list=[264, 316], label="land"),
                           AnswerGroup(question_name_list=names, answer_list=[222, 322], label="water"),
                            AnswerGroup(question_name_list=names, answer_list=[235, 355], label="land")]
        answerGroupList4 = []

        self.assertAlmostEqual(self.decision_tree.gini_coefficient_for_list(answergroup_list=answerGroupList),
                               0.4688,
                               4)
        print("  assertion 0 passed.")
        self.assertAlmostEqual(self.decision_tree.gini_coefficient_for_list(answergroup_list=answerGroupList2),
                               0.0000,
                               4)
        print("  assertion 1 passed.")
        self.assertAlmostEqual(self.decision_tree.gini_coefficient_for_list(answergroup_list=answerGroupList3),
                               0.5,
                               4)
        print("  assertion 2 passed.")
        self.assertAlmostEqual(self.decision_tree.gini_coefficient_for_list(answergroup_list=answerGroupList4),
                               0,
                               4)
        print("  assertion 3 passed.")
        print("Test_g completed.")

if __name__ == '__main__':
    unittest.main()
