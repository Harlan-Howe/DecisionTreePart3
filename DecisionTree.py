import time
from typing import Optional, List, Tuple, Dict

import cv2
import numpy as np

from AnswerGroupFile import AnswerGroup
from ConditionFile import NumericCondition
from NodeFile import GenericNode, BranchNode, LeafNode

MAX_DIVISIONS_PER_RANGE = 9
MAX_DEPTH = 25
MIN_ITEMS_PER_BRANCH_NODE = 3
VERBOSE = True

class DecisionTree:

    def __init__(self):
        self.decision_tree_root: Optional[GenericNode] = None
        self.max_depth_used = 0
        self.debug_canvas: Optional[np.ndarray] = None

    def build_tree(self,
                   training_data: List[AnswerGroup],
                   bounds: List[int]|Tuple[int, int, int, int],
                   debug_canvas: np.ndarray = None):
        """
        Generates the tree built out of BranchNodes and LeafNodes that will allow this tree to make predictions.
        :param training_data: the collection of labelled AnswerGroups that will be used to train this tree.
        :param bounds: the size of the "world" for the (x,y) coordinates of the points in training_data
        :param debug_canvas: an optional ndarray that can be drawn into, the same size as the bounds, with a depth of 3
        for color.
        :return: None
        """
        self.debug_canvas = debug_canvas
        print(f"I am about to build a tree with {len(training_data)} data points.")
        start_time = time.perf_counter()
        self.decision_tree_root = self.make_node_for_instances_at_depth(answergroup_list=training_data,
                                                                        depth = 0,
                                                                        range=bounds)
        end_time = time.perf_counter()
        print(f"Tree built in {(end_time-start_time):0.6f} seconds.")
        if isinstance(self.decision_tree_root, BranchNode):
            print(f"The root node of the tree is based on Condition: {BranchNode(self.decision_tree_root).my_condition}.")

    def build_conditions_for_range(self, range: List[int]|Tuple[int, int, int, int]) -> List[NumericCondition]:
        """
        builds a collection of "Condition" objects to split this range into up to MAX_DIVISIONS_PER_RANGE slices in each
        of the x and y dimensions. The conditions should be evenly spaced within each direction with integer spacing. If
        the range is too small to accommodate MAX_DIVISIONS_PER_RANGE conditions, then use as many conditions as will
        fit.
        Conditions should not include the max/min values in range.
        :param range: (x_min, y_min, x_max, y_max) - max is assumed to be >= min.
        :return: a single list of both x and y conditions for this range, as many as 2 * MAX_DIVISIONS_PER_RANGE.
        """
        x_division_size = max(1, int((range[2] - range[0]) / (MAX_DIVISIONS_PER_RANGE)))
        y_division_size = max(1, int((range[3] - range[1]) / (MAX_DIVISIONS_PER_RANGE)))
        conditions_out: List[NumericCondition] = []

        # TODO: write this method, generating a NumericCondition for each x or y condition, and appending it to
        #  conditions_out.
        #  This is tested in tests a & b.

        return conditions_out

    def split_answer_groups_by_condition(self,
                                         groups: List[AnswerGroup],
                                         condition: NumericCondition) -> Tuple[List[AnswerGroup], List[AnswerGroup]]:
        """
        loops through all the AnswerGroup instances in groups and copies each one into either "no" or "yes" lists based
        on whether they meet the given Condition. The original group is unchanged.
        :param groups: a list of AnswerGroup objects
        :param condition: a condition to apply to each of the AnswerGroup objects
        :return: a list of the ag's that don't meet the condition, and a list of the ag's that do meet the condition.

        note: these conditions are spatial (x/y), NOT labels (land vs. water).
        """
        no_list: List[AnswerGroup] = []
        yes_list: List[AnswerGroup] = []

        # TODO: write this method.
        #  This is tested in test c.

        return no_list, yes_list

    def counts_per_label(self, groups: List[AnswerGroup]) -> Dict[str,int]:
        """
        counts up the number of "land" and "water" labels in the training groups given
        :param groups: AnswerGroups, each with a label of "land" or "water"
        :return: a dictionary with the number of land labels and the number of water labels.
        """
        counts = {"land":0, "water":0}
        # TODO: write this method, updating the values stored in the counts dictionary.
        #  This is tested in test d.
        return counts

    def all_labels_in_group_match(self, groups: List[AnswerGroup]) -> bool:
        """
        determines whether these groups are all "land" or all "water"
        :param groups: a list of labeled answer groups
        :return: whether they all have the same label.
        """
        # TODO: write this method. Note that you could use the results of counts_per_label, but there may be a more
        #   efficient way to find this answer that does this potentially without always having to check the complete
        #   list.
        #   This is tested in test e.
        return True  # replace this.

    def get_most_frequent_label_in_list(self, groups: List[AnswerGroup]) -> str:
        """
        returns the label that appears most often in this collection of labeled AnswerGroups. If it is a tie, selects
        "land."
        :param groups: a list of AnswerGroups, each labeled "land" or "water".
        :return: either "land" or "water".
        """
        # TODO: write this method.
        #  This is tested in test f.
        return "land" # replace this.

    def gini_coefficient_for_list(self, answergroup_list: List[AnswerGroup]) -> float:
        """
        Find the gini coefficient for the given list of AnswerGroups, based on their labels (i.e., "land" vs. "water).
        1 - (Pa**2 + Pb**2 + Pc**2 +...)   where Pa = Num "a" labels/total number of labels.
        (Note: "**2" means "Squared" in python.)
        :param answergroup_list: the list to consider
        :return: the gini coefficient of that list, or zero, if list is empty.
        """
        N = len(answergroup_list)  # total number of labels

        if N == 0:  # if this is an empty list, bail out now so we don't divide by zero later.
            return 0

        # get the dictionary of label_names --> number of that label found...
        # e.g., {"land": 12, "water": 4}
        label_counts = self.counts_per_label(groups=answergroup_list)

        # TODO: Start with gini = 1, and then loop through the values in label_counts and find P for each category.
        #  Subtract P**2 from gini each time.
        #   This is tested in test g.

        return 0 # replace this.

    def make_node_for_instances_at_depth(self,
                                         answergroup_list: List[AnswerGroup],
                                         depth: int,
                                         range: List[int]|Tuple[int, int, int, int]) -> GenericNode:
        """
        creates a Node (either a LeafNode or a BranchNode) based on the collection of answergroups given.
        A leafnode will be created if the size of the answergroup is small, if the depth is at the max depth, or if
        all the answergroups given have the same label.

        If this is going to be a LeafNode, then it will base it on the majority of labels in the answergroup_list.
        If this is going to be a BranchNode, then it will be one that generates its own "children" (yes/no) Nodes.
        :param answergroup_list: a list of answergroups to consider when making this Node.
        :param depth: the depth in the tree where this node will go. Used to keep track of when to stop!
        :return: The Node we are creating!
        """
        N = len(answergroup_list)
        if VERBOSE:
            print(f"I've been asked to make a node at depth {depth}, using {N} AnswerGroups.")

        condition_list = self.build_conditions_for_range(range)

        # checks whether this is one of the four conditions to make a leaf node.
        #  (I've written this part for you. - HH)
        if (depth == MAX_DEPTH or
                N < MIN_ITEMS_PER_BRANCH_NODE or
                len(condition_list)==0 or
                self.all_labels_in_group_match(answergroup_list)):
            return self.make_leaf_node(answergroup_list, depth, range)

        min_gini_index = 1000  # a ridiculously high number to start; we're likely to find values less than one.
        best_condition = None
        best_no_group = None
        best_yes_group = None

        # TODO: loop through all the conditions in condition_list and find the one with the lowest Gini Index, updating
        #  min_gini_index, best_condition, best_no_group, and best_yes_group. I've given you an outline below:
        #  (No unit test for this one.)

        # Loop through all the conditions in our list of possible conditions.

            # Use this condition to split our list of AnswerGroups into "no" and "yes" lists.

            # Calculate P and gini for the two sublists (no/yes) we just made:

            # Based on "p_yes," "p_no," "gini_yes," and "gini_no," calculate the gini index for this choice. Then,
            # if this gini_index is better than the others we've seen so far, update "best_condition," "best_yes_group,"
            # "best_no_group" and "min_gini_index"


        if VERBOSE:
            print(f"\tThe best condition was: {best_condition}, which had a low gini Index of {min_gini_index:3.3f}.")
            print(f"\tThis split the dataset into {len(best_yes_group)} 'yes' values with a gini coefficient of ",
                  end="")
            print(
                f"{self.gini_coefficient_for_list(best_yes_group):3.3f} and {len(best_no_group)} 'no' values with a gini ",
                end="")
            print(f"coefficient of {self.gini_coefficient_for_list(best_no_group):3.3f}.")

        # The remainder of this method is about building a branchNode based on the best condition you found,
        #  constructing the "no" and "yes" children, and returning this new BranchNode to be added to its parent or
        #  returned as the root Node. (I've written the remainder for you. - HH)

        # Make the node we're about to return, based on the favorite condition we just found.
        return self.make_branch_node(best_condition, best_no_group, best_yes_group, depth, range)

    def make_branch_node(self,
                         best_condition: NumericCondition,
                         best_no_group: List[AnswerGroup],
                         best_yes_group: List[AnswerGroup],
                         depth: int,
                         range: List[int]|Tuple[int, int, int, int]) -> BranchNode:
        result = BranchNode(best_condition, depth=depth)
        # Build subranges - how have we just split the rectangle on the map into two?
        # Reminder: we have been asking whether points are _greater_ than the threshold_value.
        if best_condition.attribute_name == "x":
            no_range = (range[0], range[1], best_condition.threshold_value, range[3])  # left
            yes_range = (best_condition.threshold_value, range[1], range[2], range[3])  # right
        else:
            no_range = (range[0], range[1], range[2], best_condition.threshold_value)  # top
            yes_range = (range[0], best_condition.threshold_value, range[2], range[3])  # bottom
        # but before we return the new BranchNode, create and connect the sub nodes for no and yes.
        result.set_no_node(self.make_node_for_instances_at_depth(answergroup_list=best_no_group,
                                                                 depth=depth + 1,
                                                                 range=no_range))
        result.set_yes_node(self.make_node_for_instances_at_depth(answergroup_list=best_yes_group,
                                                                  depth=depth + 1,
                                                                  range=yes_range))
        # Now that we have a fully-formed node and its children, send it back to the method that called this one.
        return result

    def make_leaf_node(self,
                       answergroup_list: List[AnswerGroup],
                       depth:int,
                       range: List[int]|Tuple[int, int, int, int]) -> LeafNode:
        most_frequent_label = self.get_most_frequent_label_in_list(answergroup_list)
        if VERBOSE:
            print(f"I'll make a LeafNode: [[{most_frequent_label}]]")
        if self.debug_canvas is not None:
            cv2.rectangle(img=self.debug_canvas, pt1=(range[0], range[1]), pt2=(range[2], range[3]),
                          color=(0, 196, 196),
                          thickness=5)
        if depth > self.max_depth_used:
            self.max_depth_used = depth
        return LeafNode(most_frequent_label, depth=depth)

    def predict(self, answer_group: AnswerGroup) -> str:
        return self.decision_tree_root.predict(answer_group)