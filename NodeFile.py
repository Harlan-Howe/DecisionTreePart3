from abc import ABC
from typing import Optional

from AnswerGroupFile import AnswerGroup
from ConditionFile import GenericCondition


class GenericNode(ABC):
    """
    Node is an abstract class - it is a framework for other classes so that both "BranchNode" and "LeafNode" (see below)
          objects can both be considered "Nodes," with similar behaviors. But you'll never actually create a plain,
          vanilla Node.
    """
    def __init__(self, depth: int=0):
        self.my_depth = depth

    def predict(self, answer_group: AnswerGroup) -> str:
        pass

# =====================================================================================================================
class BranchNode(GenericNode):
    """
        holds one of our conditions, and links to two other Nodes, one used if this condition is "yes," and one for "no."
        When asked to make a prediction for an Instance, this node will check its condition and delegate the prediction
            to whichever node (yes or no) is appropriate.
    """
    def __init__(self, condition: GenericCondition, depth: int):
        """
        sets up this BranchNode with the condition that will be asked and information about how far this node is from
        the start of the questioning.
        :param condition: the Condition object that will serve as the question this BranchNode asks
        :param depth: the distance this question is from the start of the questioning (i.e., the root of this Decision
        Tree). Another way of looking at this is, which number question is this that is being asked of this particular
        AnswerGroup instance?
        """
        super(BranchNode, self).__init__(depth)
        self.my_condition = condition
        self.__yes_node__:Optional[GenericNode] = None
        self.__no_node__:Optional[GenericNode] = None

    def set_yes_node(self, node: GenericNode):
        self.__yes_node__ = node

    def set_no_node(self, node: GenericNode):
        self.__no_node__ = node

    def predict(self, answer_group: AnswerGroup) -> str:
        """
        Given an instance, pick which category this instance belongs to. This is done recursively - i.e., by asking
        another Node for the answer.
        :param answer_group: the instance of AnswerGroup for which we wish to predict
        :return: which category is predicted for this Instance.
        """
        if self.my_condition.ask(answer_group):
            return self.__yes_node__.predict(answer_group)
        else:
            return self.__no_node__.predict(answer_group)

    def __repr__(self):
        """
        get a representation of this node (and its children) that we can draw to the screen
        :return: a string describing this node and its children
        """
        indent = "\t"*self.my_depth
        return f"{self.__yes_node__}{indent}{self.my_condition}\n{self.__no_node__}"

# ====================================================================================================================
class LeafNode(GenericNode):
    """
    The LeafNode is the end of the line in our questioning - it doesn't divide things up any further; it just gives
    us an answer to which category to recommend, regardless of the Instance. In other words, by the time we get to
    a leaf node, our mind is made up.
    """
    def __init__(self, category: str, depth: int):
        super(LeafNode, self).__init__(depth)
        self.my_label = category

    def predict(self, answer_group: AnswerGroup) -> str:
        """
        returns the recommended category for this Leaf
        :param answer_group: the instance for which we are making a prediction (ignored by the leaf)
        :return: the predicted category (a.k.a. label)
        """
        return self.my_label

    def __repr__(self):
        """
        get a representation of this node that we can draw to the screen. It will be indented proportional to its depth.
        :return: a string describing this LeafNode
        """
        indent = "\t"*self.my_depth
        return f"{indent}[[{self.my_label}]]\n"
