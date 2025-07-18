from abc import ABC
from AnswerGroupFile import AnswerGroup
from typing import Optional

class GenericCondition(ABC):
    """
    An abstract class that we won't instantiate directly - it represents a question to be asked about an attribute of
    an AnswerGroup.
    Known subclasses: NumericalCondition, CategoryCondition
    """
    def __init__(self, attribute_name:str):
        self.attribute_name = attribute_name

    def ask(self, instance: AnswerGroup) -> bool:
        pass

    def __repr__(self):
        pass

class NumericCondition(GenericCondition):
    """
    This class represents a question to be asked about a given AnswerGroup object: is the value for the attribute held
    by this Condition greater than the threshold held by this condition?
    """
    def __init__(self, attribute_name: str, threshold: float):
        """
        sets this NumericCondition object up with the given attribute name and threshold.
        :param attribute_name: the name of the attribute we'll be checking
        :param threshold: the numerical value that will determine if the value stored in the attribute is big enough.
        """
        super(NumericCondition, self).__init__(attribute_name)
        self.threshold_value = threshold

    def ask(self, answer_group: AnswerGroup) -> bool:
        """
        asks the given AnswerGroup has a value for this condition's attribute that is larger than this condition's
        threshold.
        :param answer_group: the AnswerGroup in question
        :return: whether the AnswerGroup instance has a large enough value
        """
        return answer_group.get_attribute_for_name(self.attribute_name) > self.threshold_value

    def __repr__(self):
        return f"Is {self.attribute_name} > {self.threshold_value}?"

class CategoryCondition(GenericCondition):
    def __init__(self, attribute_name: str, value: str):
        super(CategoryCondition, self).__init__(attribute_name)
        self.value = value

    def ask(self, instance: AnswerGroup) -> bool:
        return instance.get_attribute_for_name(self.attribute_name) == self.value

    def __repr__(self):
        return f"Is {self.attribute_name} equal to \"{self.value}\"?"

