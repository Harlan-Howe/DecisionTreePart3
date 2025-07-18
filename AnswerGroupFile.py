
from typing import Generic, TypeVar, List, Optional, Any


class AnswerGroup():
    """
    An "AnswerGroup" corresponds to one complete set of questions answered by the user.
    (Later, it will also correspond to one row of data in our spreadsheet.)
    It should have a matching set of attribute names and the values the user selected.
    It might also have a "Label" - used by the computer to train this table and/or to check correctness. But for a
        simple recommendation for a user input, this isn't needed.
    """
    def __init__(self, question_name_list: List[str], answer_list: List[Any], label: Optional[str] = None):
        self.answers = {}

        for i in range(len(question_name_list)):
            self.answers[question_name_list[i]] = answer_list[i]
        self.label = label

    def get_attribute_for_name(self, name: str):
        return self.answers[name]

    def get_label(self) -> str:
        return self.label

    def set_label(self, label:str):
        self.label = label

    def __repr__(self):
        """
        This creates a string describing this AnswerGroup, in case we want to print it out.
        :return: a descriptive string
        """
        result = ""
        for key in self.answers.keys():
            result += f"{key}:{self.answers[key]}\t"
        if self.label is not None:
            result+= f"--> [{self.label}]"
        return result


# test code. Run this particular file to see it work.
if __name__=="__main__":
    ag_demo = AnswerGroup(question_name_list=["A","B","C"], answer_list=[0,1,2], label="*")
    print(ag_demo)
    print(ag_demo.get_attribute_for_name("B"))
    print(ag_demo.get_label())