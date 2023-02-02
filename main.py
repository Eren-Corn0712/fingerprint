from matcher.engine.matcher import BaseMatcher
from matcher.data.base import BaseDataset
from matcher.data.dataset import FingerPrintDataset
from matcher.utils import LOGGER


class Test(object):
    def __init__(self,
                 ):
        self.console = LOGGER

    def test_base_matcher(self):
        self.console.info("Test Base Matcher")
        base_matcher = BaseMatcher()
        self.console.info("Base Matcher is OK.")

    def test_base_dataset(self):
        self.console.info("Test Base Dataset")
        base_dataset = BaseDataset("FingerPrintDataset")
        self.console.info("Test Base Dataset is OK")

    def test_finger_print_dataset(self):
        self.console.info("Test Finger Print Dataset")
        finger_print_dataset = FingerPrintDataset("FingerPrintDataset",)
        for i in finger_print_dataset:
            print(i)
        self.console.info("Test Finger Print Dataset is OK")

    def __call__(self, *args, **kwargs):
        self.test_base_matcher()
        # self.test_base_dataset()
        self.test_finger_print_dataset()


if __name__ == '__main__':
    Test()()
