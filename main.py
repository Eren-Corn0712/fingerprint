from engine.matcher import BaseMatcher
from data.base import BaseDataset
from utils import LOGGER


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

    def __call__(self, *args, **kwargs):
        self.test_base_matcher()
        self.test_base_dataset()


if __name__ == '__main__':
    Test()()
