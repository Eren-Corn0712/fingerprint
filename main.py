from engine.matcher import BaseMatcher
from utils import LOGGER


class Test(object):
    def __init__(self,
                 ):
        self.console = LOGGER

    def test_base_matcher(self):
        self.console.info("Test Base Matcher")
        base_matcher = BaseMatcher()
        self.console.info("Base Matcher is OK.")

    def __call__(self, *args, **kwargs):
        self.test_base_matcher()


if __name__ == '__main__':
    Test()()
