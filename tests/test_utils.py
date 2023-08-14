import logging
import unittest
import os
import sys
from pathlib import Path
# project modules
from utils import load_json

FILE = Path(__file__).resolve()
test_dir = FILE.parents[0]
project_dir = FILE.parents[1]
sys.path.append(str(project_dir))
sys.path.append(str(test_dir))
os.chdir(test_dir)
logger = logging.getLogger('unittest_log')


class TestUtilities(unittest.TestCase):
    def test_givenAJSONFile_whenJSONDataIsRequired_thenLoadJSONData(self):
        expected_result = './outputs'
        json_data = load_json(os.path.join(project_dir, 'configs.json'))
        self.assertEqual(json_data['common_attr']['output_path'], expected_result)
