import unittest
from modern_soar import SoarAgent

class TestSoarAgent(unittest.TestCase):
    def setUp(self):
        self.agent = SoarAgent("test_agent", "analysis")

    def test_initialization(self):
        self.assertEqual(self.agent.agent_id, "test_agent")
        self.assertEqual(self.agent.specialization, "analysis")

    def test_process_natural_language_instruction(self):
        # Mock or stub LLM response as needed
        result = self.agent.process_natural_language_instruction("Detect malware")
        self.assertIsNotNone(result)

if __name__ == "__main__":
    unittest.main()