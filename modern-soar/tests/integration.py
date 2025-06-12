import unittest
from modern_soar import CyberSecurityWorkflow

class TestWorkflowIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_cybersecurity_workflow(self):
        workflow = CyberSecurityWorkflow()
        network_data = "unusual_network_traffic_detected"
        result = await workflow.process_security_incident(network_data)
        self.assertIn("incident_id", result)

if __name__ == "__main__":
    unittest.main()