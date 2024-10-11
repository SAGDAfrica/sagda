import unittest
from sagda.data_generation import generate

class TestDataGeneration(unittest.TestCase):

    def test_generate_data_without_api(self):
        """Test synthetic data generation without using APIs."""
        data = generate(num_records=10, start_date="2020-01-01", end_date="2020-12-31", lat=34.0522, lon=-118.2437)
        self.assertEqual(len(data), 10)
        self.assertIn('latitude', data.columns)
        self.assertIn('longitude', data.columns)

    def test_generate_data_with_nasa_api(self):
        """Test synthetic data generation using NASA API (mocked)."""
        # For this, you would mock the API call.
        # In a real test, you would use a tool like `unittest.mock` to mock the API response.
        data = generate(start_date="2020-01-01", end_date="2020-12-31", lat=34.0522, lon=-118.2437, use_nasa=True, nasa_api_key="mocked_api_key")
        self.assertIn('temperature_min', data.columns)
        self.assertIn('rainfall', data.columns)

if __name__ == '__main__':
    unittest.main()
