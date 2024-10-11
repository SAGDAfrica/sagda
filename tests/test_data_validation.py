import unittest
import pandas as pd
from sagda.data_validation import validate

class TestDataValidation(unittest.TestCase):

    def test_validate_data_without_real_data(self):
        """Test validation of synthetic data without real data comparison."""
        synthetic_data = pd.DataFrame({
            'soil_ph': [6.5, 6.4, 6.6],
            'temperature_min': [12, 13, 14],
            'rainfall': [40, 42, 43],
            'fertilizer_n': [100, 105, 110],
            'yield_kg_per_ha': [3200, 3150, 3300]
        })

        validation_report = validate(synthetic_data)
        self.assertIn('summary_statistics', validation_report)
        self.assertIn('correlation_matrix', validation_report)

    def test_validate_data_with_real_data(self):
        """Test validation of synthetic data with real data comparison."""
        real_data = pd.DataFrame({
            'soil_ph': [6.5, 6.3, 6.7],
            'temperature_min': [12, 14, 13],
            'rainfall': [40, 50, 45],
            'fertilizer_n': [100, 90, 95],
            'yield_kg_per_ha': [3200, 3100, 3300]
        })

        synthetic_data = pd.DataFrame({
            'soil_ph': [6.4, 6.5, 6.6],
            'temperature_min': [12.5, 13.0, 14.0],
            'rainfall': [43, 44, 46],
            'fertilizer_n': [101, 96, 99],
            'yield_kg_per_ha': [3205, 3105, 3295]
        })

        validation_report = validate(synthetic_data, real_data)
        self.assertIn('distribution_tests', validation_report)
        self.assertIn('ks_statistic', validation_report['distribution_tests']['soil_ph'])

if __name__ == '__main__':
    unittest.main()
