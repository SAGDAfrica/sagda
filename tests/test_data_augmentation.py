import unittest
import pandas as pd
from sagda.data_augmentation import augment

class TestDataAugmentation(unittest.TestCase):

    def test_augment_with_random(self):
        """Test augmentation using random sampling."""
        real_data = pd.DataFrame({
            'soil_ph': [6.5, 6.3, 6.7],
            'temperature_min': [12, 14, 13],
            'rainfall': [40, 50, 45],
            'fertilizer_n': [100, 90, 95],
            'yield_kg_per_ha': [3200, 3100, 3300]
        })

        augmented_data = augment(real_data, num_augmented_records=5, technique='random')
        self.assertEqual(len(augmented_data), 8)  # 3 real + 5 synthetic
        self.assertIn('yield_kg_per_ha', augmented_data.columns)

    def test_augment_with_linear_regression(self):
        """Test augmentation using linear regression."""
        real_data = pd.DataFrame({
            'soil_ph': [6.5, 6.3, 6.7],
            'temperature_min': [12, 14, 13],
            'rainfall': [40, 50, 45],
            'fertilizer_n': [100, 90, 95],
            'yield_kg_per_ha': [3200, 3100, 3300]
        })

        augmented_data = augment(real_data, num_augmented_records=5, technique='linear_regression')
        self.assertEqual(len(augmented_data), 8)  # 3 real + 5 synthetic
        self.assertIn('yield_kg_per_ha', augmented_data.columns)

if __name__ == '__main__':
    unittest.main()
