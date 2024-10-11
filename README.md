# **SADGA**: Synthetic Agriculture Data in Africa

**SADGA** (Synthetic Agriculture Data in Africa) is a Python library for generating, augmenting, and validating synthetic agricultural data for African regions. The library allows users to create realistic agricultural datasets, augment them using machine learning techniques, and validate the generated data against real data. SADGA can also integrate real-world data from APIs like NASA POWER and OpenWeatherMap to enhance synthetic data generation.

## **Key Features**

- **Generate Synthetic Agricultural Data**: Create time-series datasets with geospatial, climate, soil, crop yield, and fertilizer information.
- **Augment Data**: Augment real agricultural datasets using machine learning techniques such as random sampling, interpolation, linear regression, autoencoders, and GANs.
- **Validate Data**: Validate synthetic data using statistical tests (e.g., KS test) and compare it against real data.
- **API Integration**: Fetch real-world climate data using NASA POWER and OpenWeatherMap APIs, or generate synthetic data based on user-specified parameters.

## **Installation**

You can install SADGA from PyPI using `pip`:

```bash
pip install sadga
```

## **Dependencies**

- `pandas`
- `numpy`
- `requests`
- `scipy`
- `tensorflow`
- `scikit-learn`

## **Usage Examples**

### **1. Generate Synthetic Agricultural Data**

You can generate synthetic data using either random generation or real-world data from APIs (e.g., NASA POWER, OpenWeatherMap).

#### **Basic Usage** (Without API)

```python
from sadga import generate

# Generate synthetic data without API
data = generate(
    num_records=12,
    start_date="2020-01-01",
    end_date="2020-12-31",
    lat=34.0522,
    lon=-118.2437,
    crop_type="corn",
    frequency='monthly'
)

print(data)
```

#### **Usage with NASA POWER API**

```python
from sadga import generate

# Generate synthetic data using NASA POWER API
data = generate(
    start_date="2020-01-01",
    end_date="2020-12-31",
    lat=34.0522,
    lon=-118.2437,
    use_nasa=True,
    nasa_api_key="your_nasa_api_key",
    frequency='monthly'
)

print(data)
```

### **2. Augment Agricultural Data**

You can augment real data using techniques like random sampling, linear regression, autoencoders, or GANs.

#### **Linear Regression-Based Augmentation**

```python
from sadga import augment
import pandas as pd

# Example real data
real_data = pd.DataFrame({
    'soil_ph': [6.5, 6.3, 6.7],
    'temperature_min': [12, 14, 13],
    'rainfall': [40, 50, 45],
    'fertilizer_n': [100, 90, 95],
    'yield_kg_per_ha': [3200, 3100, 3300]
})

# Augment the data using linear regression
augmented_data = augment(
    real_data=real_data,
    num_augmented_records=50,
    start_date="2021-01-01",
    end_date="2021-12-31",
    technique='linear_regression',
    lat=34.0522,
    lon=-118.2437
)

print(augmented_data)
```

### **3. Validate Synthetic Data**

Validate the generated synthetic data by comparing it with real-world data.

#### **Validation Example**

```python
from sadga import validate
import pandas as pd

# Example real data
real_data = pd.DataFrame({
    'soil_ph': [6.5, 6.3, 6.7],
    'temperature_min': [12, 14, 13],
    'rainfall': [40, 50, 45],
    'fertilizer_n': [100, 90, 95],
    'yield_kg_per_ha': [3200, 3100, 3300]
})

# Example synthetic data
synthetic_data = pd.DataFrame({
    'soil_ph': [6.4, 6.5, 6.6],
    'temperature_min': [12.5, 13.0, 14.0],
    'rainfall': [43, 44, 46],
    'fertilizer_n': [101, 96, 99],
    'yield_kg_per_ha': [3205, 3105, 3295]
})

# Validate the synthetic data
validation_report = validate(synthetic_data, real_data=real_data)
print(validation_report)
```

### **4. API Integration**

SADGA integrates with NASA POWER and OpenWeatherMap APIs for fetching real-world climate data. You can pass your API keys as parameters.

#### **NASA POWER API Example**

```python
from sadga import generate

# Generate data using NASA POWER API
data = generate(
    start_date="2020-01-01",
    end_date="2020-12-31",
    lat=34.0522,
    lon=-118.2437,
    use_nasa=True,
    nasa_api_key="your_nasa_api_key",
    frequency='monthly'
)

print(data)
```

---

## **Project Structure**

```
SADGA/
│
├── sagda/                   # Main package directory
│   ├── __init__.py          # Initialize the package
│   ├── data_generation.py   # Generate synthetic agricultural data
│   ├── data_augmentation.py # Augment real data with synthetic data
│   ├── data_validation.py   # Validate synthetic data
│   ├── api_utils.py         # API integration functions
│   ├── utils.py             # Helper functions
├── tests/                   # Test directory
│   ├── test_data_generation.py # Test synthetic data generation
│   ├── test_data_augmentation.py # Test data augmentation
│   ├── test_data_validation.py # Test data validation
├── setup.py                 # Configuration for PyPI
├── README.md                # Project documentation
├── LICENSE                  # License file
├── MANIFEST.in              # Include non-Python files
├── requirements.txt         # Dependencies
```

---

## **Contributing**

We welcome contributions! Please fork the repository, create a new branch for your feature or bug fix, and submit a pull request.

---

## **License**

This project is licensed under the MIT License.
