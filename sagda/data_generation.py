import pandas as pd
import numpy as np
from .api_utils import fetch_nasa_power_data, fetch_openweathermap_data

def generate(num_records=None, start_date=None, end_date=None, lat=None, lon=None, crop_type=None, use_nasa=False,
             use_openweather=False, nasa_api_key=None, openweather_api_key=None, frequency='monthly'):
    """
    Generate synthetic agricultural data (geospatial, climate, soil, crop yield, fertilizer) as a time series.

    :param num_records: Number of synthetic records to generate (if APIs are not used).
    :param start_date: Start date for the time series.
    :param end_date: End date for the time series.
    :param lat: Latitude for fetching real data via API.
    :param lon: Longitude for fetching real data via API.
    :param crop_type: Optional crop type to generate data for.
    :param use_nasa: Boolean flag to fetch data from NASA POWER API.
    :param use_openweather: Boolean flag to fetch data from OpenWeatherMap API.
    :param nasa_api_key: API key for NASA POWER API (if using NASA).
    :param openweather_api_key: API key for OpenWeatherMap API (if using OpenWeather).
    :param frequency: Frequency for the time series ('daily', 'weekly', 'monthly', 'yearly').
    :return: DataFrame of synthetic agricultural data including latitude and longitude.
    """
    
    if not start_date or not end_date:
        raise ValueError("Start date and end date are required.")
    
    if not num_records and not (lat and lon):
        raise ValueError("Either 'num_records' or 'lat/lon' for API fetching must be provided.")

    # Generate date range based on the chosen frequency
    if frequency == 'weekly':
        date_range = pd.date_range(start=start_date, end=end_date, freq='W')
    elif frequency == 'yearly':
        date_range = pd.date_range(start=start_date, end=end_date, freq='Y')
    else:  # Default is monthly
        date_range = pd.date_range(start=start_date, end=end_date, freq='M')

    data = []
    
    # Fetch data from NASA POWER API if the flag is set
    if use_nasa and lat and lon:
        if not nasa_api_key:
            raise ValueError("NASA API key is required when using NASA POWER API.")
        nasa_data = fetch_nasa_power_data(lat, lon, start_date, end_date, nasa_api_key)
        nasa_data['latitude'] = lat
        nasa_data['longitude'] = lon
        data.append(nasa_data)
    
    # Fetch data from OpenWeatherMap API if the flag is set
    if use_openweather and lat and lon:
        if not openweather_api_key:
            raise ValueError("OpenWeather API key is required when using OpenWeatherMap API.")
        openweather_data = fetch_openweathermap_data(lat, lon, openweather_api_key)
        openweather_data['latitude'] = lat
        openweather_data['longitude'] = lon
        data.append(openweather_data)

    # If no API is used, generate synthetic data
    if not use_nasa and not use_openweather:
        if not num_records:
            raise ValueError("Please provide either 'num_records' or set API flags.")
        
        synthetic_data = {
            'date': date_range[:num_records],
            'latitude': [lat] * num_records,
            'longitude': [lon] * num_records,
            'crop_type': np.random.choice([crop_type] if crop_type else ['wheat', 'corn', 'soybean', 'rice'], num_records),
            'soil_ph': np.random.normal(loc=6.5, scale=0.5, size=num_records),
            'soil_moisture': np.random.uniform(20, 50, num_records),
            'temperature_min': np.random.uniform(10, 20, num_records),
            'temperature_max': np.random.uniform(25, 35, num_records),
            'rainfall': np.random.uniform(0, 50, num_records),
            'fertilizer_n': np.random.uniform(50, 150, num_records),
            'fertilizer_p': np.random.uniform(20, 80, num_records),
            'fertilizer_k': np.random.uniform(30, 90, num_records),
            'yield_kg_per_ha': np.random.normal(loc=3000, scale=500, size=num_records)
        }
        data.append(pd.DataFrame(synthetic_data))

    return pd.concat(data, ignore_index=True)
