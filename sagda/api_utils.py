import requests
import pandas as pd

def fetch_nasa_power_data(lat, lon, start_date, end_date, api_key):
    """Fetch weather data from NASA POWER API using an API key."""
    url = f"https://power.larc.nasa.gov/api/temporal/daily/point?start={start_date}&end={end_date}&latitude={lat}&longitude={lon}&format=JSON&parameters=T2M,PRECTOT&api_key={api_key}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()['properties']['parameter']
        df = pd.DataFrame({
            'date': pd.date_range(start=start_date, end=end_date),
            'temperature_min': data['T2M_MIN'],
            'temperature_max': data['T2M_MAX'],
            'rainfall': data['PRECTOT']
        })
        df['latitude'] = lat
        df['longitude'] = lon
        return df
    else:
        raise Exception(f"Error fetching data from NASA POWER API: {response.text}")


def fetch_openweathermap_data(lat, lon, api_key):
    """Fetch weather data from OpenWeatherMap API using an API key."""
    url = f"http://api.openweathermap.org/data/2.5/onecall?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()['daily']
        df = pd.DataFrame({
            'date': pd.date_range(start=pd.Timestamp.now(), periods=len(data)),
            'temperature_min': [day['temp']['min'] for day in data],
            'temperature_max': [day['temp']['max'] for day in data],
            'rainfall': [day.get('rain', 0) for day in data]
        })
        df['latitude'] = lat
        df['longitude'] = lon
        return df
    else:
        raise Exception(f"Error fetching data from OpenWeatherMap API: {response.text}")
