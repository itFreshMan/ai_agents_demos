import requests
import datetime
import wikipedia

def get_weather_by_city_func(city_name: str) -> str:

    # First get the coordinates for the city
    coords_result = __get_city_coordinates(city_name)
    
    if 'error' in coords_result:
        return f"Could not find coordinates for {city_name}: {coords_result['error']}"
    
    # Then get the weather for those coordinates
    weather_result = __get_current_temperature(
        latitude=coords_result['latitude'],
        longitude=coords_result['longitude']
    )

    return f"Weather for {coords_result['display_name']}: {weather_result}"


def search_wikipedia_func(query: str) -> str:
    """Run Wikipedia search and get page summaries."""
    page_titles = wikipedia.search(query)
    summaries = []
    for page_title in page_titles[: 3]:
        try:
            wiki_page =  wikipedia.page(title=page_title, auto_suggest=False)
            summaries.append(f"Page: {page_title}\nSummary: {wiki_page.summary}")
        except (
            wikipedia.exceptions.PageError,
            wikipedia.exceptions.DisambiguationError,
        ):
            pass
    if not summaries:
        return "No good Wikipedia Search Result was found"
    return "\n\n".join(summaries)



def __get_city_coordinates(city_name: str) -> dict:
    """Get latitude and longitude coordinates for a given city name using OpenStreetMap Nominatim API."""
    
    # Using OpenStreetMap Nominatim API (free, no API key required)
    BASE_URL = "https://nominatim.openstreetmap.org/search"
    
    # Parameters for the request
    params = {
        'q': city_name,
        'format': 'json',
        'limit': 1,
        'addressdetails': 1
    }
    
    # Add user agent header as required by Nominatim
    headers = {
        'User-Agent': 'Weather-App/1.0'
    }
    
    try:
        # Make the request
        response = requests.get(BASE_URL, params=params, headers=headers)
        
        if response.status_code == 200:
            results = response.json()
            
            if results:
                location = results[0]
                latitude = float(location['lat'])
                longitude = float(location['lon'])
                display_name = location.get('display_name', city_name)
                
                return {
                    'city': city_name,
                    'latitude': latitude,
                    'longitude': longitude,
                    'display_name': display_name,
                    'coordinates': f"{latitude}, {longitude}"
                }
            else:
                return {'error': f"City '{city_name}' not found"}
                
        else:
            return {'error': f"API Request failed with status code: {response.status_code}"}
            
    except Exception as e:
        return {'error': f"Error occurred: {str(e)}"}

def __get_current_temperature(latitude: float, longitude: float) -> dict:
    """Fetch current temperature for given coordinates."""
    
    BASE_URL = "https://api.open-meteo.com/v1/forecast"
    
    # Parameters for the request
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'hourly': 'temperature_2m',
        'forecast_days': 1,
    }

    # Make the request
    response = requests.get(BASE_URL, params=params)
    
    if response.status_code == 200:
        results = response.json()
    else:
        raise Exception(f"API Request failed with status code: {response.status_code}")

    # print(f"API Request successful: {results}")
    current_utc_time = datetime.datetime.utcnow()
    time_list = [datetime.datetime.fromisoformat(time_str.replace('Z', '+00:00')) for time_str in results['hourly']['time']]
    temperature_list = results['hourly']['temperature_2m']
    
    closest_time_index = min(range(len(time_list)), key=lambda i: abs(time_list[i] - current_utc_time))
    current_temperature = temperature_list[closest_time_index]
    
    return f'The current temperature is {current_temperature}°C'


if __name__ == "__main__":
    result = get_weather_by_city_func("sf") 
    print(result) # The current temperature is 14.5°C

    result = search_wikipedia_func("langchain")
    print(result[:100])