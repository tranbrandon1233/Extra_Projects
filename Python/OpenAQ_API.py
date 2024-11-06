import requests

def get_real_time_aqi(city, country):
    base_url = "https://api.openaq.org/v1/latest"
    params = {
        'city': city,
        'country': country
    }

    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data['results']:
            return data['results'][0]['measurements']
        else:
            print("No data available")
    else:
        print(f"Error fetching data: {response.status_code}")

    return []


def compare_aqi(real_time_data, past_data):
    # Example very simple comparison
    if not real_time_data or not past_data:
        return None

    current_values = {measurement['parameter']: measurement['value'] for measurement in real_time_data}
    trends = {}

    for record in past_data:
        parameter = record.get('parameter')
        if parameter in current_values:
            trend = current_values[parameter] - record['average']
            trends[parameter] = trend

    return trends

def notify_user_if_needed(trends, threshold=50):
    alerts = []

    for parameter, change in trends.items():
        if change >= threshold:
            alerts.append(f"Alert! High increase in {parameter}: {change}")

    if alerts:
        for alert in alerts:
            print(alert)  

def main():
    city = "Los Angeles"
    country = "US"
    
    real_time_data = get_real_time_aqi(city, country)

    # trends = compare_aqi(real_time_data, past_data)
    # if trends:
    #     notify_user_if_needed(trends)

if __name__ == "__main__":
    main()