import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
model_path = os.getenv('MODEL_PATH', 'default/path/to/model.keras')
data_path = os.getenv('DATA_PATH', 'default/path/to/data.csv')

# Load the model and scalers
model = tf.keras.models.load_model(os.environ.get('MODEL_PATH', 'electricity_demand_lstm_model.keras'))
scaler_features = MinMaxScaler()
scaler_target = MinMaxScaler()

def predict_demand(start_date, end_date, interval='hourly'):
    # Load historical data
    df_path = os.environ.get('DATA_PATH', 'electricity_demand_dataset.csv')
    df = pd.read_csv(df_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Hour'] = df['Date'].dt.hour
    df['Day'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df.drop('Date', axis=1, inplace=True)

    # Scale features and target
    features = ['Hour', 'Temperature (°C)', 'Humidity (%)', 'Wind Speed (m/s)', 'Holiday', 
                 'New Housing Units', 'Commercial Area Growth (m²)', 'Population Growth (%)']
    target = 'Electricity Demand (MW)'
    scaled_features = scaler_features.fit_transform(df[features])
    scaled_target = scaler_target.fit_transform(df[[target]])

    # Create sequences for LSTM
    timesteps = 24
    X, y = [], []
    for i in range(timesteps, len(scaled_features)):
        X.append(scaled_features[i-timesteps:i])
        y.append(scaled_target[i])
    X, y = np.array(X), np.array(y)

    # Prepare future data for prediction
    future_dates = pd.date_range(start=start_date, end=end_date, freq='h')
    future_df = pd.DataFrame(index=future_dates)
    future_df['Hour'] = future_dates.hour
    future_df['Temperature (°C)'] = np.random.uniform(15, 25, len(future_dates))
    future_df['Humidity (%)'] = np.random.uniform(30, 70, len(future_dates))
    future_df['Wind Speed (m/s)'] = np.random.uniform(0, 10, len(future_dates))
    future_df['Holiday'] = np.random.choice([0, 1], len(future_dates))
    future_df['New Housing Units'] = np.random.randint(50, 200, len(future_dates))
    future_df['Commercial Area Growth (m²)'] = np.random.randint(100, 500, len(future_dates))
    future_df['Population Growth (%)'] = np.random.uniform(0, 1, len(future_dates))
    future_df['Day'] = future_dates.day
    future_df['Month'] = future_dates.month
    future_df['Year'] = future_dates.year
    future_df = future_df[features]
    scaled_future_features = scaler_features.transform(future_df)

    # Prepare sequences for LSTM
    last_sequence = scaled_features[-timesteps:]
    future_sequences = []
    for i in range(len(future_dates)):
        future_sequence = np.concatenate((last_sequence, scaled_future_features[i].reshape(1, -1)), axis=0)
        future_sequences.append(future_sequence)
        last_sequence = np.append(last_sequence[1:], scaled_future_features[i].reshape(1, -1), axis=0)
    future_sequences = np.array(future_sequences)

    # Predict future demand
    predicted_future_demand = model.predict(future_sequences)
    predicted_future_demand = scaler_target.inverse_transform(predicted_future_demand)
    predicted_future_demand = predicted_future_demand.reshape(-1)

    # Create DataFrame for predictions
    future_df['Predicted Electricity Demand (MW)'] = predicted_future_demand

    # Aggregate data based on interval
    if interval == 'weekly':
        result = future_df.resample('W').agg({
            'Temperature (°C)': 'mean',
            'Humidity (%)': 'mean',
            'Predicted Electricity Demand (MW)': 'max'
        })
        result = result.rename_axis('Week Start').reset_index()
        result['Week Start'] = result['Week Start'].dt.strftime('%Y-%m-%d')
        result['Week Number'] = result['Week Start'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').strftime('%U'))
        result['Predicted Electricity Demand (MW)'] = result['Predicted Electricity Demand (MW)'].astype(int)

        return result.to_dict(orient='records')

    elif interval == 'monthly':
        result = future_df.resample('ME').agg({
            'Temperature (°C)': 'mean',
            'Humidity (%)': 'mean',
            'Predicted Electricity Demand (MW)': 'max'
        })
        result = result.rename_axis('Month').reset_index()
        result['Month'] = result['Month'].dt.strftime('%B %Y')
        result['Predicted Electricity Demand (MW)'] = result['Predicted Electricity Demand (MW)'].astype(int)
        
        return result.to_dict(orient='records')

    else:  # Default to hourly
        result = future_df.reset_index()
        result['Predicted Electricity Demand (MW)'] = result['Predicted Electricity Demand (MW)'].astype(int)
        return result.to_dict(orient='records')

# API: Hourly Prediction
@csrf_exempt
def api_hourly(request):
    start_date = request.GET.get('start_date', '2024-10-01')
    end_date = request.GET.get('end_date', '2024-11-30')

    try:
        datetime.fromisoformat(start_date)
        datetime.fromisoformat(end_date)
    except ValueError:
        return JsonResponse({'error': 'Invalid date format. Use YYYY-MM-DD.'}, status=400)

    prediction_data = predict_demand(start_date, end_date, interval='hourly')
    filtered_data = [
        {
            'Date': item['index'].strftime('%Y-%m-%d %H:%M:%S'),
            'Hour': item['Hour'],
            'Temperature (°C)': item['Temperature (°C)'],
            'Humidity (%)': item['Humidity (%)'],
            'Predicted Electricity Demand (MW)': int(item['Predicted Electricity Demand (MW)'])
        }
        for item in prediction_data
    ]
    return JsonResponse(filtered_data, safe=False)

# API: Weekly Prediction
@csrf_exempt
def api_weekly(request):
    start_date = request.GET.get('start_date', '2024-10-01')
    end_date = request.GET.get('end_date', '2024-11-30')

    try:
        datetime.fromisoformat(start_date)
        datetime.fromisoformat(end_date)
    except ValueError:
        return JsonResponse({'error': 'Invalid date format. Use YYYY-MM-DD.'}, status=400)

    prediction_data = predict_demand(start_date, end_date, interval='weekly')
    filtered_data = [
        {
            'Week Start': item['Week Start'],
            'Week Number': int(item['Week Number']),
            'Temperature (°C)': item['Temperature (°C)'],
            'Humidity (%)': item['Humidity (%)'],
            'Predicted Electricity Demand (MW)': int(item['Predicted Electricity Demand (MW)'])
        }
        for item in prediction_data
    ]
    return JsonResponse(filtered_data, safe=False)

# API: Monthly Prediction
@csrf_exempt
def api_monthly(request):
    start_date = request.GET.get('start_date', '2024-10-01')
    end_date = request.GET.get('end_date', '2024-11-30')

    try:
        datetime.fromisoformat(start_date)
        datetime.fromisoformat(end_date)
    except ValueError:
        return JsonResponse({'error': 'Invalid date format. Use YYYY-MM-DD.'}, status=400)

    prediction_data = predict_demand(start_date, end_date, interval='monthly')
    filtered_data = [
        {
            'Month': item['Month'],
            'Temperature (°C)': item['Temperature (°C)'],
            'Humidity (%)': item['Humidity (%)'],
            'Predicted Electricity Demand (MW)': int(item['Predicted Electricity Demand (MW)'])
        }
        for item in prediction_data
    ]
    return JsonResponse(filtered_data, safe=False)

# API: Maximum Predicted Peak Load
@csrf_exempt
def api_max_peak_load(request):
    start_date = request.GET.get('start_date', '2024-10-01')
    end_date = request.GET.get('end_date', '2024-11-30')

    try:
        start_date_obj = datetime.fromisoformat(start_date)
        end_date_obj = datetime.fromisoformat(end_date)
    except ValueError:
        return JsonResponse({'error': 'Invalid date format. Use YYYY-MM-DD.'}, status=400)

    prediction_data = predict_demand(start_date, end_date, interval='hourly')

    predicted_demand_values = [item['Predicted Electricity Demand (MW)'] for item in prediction_data]
    
    if predicted_demand_values:
        max_predicted_demand = max(predicted_demand_values)
        max_demand_item = next(item for item in prediction_data if item['Predicted Electricity Demand (MW)'] == max_predicted_demand)
        
        result = {
            'Date': max_demand_item['Date'],
            'Hour': max_demand_item['Hour'],
            'Max Predicted Electricity Demand (MW)': max_predicted_demand
        }
        return JsonResponse(result, safe=False)

    return JsonResponse({'error': 'No demand data available.'}, status=404)
