from flask import Flask, jsonify, Response, send_file
import requests
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
import io
from datetime import timedelta, datetime
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import openpyxl

app = Flask(__name__)


class SolarDataFetcher:
    def __init__(self):
        self.sunspot_number = "Loading..."
        self.obsdate = "Loading..."

    def fetch_solar_data(self):
        url = 'https://services.swpc.noaa.gov/json/solar-cycle/swpc_observed_ssn.json'
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            last_entry = data[-1]  # Most recent data
            self.sunspot_number = str(last_entry['swpc_ssn'])
            self.obsdate = last_entry['Obsdate']
        else:
            print(f"Failed to retrieve data: {response.status_code}")
            self.sunspot_number = "Error"
            self.obsdate = "Error"


class GeomagneticPredictor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.xgb = XGBClassifier(
            n_estimators=180,
            max_depth=1000,
            learning_rate=10000,
            random_state=42,
            use_label_encoder=False
        )

        # Load datasets
        self.SC19_24 = pd.read_excel('Daily Solar Cycle 19-24 Classifier.xlsx')
        self.SC25 = pd.read_excel('Daily Solar Cycle 25 Classifier.xlsx')

        # Prepare training data
        self.X_train = self.SC19_24[['Daily Ap', 'SN']].dropna()
        self.y_train = self.SC19_24['Latitudes Day'].loc[self.X_train.index]

        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)

        # Map labels to integers
        self.unique_classes = sorted(set(self.y_train))
        self.class_mapping = {label: idx for idx, label in enumerate(self.unique_classes)}
        self.inverse_mapping = {idx: label for label, idx in self.class_mapping.items()}
        self.y_train_mapped = self.y_train.map(self.class_mapping)

        # Train the model
        self.xgb.fit(self.X_train_scaled, self.y_train_mapped)

    def get_latest_ap_index(self):
        url = 'https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json'
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            last_entry = data[-1]
            if isinstance(last_entry, list):
                return last_entry[-2]
        print("Failed to retrieve Ap index or unexpected data structure.")
        return None

    def predict_latitude(self):
        latest_ap_index = self.get_latest_ap_index()
        if latest_ap_index is None:
            return {"latitude": "Unavailable", "accuracy": "Unavailable"}

        latest_feature = self.scaler.transform([[latest_ap_index, 0]])
        predicted_class = self.xgb.predict(latest_feature)
        predicted_latitude = self.inverse_mapping[predicted_class[0]]

        X_test = self.SC25[['Daily Ap', 'SN']].dropna()
        y_test = self.SC25['Latitudes Day'].loc[X_test.index]
        X_test_scaled = self.scaler.transform(X_test)
        y_test_mapped = y_test.map(self.class_mapping)

        y_pred_mapped = self.xgb.predict(X_test_scaled)
        accuracy = accuracy_score(y_test_mapped, y_pred_mapped)

        return {
            "latitude": predicted_latitude,
            "accuracy": f"{int(accuracy * 100)}%"
        }

class ApRetriever:
    @staticmethod
    def fetch_ap_index():
        """Fetch the latest Ap Index from NOAA API."""
        url = 'https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json'
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                last_entry = data[-1]  # Get the last entry
                if isinstance(last_entry, list):
                    return str(last_entry[-2])  # Assuming Ap Index is the second-to-last element
                else:
                    return "Error: Unexpected Data"
            else:
                return f"Error: {response.status_code}"
        except Exception as e:
            return f"Error: {e}"

class GraphGenerator:
    def __init__(self):
        # Initialize any necessary variables if needed
        pass

    def create_graph(self, forecast_dates, arima_forecast, recent_ap_values, recent_dates_formatted):
        # Create the matplotlib figure
        fig, ax = plt.subplots()

        # Plot data
        ax.plot(recent_dates_formatted, recent_ap_values, label="Observed", marker='o')
        ax.plot(forecast_dates, arima_forecast, label="Forecasted", marker='^', color='orange')

        # Customize the plot
        ax.set_xlabel("Date")
        ax.set_ylabel("Ap Index")
        ax.legend()
        ax.grid()

        # Set x-ticks and rotate labels
        all_dates = recent_dates_formatted + forecast_dates
        ax.set_xticks(range(0, len(all_dates), 3))
        ax.set_xticklabels(all_dates[::3], rotation=20)
        ax.tick_params(axis='y', labelrotation=90)

        # Adjust layout to fix padding
        fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.35)  # Adjust margins as needed

        return fig

class GraphGenerator24:
    def __init__(self):
        # Initialize any necessary variables if needed
        pass

    def create_graph(self, Forecast_Dates, arima_FORECAST, Recent_VALUES, Observed_Dates):
        # Create the matplotlib figure
        fig, ax = plt.subplots()

        ax.plot(Observed_Dates, Recent_VALUES, label="Observed", marker='o')
        ax.plot(Forecast_Dates, arima_FORECAST, label="Forecast", marker='^')

        # Customize the plot
        ax.set_xlabel("MM-DD T")
        ax.set_ylabel("Ap Index")
        ax.legend()
        ax.grid()

        ax.tick_params(axis='x', labelrotation=20)
        ax.tick_params(axis='y', labelrotation=90)

        # Adjust layout to fix padding
        fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.35)  # Adjust margins as needed

        # Set figure background to transparent
        fig.patch.set_facecolor('none')  # Transparent figure background
        ax.set_facecolor('none')  # Transparent axes background

        return fig

class SunspotDataPredict:
    def __init__(self, file_path):
        self.file_path = file_path

    def get_current_sunspot_value(self):
        """
        Fetch the sunspot value for the current year and month from an Excel file.

        Returns:
            tuple: (sunspot_value, date_string) or (None, None) if no match is found.
        """
        try:
            # Load the Excel file
            wb = openpyxl.load_workbook(self.file_path)
            sheet = wb.active

            # Get the current year and month
            current_year = datetime.now().year
            current_month = datetime.now().month

            # Find the value for the current year and month
            for row in sheet.iter_rows(min_row=1, values_only=True):  # Adjust if headers exist
                year, month, value = row[0], row[1], row[3]  # Adjust indices if needed
                if year == current_year and month == current_month:
                    date_string = f"{int(year)}-{int(month):02d}"  # Format as YYYY-MM
                    return value, date_string
            return None, None
        except Exception as e:
            print(f"Error: {e}")
            return None, None

@app.route('/SunspotPredict', methods=['GET'])
def current_sunspot():
    """
    Flask route to fetch the current sunspot value and return it as JSON.
    """
    sunspot_data = SunspotDataPredict("Predicted Sunspot Number.xlsx")  # Adjust path if needed
    value, date_string = sunspot_data.get_current_sunspot_value()

    if value is not None and date_string is not None:
        return jsonify({
            "status": "success",
            "data": {
                "sunspot_value": value,
                "date": date_string
            }
        })
    else:
        return jsonify({
            "status": "error",
            "message": "No matching data found or an error occurred."
        }), 404


@app.route('/SevenDay', methods=['GET'])
def SevenDay():
    try:
        # Fetch data
        url = "https://kp.gfz-potsdam.de/app/files/Kp_ap_Ap_SN_F107_nowcast.txt"
        response = requests.get(url)
        response.raise_for_status()

        # Parse data
        lines = response.text.splitlines()
        data_start_index = next(i for i, line in enumerate(lines) if not line.startswith('#'))
        data = pd.read_csv(io.StringIO('\n'.join(lines[data_start_index:])), sep='\s+', header=None)

        # Extract relevant columns
        dates = pd.to_datetime(data[[0, 1, 2]].rename(columns={0: 'Year', 1: 'Month', 2: 'Day'}))
        ap_data = data.iloc[:, 23]

        # Process recent data
        recent_ap_values = ap_data.iloc[-8:-1].values
        arima_model = ARIMA(recent_ap_values, order=(2, 1, 2))
        arima_fitted = arima_model.fit()
        arima_forecast = np.clip(np.round(arima_fitted.forecast(steps=7)).astype(int), 0, None)

        # Generate forecast dates
        last_date = pd.Timestamp(dates.iloc[-2])
        forecast_dates = [last_date + timedelta(days=i) for i in range(1, 8)]
        forecast_dates_formatted = [date.strftime('%Y-%m-%d') for date in forecast_dates]

        # Send data as JSON
        return jsonify({
            "forecast_dates": forecast_dates_formatted,
            "7Day__forecast": arima_forecast.tolist(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/SevenDayGraph', methods=['GET'])
def SevenDayGraph():
    try:
        # Fetch data
        url = "https://kp.gfz-potsdam.de/app/files/Kp_ap_Ap_SN_F107_nowcast.txt"
        response = requests.get(url)
        response.raise_for_status()

        # Parse data
        lines = response.text.splitlines()
        data_start_index = next(i for i, line in enumerate(lines) if not line.startswith('#'))
        data = pd.read_csv(io.StringIO('\n'.join(lines[data_start_index:])), sep='\s+', header=None)

        # Extract relevant columns
        dates = pd.to_datetime(data[[0, 1, 2]].rename(columns={0: 'Year', 1: 'Month', 2: 'Day'}))
        ap_data = data.iloc[:, 23]

        # Process recent data
        recent_ap_values = ap_data.iloc[-8:-1].values
        recent_dates = dates.iloc[-8:-1]
        recent_dates_formatted = recent_dates.dt.strftime('%Y-%m-%d').tolist()

        arima_model = ARIMA(recent_ap_values, order=(2, 1, 2))
        arima_fitted = arima_model.fit()
        arima_forecast = np.clip(np.round(arima_fitted.forecast(steps=7)).astype(int), 0, None)

        # Generate forecast dates
        last_date = pd.Timestamp(recent_dates.iloc[-1])
        forecast_dates = [last_date + timedelta(days=i) for i in range(1, 8)]
        forecast_dates_formatted = [date.strftime('%Y-%m-%d') for date in forecast_dates]

        # Generate graph using GraphGenerator class
        graph_generator = GraphGenerator()
        fig = graph_generator.create_graph(forecast_dates_formatted, arima_forecast, recent_ap_values, recent_dates_formatted)

        # Convert graph to PNG and return as a response
        output = io.BytesIO()
        fig.savefig(output, format='png')
        output.seek(0)
        plt.close(fig)

        return Response(output.getvalue(), mimetype='image/png')

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/SunspotN', methods=['GET'])
def SunspotN():
    fetcher = SolarDataFetcher()
    fetcher.fetch_solar_data()
    return jsonify({
        "sunspot_number": fetcher.sunspot_number,
        "observation_date": fetcher.obsdate
    })


@app.route('/PredictGeomagneticLatitude', methods=['GET'])
def PredictGeomagneticLatitude():
    predictor = GeomagneticPredictor()
    result = predictor.predict_latitude()
    return jsonify(result)


@app.route('/ApRetrieve', methods=['GET'])
def ApRetrieve():
    Fetcher = ApRetriever()
    latest_ap_index = Fetcher.fetch_ap_index()
    return jsonify({"latest_ap_index": latest_ap_index})


if __name__ == '__main__':
    app.run(debug=True)
