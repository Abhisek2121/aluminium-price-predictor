from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from io import BytesIO
import base64

app = Flask(__name__, static_folder='static')


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    chart_url = None

    # === Load and process dataset ===
    try:
        df = pd.read_csv('data/price_changes_1995_2007.csv')

        # Try to clean price values
        if df['Price'].dtype == 'object':
            df['Price'] = df['Price'].str.replace(',', '').astype(float)

        # Convert month to datetime
        df['Date'] = pd.to_datetime(df['Month'], format='%b-%y')
        df = df.sort_values('Date')

        # === Generate line chart ===
        plt.figure(figsize=(6, 3))
        plt.plot(df['Date'], df['Price'], color='blue', linewidth=2)
        plt.title("Aluminum Price Trend (1995–2007)")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.tight_layout()

        # Convert plot to base64
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        chart_url = base64.b64encode(img.read()).decode()
        plt.close()

    except Exception as e:
        print("Error generating chart:", e)

    # === Handle prediction form ===
    if request.method == 'POST':
        try:
            year = int(request.form['year'])
            month = int(request.form['month'])

            # Convert year/month to timestamp (same feature used in model)
            date = pd.Timestamp(year=year, month=month, day=1)
            timestamp = date.timestamp()

            # Load trained model
            with open('model.pkl', 'rb') as f:
                model = pickle.load(f)

            # Predict
            predicted_price = model.predict([[timestamp]])[0]
            prediction = round(predicted_price, 2)
            
        except Exception as e:
            prediction = f"Error: {e}"

    return render_template('index.html', prediction=prediction, chart_url=chart_url)


if __name__ == '__main__':
    from waitress import serve
    serve(app, host='0.0.0.0', port=8080)

