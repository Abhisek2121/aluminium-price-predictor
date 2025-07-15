from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from io import BytesIO
import base64
import datetime
import random  # For mock prices

app = Flask(__name__, static_folder='static')
visitor_count = 0

# Rotating Fun Facts
fun_facts = [
    "Aluminum is the most abundant metal in Earth's crust.",
    "Recycling one aluminum can saves enough energy to run a TV for 3 hours.",
    "Aluminum does not rust like iron.",
    "The Wright brothers used aluminum to build the engine for their first airplane.",
    "Aluminum is 100% recyclable and retains its properties indefinitely.",
    "It is used in space shuttles due to its light weight and strength.",
    "Aluminum foil is less than 0.2 millimeters thick!",
    "It takes about 60 days to recycle an aluminum can into a new one.",
    "Aluminum reflects about 92% of visible light.",
    "Most beverage cans are made of aluminum due to its ease of recycling."
]

@app.route('/', methods=['GET', 'POST'])
def index():
    global visitor_count
    visitor_count += 1

    prediction_inr = None
    prediction_usd = None
    chart_url = None

    # Mock Live Data (replace with API later)
    usd_to_inr_live = round(random.uniform(82.8, 83.3), 2)
    nalco_share_price = round(random.uniform(130, 140), 2)

    try:
        df = pd.read_csv('data/price_changes_1995_2007.csv')
        if df['Price'].dtype == 'object':
            df['Price'] = df['Price'].str.replace(',', '').astype(float)
        df['Date'] = pd.to_datetime(df['Month'], format='%b-%y')
        df = df.sort_values('Date')

        plt.figure(figsize=(6, 3))
        plt.plot(df['Date'], df['Price'], color='blue', linewidth=2)
        plt.title("Aluminum Price Trend (1995–2024)")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.tight_layout()

        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        chart_url = base64.b64encode(img.read()).decode()
        plt.close()
    except Exception as e:
        print("Error generating chart:", e)

    if request.method == 'POST':
        try:
            year = int(request.form['year'])
            month = int(request.form['month'])
            date = pd.Timestamp(year=year, month=month, day=1)
            timestamp = date.timestamp()
            with open('model.pkl', 'rb') as f:
                model = pickle.load(f)
            predicted_price = model.predict([[timestamp]])[0]
            prediction_inr = round(predicted_price, 2)
            prediction_usd = round(prediction_inr / usd_to_inr_live, 2)
        except Exception as e:
            prediction_inr = f"Error: {e}"

    return render_template(
        'index.html',
        prediction_inr=prediction_inr,
        prediction_usd=prediction_usd,
        chart_url=chart_url,
        today=datetime.date.today().strftime("%d-%b-%Y"),
        visitor_count=visitor_count,
        fun_facts=fun_facts,
        usd_to_inr_live=usd_to_inr_live,
        nalco_share_price=nalco_share_price
    )

if __name__ == '__main__':
    app.run(debug=True)
