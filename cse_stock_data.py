# This code is used to fetch the datasets we use for the ML model

import requests
import pandas as pd

# API of CSE taken from the official website
URL = "https://www.cse.lk/api/companyChartDataByStock"
STOCKS = {
    "COMB": 208, # Com bank
    "ABL": 3046, # Amana bank
    "LOFC": 1847, # LOLC finance
    "SFCL": 1789, # Softlogic finance
    "CTEA": 327, # Dilmah ceylon tea
    "WATA": 148, # Watawala plantations
    "SIRA": 142, # Sierra cables
    "DPL": 123, # Dankotuwa porcelain
    "DIPD": 399, # Dipped products 
    "TKYO": 411, # Tokyo cement
    "LMF": 518, # Lanka milk foods
    "KFP": 227, # Keells food products
    "RICH": 511, # Richard pieris and company
    "JKH": 297, # John Keells holdings
}

# period = 5 is data for the last year
def fetch(stock, stock_id, period=5):
    print("Fetching for stock: ", stock)
    data = {
        "stockId": stock_id,
        "period": period
    }
    r = requests.post(URL, data=data)
    if r.status_code == 200:
      r.raise_for_status()
      data = r.json()["chartData"]

      df = pd.DataFrame(data)
      df["date"] = pd.to_datetime(df["t"], unit="ms")
      df = df.rename(columns={
          "h": "high",
          "l": "low",
          "p": "close",
          "q": "volume"
      })

      df = df[["date", "high", "low", "close", "volume"]]
      df.sort_values("date", inplace=True)
      df.to_csv(f"data/raw/{stock}.csv", index=False)
      print(f"Saved {stock}.csv")
    else:
       print("Failed to fetch data")
       print(r.reason)

for s, i in STOCKS.items():
    fetch(s, i)