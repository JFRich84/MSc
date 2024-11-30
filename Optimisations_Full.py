
# Required Libraries 
import requests
import pandas as pd
import os
from dotenv import load_dotenv
import re


def fetch_and_save_data():
    '''
    Combines user input handling and data fetching into a single function.
    If symbols are provided in a CSV, pulls data from Tiingo API, otherwise returns prices from a provided CSV.
    '''
    user_input = input("Have you entered symbols into a CSV file? (Y/N): ").strip().lower()
    if user_input in ['y', 'yes']:
        try:
            # Load symbols, start date, and end date from CSV
            df = pd.read_csv('symbols.csv')

            # Check if required columns are present
            required_columns = ['symbol', 'start_date', 'end_date']
            df.columns = df.columns.str.lower() # ignore case
            for col in required_columns:
                if col not in df.columns:
                    raise KeyError(f"The CSV file must contain a column named '{col}'.")

            symbols = df['symbol'].dropna().tolist()
            start_date = df['start_date'].dropna().iloc[0]
            end_date = df['end_date'].dropna().iloc[0]

            # Validate that start_date and end_date are provided
            if not start_date or not end_date:
                raise ValueError("Start date and end date must be provided in the CSV file.")

            # Load the API key from the .env file
            load_dotenv()
            API_KEY = os.getenv('Tiingo')

            if not API_KEY:
                raise ValueError("API key not found in environment variables.")
            
            # API EOD URL from Tiingo
            url = 'https://api.tiingo.com/tiingo/daily/{symbol}/prices'

            # Function to fetch EOD data for a single symbol
            def get_eod_data(symbol, start_date, end_date, api_key):
                # Define parameters for the API request
                headers = {
                    'Content-Type': 'application/json',
                    'Authorization': f'Token {api_key}'
                }
                querystring = {
                    'startDate': start_date,
                    'endDate': end_date
                }

                # Make the request
                response = requests.get(url.format(symbol=symbol), headers=headers, params=querystring)

                # Check if all data has been received, error reporting
                if response.status_code == 200:
                    data = response.json()
                    if not data:
                        return pd.DataFrame()  # Return empty DataFrame if no data available

                    df = pd.DataFrame(data)
                    df['symbol'] = symbol  # Add a column for the symbol
                    return df
                else:
                    print(f"Error: {response.status_code}, {response.text}")
                    return pd.DataFrame()

            # Initialise an empty DataFrame to store all the data
            price_df = pd.DataFrame()

            # Loop through each symbol and fetch price data
            for symbol in symbols:
                symbol_df = get_eod_data(symbol, start_date, end_date, API_KEY)
                price_df = pd.concat([price_df, symbol_df], ignore_index=True)

            # Check if the data is non-empty before proceeding. 
            if not price_df.empty:
                # Format the date
                price_df = price_df[['date', 'adjClose', 'symbol']]
                price_df['date'] = pd.to_datetime(price_df['date']).dt.date

                # Pivot to get symbols in columns and dates as index
                price_df = price_df.pivot(index='date', columns='symbol', values='adjClose')

                # print(price_df.head())

                # Save to CSV
                price_df.to_csv('prices.csv', index=True)
                return 'prices.csv'
            else:
                print("No data available")
                return None
        except KeyError as e:
            raise KeyError(str(e))
        except FileNotFoundError:
            raise FileNotFoundError("The file 'symbols.csv' was not found.")
        except ValueError as e:
            print(e)
            return None
    
    user_input = input("Have you provided a list of symbols and prices in a CSV format? (Y/N): ").strip().lower()
    if user_input in ['y', 'yes']:
        try:
            prices = pd.read_csv('prices.csv')
            # print(prices.head())
            return 'prices.csv'
        except FileNotFoundError:
            raise FileNotFoundError("The file 'prices.csv' was not found. Please ensure it is in the correct directory.")
    
    return None


result = fetch_and_save_data()
if result:
    print(f"Data saved to {result}")
