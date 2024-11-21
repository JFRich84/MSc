import requests
import pandas as pd
import os
from dotenv import load_dotenv

def pull_price_data(symbols, start_date, end_date, output_file='combined_eod_data_tiingov2.csv'):
    '''
    Pulls price data from Tiingo API for the specified symbols and date range,
    and saves the resulting data as a CSV.
    
    Args:
    symbols: list of symbols to pull adjusted close for
    start_date: start date for historical data
    end_date: end date for historical data
    output_file: name of the output CSV file
    '''
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

        # Check if all data has been received
        if response.status_code == 200:
            data = response.json()
            if not data:
                return pd.DataFrame()  # Return empty DataFrame if no data available

            # Convert the collected data to a pandas DataFrame
            df = pd.DataFrame(data)
            df['symbol'] = symbol  # Add a column for the symbol
            return df
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return pd.DataFrame()

    # Initialize an empty DataFrame to store all the data
    price_df = pd.DataFrame()

    # Loop through each symbol and fetch its data
    for symbol in symbols:
        symbol_df = get_eod_data(symbol, start_date, end_date, API_KEY)
        price_df = pd.concat([price_df, symbol_df], ignore_index=True)

    # Check if the data is non-empty before proceeding
    if not price_df.empty:
        # Format the date
        price_df = price_df[['date', 'adjClose', 'symbol']]
        price_df['date'] = pd.to_datetime(price_df['date']).dt.date

        # Pivot to get symbols in columns and dates as index
        price_df = price_df.pivot(index='date', columns='symbol', values='adjClose')

        print(price_df.head())

        # Save to CSV
        price_df.to_csv(output_file, index=True)
    else:
        print("No data available")



def get_user_input():
    '''
    Asks the user for input to determine if symbols or prices should be loaded from a CSV file.
    '''
    user_response = input("Have you entered symbols into a CSV file? (Y/N): ").strip().lower()
    if user_response in ['y', 'yes']:
        symbols = pd.read_csv('symbols.csv')['symbol'].tolist()
        return symbols
    
    user_response = input("Have you provided a list of symbols and prices in a CSV format? (Y/N): ").strip().lower()
    if user_response in ['y', 'yes']:
        prices = pd.read_csv('prices.csv')
        print(prices.head())
        return prices
    
    return None

symbols = get_user_input()