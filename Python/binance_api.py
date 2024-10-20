import requests

def get_lot_size(symbol: str) -> dict | None:
    """
    Attempts to fetch the LOT_SIZE filter info for 'symbol' from Binance.us API.

    Args:
        symbol (str): The cryptocurrency symbol, e.g., BTCUSDT

    Returns:
        dict | None: The LOT_SIZE filter dict, or None if not found
    """

    url = "https://api.binance.us/api/v3/exchangeInfo"

    try:
        response = requests.get(url).json()
        symbols = response["symbols"]

        for sym_info in symbols:
            if sym_info["symbol"] == symbol:
                for filter_ in sym_info["filters"]:
                    if filter_["filterType"] == "LOT_SIZE":
                        return filter_

        return None  # Symbol not found or has no LOT_SIZE

    except (requests.exceptions.RequestException, ValueError) as e:
        print(f"Error fetching data: {e}")
        return None

def adjust_iceberg_size(lot_size: dict, iceberg_size: float) -> float:
    """
    Adjusts the size of an iceberg order to a valid step size based on LOT_SIZE.

    Args:
        lot_size (dict): The LOT_SIZE information dict (from get_lot_size())
        iceberg_size (float): The desired iceberg order size

    Returns:
        float: The adjusted iceberg size conforming to step size
    """

    step_size = float(lot_size["stepSize"])
    return round(iceberg_size / step_size) * step_size

def get_quote(symbol: str) -> dict | None:
    """
    Fetches the latest quote (price) for a symbol from Binance.us.

    Args:
        symbol (str): The cryptocurrency symbol, e.g., BTCUSDT

    Returns:
        dict | None: The quote info (including price), or None on error
    """

    url = f"https://api.binance.us/api/v3/ticker/price?symbol={symbol}"

    try:
        response = requests.get(url).json()
        # Basic validation, Binance's responses can be unreliable on error
        if "symbol" in response and response["symbol"] == symbol:
            return response
        else:
            return None

    except (requests.exceptions.RequestException, ValueError) as e:
        print(f"Error fetching quote: {e}")
        return None
    
if __name__ == "__main__":
    size_info = get_lot_size("BTCUSDT")
    if size_info:
        print(f"LOT_SIZE for BTCUSDT: {size_info}")
    else:
        print("Failed to get LOT_SIZE")
        
    symbol = "BTCUSDT"
    desired_iceberg_size = 0.12345

    lot_size_info = get_lot_size(symbol)
    if lot_size_info:
        adjusted_size = adjust_iceberg_size(lot_size_info, desired_iceberg_size)
        print(f"Adjusted iceberg size for {symbol}: {adjusted_size}")
    else:
        print(f"Failed to get LOT_SIZE for {symbol}")
        
    quote = get_quote(symbol)
    print(f"Quote for {symbol}: {quote}")