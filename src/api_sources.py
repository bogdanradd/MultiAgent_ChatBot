from datetime import datetime, timedelta
from typing import Optional
from langchain_core.documents import Document


def fetch_stock_data(symbol: str, days: int = 30) -> list[Document]:
    """
    Fetch stock price data using yfinance library.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
        days: Number of days of historical data to fetch

    Returns:
        List of Documents, one per day with OHLCV data
    """
    import yfinance as yf

    try:
        ticker = yf.Ticker(symbol)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        df = ticker.history(start=start_date, end=end_date)

        if df.empty:
            raise ValueError(f"No stock data found for {symbol}")

        docs = []
        for i, (date, row) in enumerate(df.iterrows(), 1):
            date_str = date.strftime('%Y-%m-%d')
            content = (
                f"Stock: {symbol} | "
                f"Date: {date_str} | "
                f"Open: ${row['Open']:.2f} | "
                f"High: ${row['High']:.2f} | "
                f"Low: ${row['Low']:.2f} | "
                f"Close: ${row['Close']:.2f} | "
                f"Volume: {int(row['Volume'])}"
            )

            docs.append(Document(
                page_content=content,
                metadata={
                    "source": f"Yahoo Finance ({symbol})",
                    "symbol": symbol,
                    "date": date_str,
                    "doc_id": f"stock_{symbol.lower()}",
                    "row": i
                }
            ))

        return docs

    except Exception as e:
        raise ValueError(f"Failed to fetch stock data for {symbol}: {e}")


def fetch_company_info(symbol: str) -> Optional[Document]:
    """
    Fetch company profile information using yfinance library.

    Args:
        symbol: Stock ticker symbol

    Returns:
        Document with company profile information
    """
    import yfinance as yf

    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        if not info or 'shortName' not in info:
            return None

        content = (
            f"Company: {info.get('longBusinessSummary', 'N/A')} | "
            f"Name: {info.get('shortName', 'N/A')} | "
            f"Sector: {info.get('sector', 'N/A')} | "
            f"Industry: {info.get('industry', 'N/A')} | "
            f"Website: {info.get('website', 'N/A')} | "
            f"Employees: {info.get('fullTimeEmployees', 'N/A')} | "
            f"City: {info.get('city', 'N/A')} | "
            f"State: {info.get('state', 'N/A')} | "
            f"Country: {info.get('country', 'N/A')} | "
            f"Market Cap: {info.get('marketCap', 'N/A')} | "
            f"Currency: {info.get('currency', 'N/A')}"
        )

        return Document(
            page_content=content,
            metadata={
                "source": f"Yahoo Finance ({symbol} Profile)",
                "symbol": symbol,
                "doc_id": f"profile_{symbol.lower()}",
                "type": "company_profile"
            }
        )

    except Exception as e:
        raise ValueError(f"Failed to fetch company info for {symbol}: {e}")


def ingest_api_data(data_type: str, identifier: str, **kwargs) -> str:
    """
    Unified API data ingestion function.

    Args:
        data_type: Type of data ('stock', 'company_info')
        identifier: Stock symbol
        **kwargs: Additional parameters (e.g., days for stock data)

    Returns:
        doc_id of ingested data
    """
    from src.vectorstore import add_documents

    docs = []
    doc_id = f"api_{data_type}_{identifier.lower().replace(' ', '_')}"

    if data_type == 'stock':
        days = kwargs.get('days', 30)
        docs = fetch_stock_data(identifier, days)
    elif data_type == 'company_info':
        doc = fetch_company_info(identifier)
        if doc:
            docs = [doc]
    else:
        raise ValueError(f"Unknown data type: {data_type}")

    if not docs:
        raise ValueError(f"No data retrieved for {data_type}: {identifier}")

    # Ensure all docs have consistent doc_id matching what we return
    for doc in docs:
        doc.metadata["doc_id"] = doc_id

    add_documents(docs, doc_id)
    return doc_id
