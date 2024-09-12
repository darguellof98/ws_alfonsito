from dotenv import load_dotenv

from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.prompts import ChatPromptTemplate
## Defino las funciones que vamos a usar
import yfinance as yf 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 


## Voy a hacer cadenas condicionales para que si la pregunta no tiene que ver se corte  



# Load environment variables from .env
load_dotenv()


# Obtengo los datos
@tool
def prices_download(
        ticker:str,
        start:str,
        end:str,
        user: str= 'userX') -> str:
    """
    Esta función descarga los precios de Yahoo Finance de un/unos tickers en un determinado rango de fechas y los guarda en una ruta especifica
    La estructura del archivo descargado es la siguiente:
    Date: Fecha
    Ticker: Ticker de la acción
    Open: Precio de apertura
    High: Precio máximo
    Low: Precio mínimo
    Close: Precio de cierre
    Adj_Close: Precio de cierre ajustado
    Volume: Volumen de transacciones

    Args:
        ticker: Tiene que ser una lista con el/los tickers que se quiera descargar
        start: Fecha de inicio en formato string "YYYY-MM-DD"
        end: Fecha de fin en formato string "YYYY-MM-DD" 
    Returns:
        str: Ruta donde se guardó el archivo

    ejemplo:
    prices_download(ticker = "AAPL", start = "2020-01-01")
    >>> '6_langchain_app/datasets/userX_AAPL.csv'
    """
    data = (yf.download(ticker, start=start, end=end, progress=False)
            .reset_index()
            .assign(Ticker = ticker)
            .loc[:, ["Date", "Ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
            .rename(columns = {"Adj Close": "Adj_Close"})
            
            )
    
    dir_folder = f'6_langchain_app/datasets/{user}_{ticker}.csv'
    data.to_csv(dir_folder, index = False)
    return dir_folder

# Alfonsito Analista
@tool
def sma(
        data_dir:str, 
        windows:list) -> str:
    """
    Esta función calcula la media movil simple de una serie de precios en una/unas ventanas de tiempo
    El archivo de entrada debe tener una columna llamada "Adj_Close" con los precios de cierre ajustados
    La funcion lo que hace es crear nuevas columnas segun las ventanas de tiempo especificadas
    Los nombres de las columnas serán "SMA_{w}" donde w es la ventana de tiempo

    Args:
        data_dir: es la ruta donde esta el archivo con las cotizaciones de alguna accion
        windows: list: Lista con las ventanas de tiempo que se quieran calcular (debe ser una lista donde sus elementos sean enteros)
    Returns:
        str: Ruta donde se guardó el archivo con las medias moviles simples
    
    Ejemplo:
    sma(data_dir = '6_langchain_app/datasets/userX_AAPL.csv', windows = [10, 20, 50])
    >>> '6_langchain_app/datasets/userX_AAPL_sma.csv'
    """
    data = pd.read_csv(data_dir)

    for w in windows:
        data[f"SMA_{w}"] = data["Adj_Close"].rolling(window = w).mean()

    data_dir_sma = data_dir.replace(".csv", "_sma.csv")
    data.to_csv(data_dir_sma, index = False)

    return data_dir_sma

@tool
def returns(
        data_dir:str,
        cum:bool = False,
        log:bool = False) -> str:
    """
    Esta función calcula los retornos de una serie de precios, puede calcular los retornos simples, logaritmicos y acumulados
    El archivo de entrada debe tener una columna llamada "Adj_Close" con los precios de cierre ajustados
    La funcion lo que hace es crear nuevas columnas segun lo especificado en los argumentos
    Los nombres de las columnas pueden ser: "Return", "Cum_Return", "Log_Return", "Cum_Log_Return"
    Args:
        data: str: Ruta donde esta el archivo con las cotizaciones de alguna accion
        cum: bool: Si se quiere calcular los retornos acumulados
        log: bool: Si se quiere calcular los retornos logaritmicos
    
    Returns:
        str: La ruta donde se guardó el archivo con los retornos
    
    Ejemplo:
    returns(data_dir = '6_langchain_app/datasets/userX_AAPL.csv', log = True)
    >>> '6_langchain_app/datasets/userX_AAPL_returns.csv'

    """
    data = pd.read_csv(data_dir)
    data["Return"] = data["Adj_Close"].pct_change()
    data["Cum_Return"] = (1 + data["Return"]).cumprod()
    data['Log_Return'] = np.log(data['Adj_Close'] / data['Adj_Close'].shift(1))
    data["Cum_Log_Return"] = data["Log_Return"].cumsum()
    columns = ["Return", "Cum_Return", "Log_Return", "Cum_Log_Return"]
    if log==False:
        data = data.drop(columns = ["Log_Return", "Cum_Log_Return"])

    data_dir_returns = data_dir.replace(".csv", "_returns.csv")
    data.to_csv(data_dir_returns, index = False)
    
    return data_dir_returns
    



# Alfonsito Graficador
@tool
def line_graph_prices(
        data_dir: str,
        x: str,
        y: list,
        filename: str = "graph.png",
        user: str = 'userX',
        title = None) -> str:
    """
    Grafica un diagrama de lineas de una o varias series de precios y lo guarda en un directorio

    Args:
        data_dir: str: Ruta donde esta el archivo con las cotizaciones de alguna accion
        x: str: Nombre de la columna que se quiere en el eje x
        y: list: Lista con los nombres de las columnas que se quieren en el eje y
        save_dir: str: Directorio donde se quiere guardar la imagen
        filename: str: Nombre del archivo
        user: str: Nombre del usuario
        title: str: Titulo del grafico

    Ejemplo:
    line_graph_prices(data = '6_langchain_app/datasets/userX_AAPL.csv', x = "Date", y = ["AAPL", "MSFT"], save_dir = "graphs", filename = "graph.png", user = "userX", title = "Precios de AAPL y MSFT")
    >>> '6_langchain_app/graphs/userX-graph.png'
    """
    data = pd.read_csv(data_dir)
    plt.figure(figsize=(12, 6))
    for i in y:
        plt.plot(data[x], data[i], label=i)
    
    plt.legend()
    if title is not None:
        plt.title(title)
    
    plt.xlabel(x)
    # Adjust layout to remove unnecessary margins
    plt.tight_layout()
    full_path = ''
    save_dir = '/Users/mini/Library/Mobile Documents/com~apple~CloudDocs/PythonProjects/LLM-Agents-RAG/6_langchain_app/graphs'
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    # Define the full path
    full_path = os.path.join(save_dir, f'{user}-{filename}')
    # Save the plot
    plt.savefig(full_path, format='png', bbox_inches='tight')
    print(f"Graph saved at: {full_path}")


    # Close the plot to free memory
    plt.close()
    return str(full_path) # type: ignore

@tool
def hist_plot(
        data_dir: str,
        x: str,
        filename: str = "hist.png",
        user: str = 'userX',
        title = None,
        normal = None) -> str:
    """
    Grafica un histograma de una serie de precios y lo guarda en un directorio

    Args:
        data_dir: str: Ruta donde esta el archivo con la informacion necesaria
        x: str: Nombre de la columna que se quiere graficar
        save_dir: str: Directorio donde se quiere guardar la imagen
        filename: str: Nombre del archivo
        user: str: Nombre del usuario
        title: str: Titulo del grafico
        normal: bool: Si se quiere graficar la distribución normal

    Ejemplo:
    hist_plot(data_dir = '6_langchain_app/datasets/userX_TSLA_returns.csv', x = "Returns", save_dir = "graphs", filename = "hist.png", user = "userX", title = "Histograma de precios de AAPL", normal = True
    >>> '6_langchain_app/graphs/userX-hist.png'
    """
    data = pd.read_csv(data_dir)
    serie_plot = data[x]
    plt.figure(figsize=(12, 6))
    sns.histplot(serie_plot, kde=True, stat='density', color='blue', label='Data', bins=30) # type: ignore
    if normal is not None:
        mu = np.mean(serie_plot)
        sigma = np.std(serie_plot)
        x_normal= np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
        y_normal = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(- (x_normal - mu)**2 / (2 * sigma**2))
        plt.plot(x_normal, y_normal, color='red', label='Normal Distribution')
    plt.legend()
    if title is not None:
        plt.title(title)
    full_path = ''
    save_dir = '/Users/mini/Library/Mobile Documents/com~apple~CloudDocs/PythonProjects/LLM-Agents-RAG/6_langchain_app/graphs'
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    # Define the full path
    full_path = os.path.join(save_dir, f'{user}-{filename}')
    # Save the plot
    plt.savefig(full_path, format='png', bbox_inches='tight')
    print(f"Graph saved at: {full_path}")

    # Close the plot to free memory
    plt.close()
    return str(full_path) # type: ignore






llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

tools = [prices_download, sma, returns, line_graph_prices, hist_plot]
prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente muy util, y capaz de usar multiples funciones para solucionar los requerimientos del usuario"), 
    ("human", "{input}"), 
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt) # type: ignore
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True) # type: ignore

prompt = """
Hola, soy David, me gustaría saber como se han comportado los precios de Tesla en los últimos años.
1. Descarga los precios de TSLA desde  el 2015, 
2. Calcula los retornos simples 
3. Grafica un histograma incluyendo como deberian ser si fueran normales"
"""

prompt = """
Hola, soy David, me gustaría saber como se han comportado los precios de TSLA.
1. Descarga los precios desde el 2015 hasta hoy, 
3. Haz un grafico de lineas e incluye las medias moviles simples de 10, 20 y 50 periodos"
"""

agent_executor.invoke({"input": prompt, })


OPENAI_API_KEY = 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
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="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"
LANGCHAIN_PROJECT="pr-large-account-85"
