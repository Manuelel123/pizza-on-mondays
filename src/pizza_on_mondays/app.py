import os
import tempfile

import pandas as pd
import quantstats as qs
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf

SECTORS = {
    "Oil & Gas": {
        "stocks": ["BP", "CVX", "EC", "SHEL", "SU", "TTE", "XOM"],
        "start": "2020-01-01",
        "end": "2026-12-31",
    },
    "Real Estate": {
        "stocks": [
            "ADC", "AKR", "BRX", "EPRT", "FCPT", "KIM", "KRG", "MAC", "NNN",
            "O", "PECO", "REG", "UE",
        ],
        "start": "2026-01-01",
        "end": "2026-12-31",
    },
    "Criptomonedas": {
        "stocks": ["BTC-USD", "DOGE-USD", "ZEC-USD"],
        "start": "2020-01-01",
        "end": "2026-12-31",
    },
}

SECTOR_ICONS = {
    "Oil & Gas": "🛢️",
    "Real Estate": "🏢",
    "Criptomonedas": "🪙",
}

TITLE_COLOR = "#87CEFA"  # azul claro

# ETFs líquidos: se pueden comprar directamente, a diferencia de un índice puro (ej. ^GSPC).
BENCHMARK_TICKERS = {
    "S&P 500 (SPY)": "SPY",
    "Nasdaq 100 (QQQ)": "QQQ",
    "Dow Jones (DIA)": "DIA",
    "Small Caps (IWM)": "IWM",
    "Oro (GLD)": "GLD",
}


def colored_title(text: str) -> None:
    st.markdown(f"<h1 style='color:{TITLE_COLOR}'>{text}</h1>", unsafe_allow_html=True)


def colored_subheader(text: str) -> None:
    st.markdown(f"<h3 style='color:{TITLE_COLOR}'>{text}</h3>", unsafe_allow_html=True)


@st.cache_data
def load_returns(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    data = yf.download(tickers, start=start, end=end)["Close"]
    data = data.ffill()  # NO miedo
    data = data.bfill()  # miedo te crea un lookahead bias
    return data.pct_change().dropna()


def all_sector_tickers() -> list[str]:
    return sorted({ticker for sector in SECTORS.values() for ticker in sector["stocks"]})


@st.cache_data
def load_price_returns(ticker: str, start: str, end: str) -> pd.Series:
    data = yf.download(ticker, start=start, end=end)["Close"]
    if isinstance(data, pd.DataFrame):
        data = data.iloc[:, 0]
    data = data.ffill().bfill()
    return data.pct_change().dropna().rename(ticker)


@st.cache_data(show_spinner=False)
def build_quantstats_report(
    returns: pd.Series, benchmark: pd.Series, asset: str, benchmark_ticker: str
) -> str:
    """Genera el tearsheet HTML de quantstats (gráficos embebidos como SVG)."""
    fd, output_path = tempfile.mkstemp(suffix=".html")
    os.close(fd)
    try:
        qs.reports.html(
            returns,
            benchmark=benchmark,
            output=output_path,
            title=f"{asset} vs {benchmark_ticker}",
            download_filename=f"quantstats_{asset}_vs_{benchmark_ticker}.html",
        )
        with open(output_path, encoding="utf-8") as f:
            return f.read()
    finally:
        os.remove(output_path)


def build_summary(returns: pd.DataFrame, momentum_window: int = 21) -> pd.DataFrame:
    """Resume, por activo, los datos más relevantes del período seleccionado."""
    cumulative = (1 + returns).cumprod()
    n_days = len(returns)

    total_return = cumulative.iloc[-1] - 1
    annualized_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0
    annualized_vol = returns.std() * (252 ** 0.5)
    sharpe = annualized_return / annualized_vol.replace(0, pd.NA)

    running_max = cumulative.cummax()
    max_drawdown = (cumulative / running_max - 1).min()

    momentum = (
        (1 + returns.tail(momentum_window)).prod() - 1
        if n_days >= 1
        else pd.Series(0, index=returns.columns)
    )

    summary = pd.DataFrame(
        {
            "Retorno total": total_return,
            "Retorno anualizado": annualized_return,
            "Volatilidad anualizada": annualized_vol,
            "Sharpe (aprox., rf=0)": sharpe,
            "Máx. drawdown": max_drawdown,
            f"Momentum últimos {momentum_window}d": momentum,
        }
    )
    return summary.sort_values("Sharpe (aprox., rf=0)", ascending=False)


def build_recommendation(summary: pd.DataFrame, momentum_window: int = 21) -> str:
    """Genera una nota descriptiva a tener en cuenta, en base a métricas históricas.

    No es asesoramiento financiero: es una lectura rápida de los datos del período
    para usar como punto de partida antes de decidir un movimiento.
    """
    if summary.empty:
        return "No hay datos suficientes en el período seleccionado para generar una recomendación."

    momentum_col = f"Momentum últimos {momentum_window}d"

    best_sharpe = summary["Sharpe (aprox., rf=0)"].idxmax()
    best_momentum = summary[momentum_col].idxmax()
    worst_momentum = summary[momentum_col].idxmin()
    most_volatile = summary["Volatilidad anualizada"].idxmax()
    worst_drawdown = summary["Máx. drawdown"].idxmin()

    lines = [
        f"- **{best_sharpe}** muestra la mejor relación retorno/riesgo del período "
        f"(Sharpe aprox. {summary.loc[best_sharpe, 'Sharpe (aprox., rf=0)']:.2f}), "
        "un candidato a mirar si se busca eficiencia riesgo-retorno.",
        f"- **{best_momentum}** tiene el mejor impulso reciente "
        f"({summary.loc[best_momentum, momentum_col]:+.1%} en los últimos {momentum_window} días hábiles), "
        "lo que podría indicar una tendencia a favor a seguir de cerca.",
        f"- **{worst_momentum}** muestra el impulso más débil "
        f"({summary.loc[worst_momentum, momentum_col]:+.1%} en el mismo período), "
        "vale la pena entender qué lo está frenando antes de sumar exposición.",
        f"- **{most_volatile}** presenta la mayor volatilidad anualizada "
        f"({summary.loc[most_volatile, 'Volatilidad anualizada']:.1%}), por lo que conviene "
        "dimensionar cualquier posición con cautela.",
        f"- **{worst_drawdown}** tuvo la caída máxima más pronunciada del período "
        f"({summary.loc[worst_drawdown, 'Máx. drawdown']:.1%}), un dato clave para el manejo de riesgo.",
    ]
    return "\n".join(lines)


def render_sector_tab() -> None:
    sector_name = st.selectbox(
        "Sector",
        list(SECTORS.keys()),
        format_func=lambda name: f"{SECTOR_ICONS.get(name, '')} {name}",
    )
    sector = SECTORS[sector_name]

    colored_subheader(f"{SECTOR_ICONS.get(sector_name, '')} {sector_name}")

    col1, col2 = st.columns(2)
    start = col1.date_input(
        "Desde", value=pd.Timestamp(sector["start"]), key=f"start_{sector_name}"
    )
    end = col2.date_input(
        "Hasta", value=pd.Timestamp(sector["end"]), key=f"end_{sector_name}"
    )

    returns = load_returns(sector["stocks"], str(start), str(end))

    colored_subheader("📌 Resumen y recomendación")
    summary = build_summary(returns)
    st.dataframe(
        summary.style.format(
            {
                "Retorno total": "{:.1%}",
                "Retorno anualizado": "{:.1%}",
                "Volatilidad anualizada": "{:.1%}",
                "Sharpe (aprox., rf=0)": "{:.2f}",
                "Máx. drawdown": "{:.1%}",
                summary.columns[-1]: "{:+.1%}",
            }
        )
    )
    st.markdown(build_recommendation(summary))
    st.caption(
        "⚠️ Esto es una lectura descriptiva de datos históricos, no una recomendación "
        "de inversión. Cualquier decisión debe considerar el contexto macro, los "
        "fundamentals de cada compañía y tu propio perfil de riesgo."
    )

    colored_subheader("Retornos diarios")
    st.dataframe(returns)

    colored_subheader("Estadísticas")
    st.dataframe(returns.describe())

    colored_subheader("Retornos acumulados")
    st.line_chart((1 + returns).cumprod())

    colored_subheader("Volatilidad (rolling 21 días, anualizada)")
    st.line_chart(returns.rolling(21).std() * (252 ** 0.5))

    colored_subheader("Dispersión entre activos")
    tickers = st.multiselect(
        "Tickers a comparar",
        options=list(returns.columns),
        default=list(returns.columns[:4]),
        key=f"tickers_{sector_name}",
    )
    if len(tickers) >= 2:
        fig = sns.pairplot(returns[tickers], kind="scatter", plot_kws={"alpha": 0.5})
        st.pyplot(fig.figure)
    else:
        st.info("Elegí al menos dos tickers para ver la dispersión.")


def render_quantstats_tab() -> None:
    colored_subheader("📈 Activo vs. Benchmark (quantstats)")
    st.caption(
        "Elegí un activo y un benchmark invertible (un ETF que efectivamente puedas "
        "comprar, no un índice puro como el ^GSPC) para generar el tearsheet completo "
        "de quantstats: retornos, drawdowns, Sharpe, Sortino y demás métricas estándar."
    )

    col1, col2 = st.columns(2)
    asset_label = col1.selectbox(
        "Activo a analizar",
        [*all_sector_tickers(), "Personalizado"],
        key="qs_asset_label",
    )
    if asset_label == "Personalizado":
        asset = col1.text_input(
            "Ticker del activo", value="AAPL", key="qs_asset_custom"
        ).strip().upper()
    else:
        asset = asset_label

    benchmark_label = col2.selectbox(
        "Benchmark",
        [*BENCHMARK_TICKERS.keys(), "Personalizado"],
        key="qs_benchmark_label",
    )
    if benchmark_label == "Personalizado":
        benchmark_ticker = col2.text_input(
            "Ticker del benchmark", value="SPY", key="qs_benchmark_custom"
        ).strip().upper()
    else:
        benchmark_ticker = BENCHMARK_TICKERS[benchmark_label]

    col3, col4 = st.columns(2)
    start = col3.date_input("Desde", value=pd.Timestamp("2020-01-01"), key="qs_start")
    end = col4.date_input("Hasta", value=pd.Timestamp.today(), key="qs_end")

    if not asset:
        st.info("Ingresá un ticker de activo.")
        return
    if not benchmark_ticker:
        st.info("Ingresá un ticker de benchmark.")
        return
    if asset == benchmark_ticker:
        st.warning("Elegí un activo distinto al benchmark para poder compararlos.")
        return

    if st.button("Generar análisis", key="qs_generate"):
        with st.spinner("Descargando datos y generando el tearsheet de quantstats..."):
            asset_returns = load_price_returns(asset, str(start), str(end))
            benchmark_returns = load_price_returns(benchmark_ticker, str(start), str(end))

            if asset_returns.empty or benchmark_returns.empty:
                st.error("No se encontraron datos para el período y tickers seleccionados.")
                return

            html_report = build_quantstats_report(
                asset_returns, benchmark_returns, asset, benchmark_ticker
            )
        components.html(html_report, height=1600, scrolling=True)


def main() -> None:
    st.set_page_config(page_title="Pizza on Mondays", layout="wide")
    colored_title("🍕 Pizza on Mondays")

    tab_sectores, tab_quantstats = st.tabs(
        ["📊 Análisis por sector", "📈 QuantStats: Activo vs Benchmark"]
    )
    with tab_sectores:
        render_sector_tab()
    with tab_quantstats:
        render_quantstats_tab()


if __name__ == "__main__":
    main()
