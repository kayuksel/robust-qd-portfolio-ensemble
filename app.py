import streamlit as st
import pandas as pd
import numpy as np
import torch, pywt
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from utils import *

def clip_outliers(data, quantile=0.95, factor = 2):
    q3 = data.abs().stack().quantile(quantile)
    q1 = data.abs().stack().quantile(1 - quantile)
    iqr = q3 - q1
    q3_adjusted = q3 + factor * iqr
    return data.clip(lower=-q3_adjusted, upper=q3_adjusted)

def process_column(signal, wavelet='sym20', level=1):
    # Perform wavelet denoising
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    threshold = np.sqrt(2 * np.log(len(signal)))
    denoised_coeffs = [pywt.threshold(c, threshold, mode='soft') if i > 0 else c for i, c in enumerate(coeffs)]
    denoised_signal = pywt.waverec(denoised_coeffs, wavelet)
            
    # Convert denoised cumulative returns back to log returns
    denoised_signal = denoised_signal[:len(signal)]  # Ensure length consistency
    denoised_log_returns = np.log(denoised_signal[1:] / denoised_signal[:-1])
    return denoised_log_returns

# Streamlit configuration
st.set_page_config(
    page_title="Robust Portfolio Ensemble",
    page_icon="📈"
)

st.title("Robust Quality-Diversity Portfolio Ensemble Optimization Technique")
st.markdown("This app allows you to upload daily asset log returns, select an index, and run portfolio optimization.")

uploaded_file = st.file_uploader("Upload your dataset (CSV format starting with a date column, and contains asset log returns as subsequent columns)", type="csv")

if uploaded_file:
    udata = pd.read_csv(uploaded_file, parse_dates=True, index_col=0)
    udata.index = pd.to_datetime(udata.index).tz_localize(None)

    udata = clip_outliers(udata)
    data_cumsum = udata.cumsum()
    cumulative_log_returns = np.exp(data_cumsum)

    with ThreadPoolExecutor() as executor:
        results = {col: executor.submit(process_column, cumulative_log_returns[col].values) for col in cumulative_log_returns.columns}
        denoised_log_returns = {col: res.result() for col, res in results.items()}

    udata = pd.DataFrame(denoised_log_returns, index=udata.index[1:])
    st.dataframe(udata.head())

    st.write("### Select Parameters")
    date_range = st.date_input(
        "Pick a date for the test data cutoff:",
        value=udata.index[-len(udata) // 4].to_pydatetime().date(),
        min_value=udata.index[0].to_pydatetime().date(),
        max_value=udata.index[-1].to_pydatetime().date()
    )

    index_ticker = st.selectbox("Select a benchmark index ticker:", options=udata.columns)

    batch_size = st.slider("The number of sub-portfolios:", min_value=1024, max_value=8192, value=4096, step=512)
    max_epochs = st.slider("The number of epochs to train:", min_value=1, max_value=100, value=100, step=1)

    if st.button("Start Optimization"):
        data = udata.drop(columns=index_ticker)
        cutoff_index = data.index.get_indexer([pd.Timestamp(date_range)], method='nearest')[0]
        portfolio_optimizer = RobustQDPortfolioEnsemble(data.values, udata[index_ticker].values, batch_size)

        with ThreadPoolExecutor() as executor:
            future = executor.submit(portfolio_optimizer.optimize, cutoff_index, max_epochs)
            best_weights, best_epoch, test_losses, training_entropies = future.result()

        st.write("### Training Analysis")
        fig, ax1 = plt.subplots(figsize=(10, 6))
        epochs = list(range(1, len(test_losses) + 1))
        ax1.plot(epochs, test_losses, label="Validation Loss", color="blue", linestyle="-")
        ax1.set_ylabel("Validation Loss", color="blue")
        ax1.tick_params(axis='y', labelcolor="blue")
        ax1.set_xlabel("Epochs")
        ax1.set_title("Loss and Entropy over Epochs")
        ax1.grid(True)

        ax2 = ax1.twinx()
        ax2.plot(epochs, training_entropies, label="Eigen Entropy", color="green", linestyle="--")
        ax2.set_ylabel("Eigen Entropy", color="green")
        ax2.tick_params(axis='y', labelcolor="green")

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
        plt.tight_layout()
        st.pyplot(fig)

        if (best_epoch + 1) != max_epochs:
            st.markdown(f"Consider reducing the number of training epochs to {best_epoch + 1}.")

        st.write("### Cumulative Returns")
        # Plot cumulative returns against index
        index_cumulative_returns = udata[index_ticker][cutoff_index:].cumsum()
        portfolio_cumulative_returns = portfolio_optimizer.get_test_returns(cutoff_index)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data.index[cutoff_index:], index_cumulative_returns, label=f"{index_ticker} (Index)", linestyle="--", color="red")
        ax.plot(data.index[cutoff_index:], portfolio_cumulative_returns, label="Portfolio", color="blue")
        ax.set_title("Cumulative Returns: Portfolio vs Index")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Returns")
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown(f"The above chart shows the method's validation performance. Wait, I'm retraining with the whole data ...")

        with ThreadPoolExecutor() as executor:
            future = executor.submit(portfolio_optimizer.final_portfolio, max_epochs)
            avg_weights = future.result()

        non_zero_indices = avg_weights > 1e-4
        non_zero_weights = avg_weights[non_zero_indices]
        non_zero_weights /= non_zero_weights.sum()
        non_zero_assets = [data.columns[i] for i in range(len(data.columns)) if non_zero_indices[i]]

        sorted_indices = np.argsort(non_zero_weights)[::-1]
        non_zero_weights = non_zero_weights[sorted_indices]
        non_zero_assets = [non_zero_assets[i] for i in sorted_indices]

        # Portfolio weights pie chart
        st.write("### Portfolio Weights")
        fig, ax = plt.subplots()
        wedges, texts, autotexts = ax.pie(
            non_zero_weights, 
            labels=non_zero_assets, 
            autopct='%1.1f%%', 
            startangle=90
        )
        ax.axis('equal')
        plt.tight_layout()
        st.pyplot(fig)

        # Export CSV
        weights_df = pd.DataFrame({
            "Assets": non_zero_assets,
            "Weight": non_zero_weights
        }).sort_values(by="Weight", ascending=False)

        csv = weights_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Portfolio",
            data=csv,
            file_name="portfolio_weights.csv",
            mime="text/csv"
        )

        st.dataframe(weights_df.set_index("Assets"))
