import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import streamlit as st

sns.set_theme(style="whitegrid")

MAIN = "#1F5A8A"
MUTED = "#C9CED6"
ALERT = "#C0392B"
GOOD = "#2E8B57"

st.set_page_config(page_title="Dashboard E-Commerce Olist", page_icon="🛒", layout="wide")


# ----------------------------------------------------------------------------- Data
@st.cache_data
def load_data():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main_data.csv")
    df = pd.read_csv(path)
    for col in ["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"]:
        df[col] = pd.to_datetime(df[col])
    return df


def create_monthly_df(df):
    monthly = (df.set_index("order_purchase_timestamp")
               .resample("ME")
               .agg({"order_id": "nunique", "price": "sum"})
               .rename(columns={"order_id": "order_count", "price": "revenue"}))
    monthly.index = monthly.index.to_period("M")
    return monthly


def create_category_df(df):
    cat = (df.groupby("product_category_name_english")
           .agg(revenue=("price", "sum"), items_sold=("order_item_id", "count"))
           .sort_values("revenue", ascending=False))
    cat["share_%"] = cat["revenue"] / cat["revenue"].sum() * 100
    return cat


def create_delay_df(order_df):
    bins = [-np.inf, -15, -8, 0, 7, 14, np.inf]
    labels = [">14 hari lebih cepat", "8–14 hari lebih cepat", "0–7 hari lebih cepat",
              "1–7 hari terlambat", "8–14 hari terlambat", ">14 hari terlambat"]
    tmp = order_df.assign(delay_group=pd.cut(order_df["delay_days"], bins=bins, labels=labels))
    return tmp.groupby("delay_group", observed=False).agg(
        orders=("order_id", "count"),
        avg_review=("review_score", "mean"),
        pct_bad_review=("review_score", lambda s: (s <= 2).mean() * 100 if len(s) else np.nan),
    )


def create_state_df(order_df):
    return (order_df.groupby("customer_state")
            .agg(orders=("order_id", "count"),
                 late_rate=("is_late", lambda s: s.mean() * 100),
                 avg_delivery_days=("delivery_days", "mean"),
                 avg_review=("review_score", "mean"))
            .sort_values("late_rate", ascending=False))


def create_rfm_df(order_df):
    reference_date = order_df["order_purchase_timestamp"].max() + pd.Timedelta(days=1)
    rfm = order_df.groupby("customer_unique_id").agg(
        last_purchase=("order_purchase_timestamp", "max"),
        frequency=("order_id", "nunique"),
        monetary=("payment_value", "sum"),
    ).reset_index()
    rfm["recency"] = (reference_date - rfm["last_purchase"]).dt.days

    # Binning: rank() dipakai agar qcut tetap jalan walau data hasil filter kecil / banyak nilai kembar
    rfm["R_score"] = pd.qcut(rfm["recency"].rank(method="first"), q=5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["M_score"] = pd.qcut(rfm["monetary"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["F_score"] = pd.cut(rfm["frequency"], bins=[0, 1, 2, 3, np.inf], labels=[1, 3, 4, 5]).astype(int)

    conditions = [
        rfm["F_score"] >= 3,
        (rfm["R_score"] >= 4) & (rfm["M_score"] >= 4),
        rfm["R_score"] >= 4,
        rfm["R_score"] == 3,
        rfm["M_score"] >= 4,
    ]
    choices = ["Pelanggan Loyal", "Pelanggan Baru Bernilai Tinggi", "Pelanggan Baru",
               "Perlu Perhatian", "Berisiko Hilang (Bernilai Tinggi)"]
    rfm["segment"] = np.select(conditions, choices, default="Hibernasi")
    return rfm


all_df = load_data()

# ----------------------------------------------------------------------------- Sidebar
min_date = all_df["order_purchase_timestamp"].min().date()
max_date = all_df["order_purchase_timestamp"].max().date()

with st.sidebar:
    st.title("🛒 Olist E-Commerce")
    st.caption("Proyek Analisis Data — Dicoding")

    date_range = st.date_input("Rentang tanggal pembelian", value=[min_date, max_date],
                               min_value=min_date, max_value=max_date)
    if len(date_range) != 2:
        st.info("Pilih tanggal awal dan tanggal akhir.")
        st.stop()
    start_date, end_date = date_range

    all_states = sorted(all_df["customer_state"].unique())
    states = st.multiselect("State pelanggan", options=all_states, default=[],
                            placeholder="Semua state")

    st.markdown("---")
    st.caption("Data: pesanan *delivered* Januari 2017 – Agustus 2018. "
               "Pendapatan = total harga item (tanpa ongkir), dalam BRL.")

main_df = all_df[(all_df["order_purchase_timestamp"].dt.date >= start_date) &
                 (all_df["order_purchase_timestamp"].dt.date <= end_date)]
if states:
    main_df = main_df[main_df["customer_state"].isin(states)]

if main_df.empty:
    st.warning("Tidak ada data untuk filter yang dipilih. Silakan ubah rentang tanggal atau state.")
    st.stop()

order_df = main_df.drop_duplicates("order_id")

# ----------------------------------------------------------------------------- Header & KPI
st.title("Dashboard Analisis E-Commerce Olist")
st.caption(f"Periode {start_date:%d %b %Y} – {end_date:%d %b %Y}"
           + (f" · State: {', '.join(states)}" if states else " · Semua state"))

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total pesanan", f"{order_df['order_id'].nunique():,}")
col2.metric("Total pendapatan", f"BRL {main_df['price'].sum():,.0f}")
col3.metric("Rata-rata skor ulasan", f"{order_df['review_score'].mean():.2f} / 5")
col4.metric("Pesanan terlambat", f"{order_df['is_late'].mean() * 100:.1f}%")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Tren Penjualan", "🏷️ Kategori Produk", "🚚 Pengiriman & Ulasan",
                                        "👥 Segmentasi Pelanggan (RFM)", "🗺️ Peta Pesanan"])

# ----------------------------------------------------------------------------- Tab 1
with tab1:
    st.subheader("Pertanyaan 1: Tren pesanan dan pendapatan bulanan")
    monthly_df = create_monthly_df(main_df)
    peak = monthly_df["revenue"].idxmax()

    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True)
    x = np.arange(len(monthly_df))
    is_peak = monthly_df.index == peak
    axes[0].plot(x, monthly_df["revenue"] / 1000, marker="o", linewidth=2.5, color=MAIN)
    axes[0].scatter(x[is_peak], monthly_df["revenue"][is_peak] / 1000, s=150, color=ALERT, zorder=3)
    axes[0].set_ylabel("Pendapatan (ribu BRL)")
    axes[0].set_ylim(0, None)
    axes[0].set_title("Pendapatan bulanan", loc="left")
    axes[1].bar(x, monthly_df["order_count"], color=np.where(is_peak, ALERT, MAIN))
    axes[1].set_ylabel("Jumlah pesanan")
    axes[1].set_title("Jumlah pesanan bulanan", loc="left")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(monthly_df.index.strftime("%b\n%Y"), fontsize=9)
    axes[1].yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    plt.tight_layout()
    st.pyplot(fig)

    c1, c2 = st.columns(2)
    c1.metric("Bulan puncak pendapatan", peak.strftime("%B %Y"),
              f"BRL {monthly_df.loc[peak, 'revenue']:,.0f}", delta_color="off")
    if len(monthly_df) > 1:
        last, prev = monthly_df["revenue"].iloc[-1], monthly_df["revenue"].iloc[-2]
        c2.metric(f"Pendapatan {monthly_df.index[-1].strftime('%b %Y')}", f"BRL {last:,.0f}",
                  f"{(last / prev - 1) * 100:.1f}% vs bulan sebelumnya")
    with st.expander("Insight (seluruh periode, tanpa filter)"):
        st.write("Pendapatan tumbuh pesat sepanjang 2017 dan memuncak pada **November 2017** (Black Friday). "
                 "Pendapatan Agustus 2018 tumbuh **51,2%** dibanding Agustus 2017, tetapi sepanjang 2018 "
                 "pertumbuhan cenderung mendatar.")

# ----------------------------------------------------------------------------- Tab 2
with tab2:
    st.subheader("Pertanyaan 2: Kategori produk dengan pendapatan tertinggi dan terendah")
    category_df = create_category_df(main_df)
    n_show = st.slider("Jumlah kategori yang ditampilkan", min_value=5, max_value=15, value=10)

    top = category_df.head(n_show).sort_values("revenue")
    bottom = category_df.tail(n_show).sort_values("revenue")

    fig, axes = plt.subplots(1, 2, figsize=(15, 0.55 * n_show + 2))
    axes[0].barh(top.index, top["revenue"] / 1000,
                 color=[MAIN if i == len(top) - 1 else MUTED for i in range(len(top))])
    axes[0].set_title(f"{n_show} kategori dengan pendapatan tertinggi", loc="left")
    axes[0].set_xlabel("Pendapatan (ribu BRL)")
    for i, (rev, share) in enumerate(zip(top["revenue"], top["share_%"])):
        axes[0].text(rev / 1000, i, f" {share:.1f}%", va="center", fontsize=9)
    axes[1].barh(bottom.index, bottom["revenue"],
                 color=[ALERT if i == 0 else MUTED for i in range(len(bottom))])
    axes[1].set_title(f"{n_show} kategori dengan pendapatan terendah", loc="left")
    axes[1].set_xlabel("Pendapatan (BRL)")
    axes[1].xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    plt.tight_layout()
    st.pyplot(fig)

    st.metric(f"Kontribusi {n_show} kategori teratas terhadap pendapatan",
              f"{category_df['share_%'].head(n_show).sum():.1f}%")
    st.caption("Catatan: skala sumbu-x kedua grafik berbeda (ribu BRL vs BRL).")

# ----------------------------------------------------------------------------- Tab 3
with tab3:
    st.subheader("Pertanyaan 3: Keterlambatan pengiriman dan skor ulasan")
    late_score = order_df.loc[order_df["is_late"], "review_score"].mean()
    ontime_score = order_df.loc[~order_df["is_late"], "review_score"].mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("Skor ulasan — tepat waktu", f"{ontime_score:.2f}")
    c2.metric("Skor ulasan — terlambat", f"{late_score:.2f}" if not np.isnan(late_score) else "-")
    c3.metric("Selisih skor", f"{ontime_score - late_score:.2f} poin" if not np.isnan(late_score) else "-")

    delay_df = create_delay_df(order_df)
    colors = [GOOD] * 3 + [ALERT] * 3
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].bar(delay_df.index.astype(str), delay_df["avg_review"].fillna(0), color=colors)
    axes[0].set_ylim(0, 5)
    axes[0].set_ylabel("Rata-rata skor ulasan (1–5)")
    axes[0].set_title("Rata-rata skor ulasan menurut ketepatan pengiriman", loc="left")
    axes[1].bar(delay_df.index.astype(str), delay_df["pct_bad_review"].fillna(0), color=colors)
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("Ulasan skor 1–2 (%)")
    axes[1].set_title("Persentase ulasan buruk", loc="left")
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("##### Persentase keterlambatan per state")
    min_orders = st.slider("Minimal jumlah pesanan per state", 0, 500, 100, step=50)
    state_df = create_state_df(order_df)
    state_plot = state_df[state_df["orders"] >= min_orders].sort_values("late_rate")
    if state_plot.empty:
        st.info("Tidak ada state yang memenuhi batas minimal pesanan.")
    else:
        overall = order_df["is_late"].mean() * 100
        top5 = state_plot["late_rate"].nlargest(5).index
        fig, ax = plt.subplots(figsize=(12, 0.32 * len(state_plot) + 1.5))
        ax.barh(state_plot.index, state_plot["late_rate"],
                color=[ALERT if s in top5 else MUTED for s in state_plot.index])
        ax.axvline(overall, color="black", linestyle="--", linewidth=1, alpha=0.6, zorder=0)
        ax.text(overall, -0.9, f" rata-rata {overall:.1f}%", fontsize=9)
        ax.set_xlabel("Pesanan terlambat (%)")
        ax.set_ylabel("State")
        plt.tight_layout()
        st.pyplot(fig)
        with st.expander("Lihat tabel per state"):
            st.dataframe(state_df.round(2))

# ----------------------------------------------------------------------------- Tab 4
with tab4:
    st.subheader("Segmentasi pelanggan dengan RFM Analysis")
    rfm_df = create_rfm_df(order_df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Rata-rata recency", f"{rfm_df['recency'].mean():.0f} hari")
    c2.metric("Rata-rata frequency", f"{rfm_df['frequency'].mean():.2f} pesanan")
    c3.metric("Rata-rata monetary", f"BRL {rfm_df['monetary'].mean():,.2f}")

    seg = (rfm_df.groupby("segment")
           .agg(customers=("customer_unique_id", "count"), avg_monetary=("monetary", "mean"),
                total_monetary=("monetary", "sum"))
           .sort_values("customers"))
    seg["share_%"] = seg["customers"] / seg["customers"].sum() * 100
    highlight = "Berisiko Hilang (Bernilai Tinggi)"
    colors = [ALERT if s == highlight else MUTED for s in seg.index]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    axes[0].barh(seg.index, seg["customers"], color=colors)
    axes[0].set_xlabel("Jumlah pelanggan")
    axes[0].set_title("Jumlah pelanggan per segmen", loc="left")
    axes[0].xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    for i, (n, s) in enumerate(zip(seg["customers"], seg["share_%"])):
        axes[0].text(n, i, f" {s:.1f}%", va="center", fontsize=9)
    axes[1].barh(seg.index, seg["avg_monetary"], color=colors)
    axes[1].set_xlabel("Rata-rata belanja per pelanggan (BRL)")
    axes[1].set_title("Rata-rata nilai belanja per segmen", loc="left")
    plt.tight_layout()
    st.pyplot(fig)

    with st.expander("Aturan segmentasi"):
        st.markdown(
            "- **Pelanggan Loyal**: belanja ≥ 2 kali\n"
            "- **Pelanggan Baru Bernilai Tinggi**: belanja 1 kali, R_score ≥ 4, M_score ≥ 4\n"
            "- **Pelanggan Baru**: belanja 1 kali, R_score ≥ 4, M_score < 4\n"
            "- **Perlu Perhatian**: R_score = 3\n"
            "- **Berisiko Hilang (Bernilai Tinggi)**: R_score ≤ 2, M_score ≥ 4\n"
            "- **Hibernasi**: selain kondisi di atas\n\n"
            "R_score dan M_score dihitung dengan binning kuantil (1–5) pada data hasil filter."
        )

# ----------------------------------------------------------------------------- Tab 5
with tab5:
    st.subheader("Sebaran lokasi pesanan")
    st.caption("Titik merah = pesanan terlambat, titik biru = tepat waktu. "
               "Maksimal 20.000 titik diambil secara acak agar peta tetap ringan.")
    geo = order_df.dropna(subset=["geolocation_lat", "geolocation_lng"])
    geo = geo.sample(min(len(geo), 20000), random_state=42)
    map_df = pd.DataFrame({
        "lat": geo["geolocation_lat"],
        "lon": geo["geolocation_lng"],
        "color": np.where(geo["is_late"], "#C0392BCC", "#1F5A8A55"),
    })
    st.map(map_df, latitude="lat", longitude="lon", color="color", size=2500)

st.caption("© 2026 Abhista Arief Bonianto — Proyek Analisis Data Dicoding")
