from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path(__file__).parent / "data_hotel_booking_demand.csv"
ROOM_GROUP = {
    "A": "A", "B": "B", "C": "C", "D": "D", "E": "E",
    "F": "F", "G": "G", "H": "H", "L": "Other", "P": "Other",
}
NUMERICAL_FEATURES = [
    "booking_changes",
    "required_car_parking_spaces",
    "total_of_special_requests",
    "has_previous_cancellations",
    "is_domestic",
    "has_high_booking_changes",
]
CATEGORICAL_FEATURES = [
    "market_segment",
    "deposit_type",
    "customer_type",
    "reserved_room_type_grouped",
]


st.set_page_config(
    page_title="Hotel Cancellation Analytics",
    page_icon="🏨",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; padding-bottom: 3rem;}
    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #dbe5ef;
        border-radius: 12px;
        padding: 16px;
    }
    [data-testid="stMetricValue"] {color: #153f63;}
    div[data-testid="stSidebarContent"] {background-color: #f7fafc;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Memuat dan membersihkan data booking...")
def load_data(path: Path) -> tuple[pd.DataFrame, int]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {path.name}")
    raw = pd.read_csv(path)
    original_rows = len(raw)
    raw["country"] = raw["country"].fillna(raw["country"].mode()[0])
    clean = raw.drop_duplicates().copy()
    clean["status"] = clean["is_canceled"].map(
        {0: "Tidak dibatalkan", 1: "Dibatalkan"}
    )
    clean["reserved_room_type_grouped"] = (
        clean["reserved_room_type"].map(ROOM_GROUP).fillna("Other")
    )
    return clean, original_rows


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    features = frame.drop(columns=["is_canceled", "status"], errors="ignore").copy()
    if "reserved_room_type_grouped" not in features:
        features["reserved_room_type_grouped"] = (
            features["reserved_room_type"].map(ROOM_GROUP).fillna("Other")
        )
    features["has_previous_cancellations"] = (
        features["previous_cancellations"] > 0
    ).astype(int)
    features["is_domestic"] = features["country"].eq("PRT").astype(int)
    features["has_high_booking_changes"] = (
        features["booking_changes"] > 2
    ).astype(int)
    return features.drop(
        columns=[
            "previous_cancellations",
            "country",
            "days_in_waiting_list",
            "reserved_room_type",
        ]
    )


@st.cache_resource(show_spinner="Menyiapkan model Logistic Regression...")
def train_model(clean_data: pd.DataFrame):
    x = make_features(clean_data)
    y = clean_data["is_canceled"]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipeline, NUMERICAL_FEATURES),
            ("cat_onehot", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    random_state=42,
                    solver="liblinear",
                    class_weight="balanced",
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "recall": recall_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "accuracy": accuracy_score(y_test, predictions),
        "auc": roc_auc_score(y_test, probabilities),
        "confusion_matrix": confusion_matrix(y_test, predictions),
    }
    evaluation = pd.DataFrame(
        {"actual": y_test.to_numpy(), "probability": probabilities}
    )
    return model, metrics, evaluation


def apply_filters(data: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filter dashboard")
    segments = st.sidebar.multiselect(
        "Market segment", sorted(data["market_segment"].unique())
    )
    deposits = st.sidebar.multiselect(
        "Deposit type", sorted(data["deposit_type"].unique())
    )
    customers = st.sidebar.multiselect(
        "Customer type", sorted(data["customer_type"].unique())
    )
    countries = st.sidebar.multiselect(
        "Negara tamu", sorted(data["country"].unique())
    )
    filtered = data
    if segments:
        filtered = filtered[filtered["market_segment"].isin(segments)]
    if deposits:
        filtered = filtered[filtered["deposit_type"].isin(deposits)]
    if customers:
        filtered = filtered[filtered["customer_type"].isin(customers)]
    if countries:
        filtered = filtered[filtered["country"].isin(countries)]
    if st.sidebar.button("Reset filter", use_container_width=True):
        st.rerun()
    st.sidebar.caption(f"{len(filtered):,} dari {len(data):,} booking ditampilkan")
    return filtered.copy()


try:
    data, raw_rows = load_data(DATA_PATH)
    model, model_metrics, evaluation = train_model(data)
except (FileNotFoundError, ValueError, KeyError) as error:
    st.error(str(error))
    st.info("Pastikan `data_hotel_booking_demand.csv` berada bersama `app.py`.")
    st.stop()

filtered = apply_filters(data)

st.title("Hotel Cancellation Analytics")
st.caption(
    "Dashboard eksplorasi dan prediksi risiko pembatalan pemesanan hotel"
)

if filtered.empty:
    st.warning("Tidak ada booking yang sesuai dengan kombinasi filter.")
    st.stop()

cancelled = int(filtered["is_canceled"].sum())
cancel_rate = cancelled / len(filtered) * 100
duplicates_removed = raw_rows - len(data)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Booking ditampilkan", f"{len(filtered):,}")
kpi2.metric("Dibatalkan", f"{cancelled:,}", f"{cancel_rate:.1f}% cancellation rate")
kpi3.metric("Recall model", f"{model_metrics['recall']:.1%}", "threshold 0,5")
kpi4.metric("Duplikat dihapus", f"{duplicates_removed:,}")

overview_tab, eda_tab, model_tab, prediction_tab, data_tab = st.tabs(
    ["Overview", "EDA", "Evaluasi Model", "Prediksi", "Data"]
)

with overview_tab:
    left, right = st.columns([1.2, 1])
    with left:
        status_counts = filtered["status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Booking"]
        fig_status = px.bar(
            status_counts,
            x="Status",
            y="Booking",
            color="Status",
            text_auto=",.0f",
            color_discrete_map={
                "Tidak dibatalkan": "#17486e",
                "Dibatalkan": "#ed6b2f",
            },
            title="Distribusi status booking",
        )
        fig_status.update_layout(showlegend=False)
        st.plotly_chart(fig_status, use_container_width=True)
    with right:
        fig_pie = px.pie(
            status_counts,
            names="Status",
            values="Booking",
            hole=0.58,
            color="Status",
            color_discrete_map={
                "Tidak dibatalkan": "#17486e",
                "Dibatalkan": "#ed6b2f",
            },
            title="Komposisi booking",
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

    segment_summary = (
        filtered.groupby("market_segment", observed=True)["is_canceled"]
        .agg(booking="size", cancellation_rate="mean")
        .reset_index()
        .sort_values("booking", ascending=False)
    )
    segment_summary["cancellation_rate"] *= 100
    fig_segment = px.bar(
        segment_summary,
        x="market_segment",
        y="booking",
        color="cancellation_rate",
        color_continuous_scale=["#b7d5e8", "#ed6b2f"],
        text_auto=",.0f",
        title="Volume dan cancellation rate per market segment",
        labels={
            "market_segment": "Market segment",
            "booking": "Jumlah booking",
            "cancellation_rate": "Cancellation rate (%)",
        },
    )
    st.plotly_chart(fig_segment, use_container_width=True)

with eda_tab:
    st.subheader("Faktor yang terkait dengan pembatalan")
    col1, col2 = st.columns(2)
    for container, column, title in [
        (col1, "deposit_type", "Cancellation rate per deposit type"),
        (col2, "customer_type", "Cancellation rate per customer type"),
    ]:
        summary = (
            filtered.groupby(column, observed=True)["is_canceled"]
            .agg(booking="size", cancellation_rate="mean")
            .reset_index()
        )
        summary["cancellation_rate"] *= 100
        fig = px.bar(
            summary.sort_values("cancellation_rate"),
            x="cancellation_rate",
            y=column,
            orientation="h",
            text_auto=".1f",
            color_discrete_sequence=["#ed6b2f"],
            title=title,
            labels={"cancellation_rate": "Cancellation rate (%)"},
            hover_data={"booking": True},
        )
        fig.update_traces(texttemplate="%{x:.1f}%", textposition="outside")
        container.plotly_chart(fig, use_container_width=True)

    numeric = st.selectbox(
        "Pilih fitur numerik",
        [
            "previous_cancellations",
            "booking_changes",
            "days_in_waiting_list",
            "required_car_parking_spaces",
            "total_of_special_requests",
        ],
    )
    fig_numeric = px.histogram(
        filtered,
        x=numeric,
        color="status",
        barmode="overlay",
        nbins=30,
        color_discrete_map={
            "Tidak dibatalkan": "#17486e",
            "Dibatalkan": "#ed6b2f",
        },
        title=f"Distribusi {numeric}",
    )
    st.plotly_chart(fig_numeric, use_container_width=True)
    st.info(
        "Cancellation rate menunjukkan asosiasi dalam data, bukan hubungan sebab-akibat. "
        "Anomali Non Refund perlu dikonfirmasi dengan tim operasional hotel."
    )

with model_tab:
    st.subheader("Evaluasi Logistic Regression pada test set")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Recall", f"{model_metrics['recall']:.3f}")
    m2.metric("Precision", f"{model_metrics['precision']:.3f}")
    m3.metric("F1", f"{model_metrics['f1']:.3f}")
    m4.metric("Accuracy", f"{model_metrics['accuracy']:.3f}")
    m5.metric("AUC-ROC", f"{model_metrics['auc']:.3f}")

    cm = model_metrics["confusion_matrix"]
    left, right = st.columns([1, 1.3])
    with left:
        fig_cm = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=["Prediksi tidak batal", "Prediksi batal"],
                y=["Aktual tidak batal", "Aktual batal"],
                text=cm,
                texttemplate="%{text}",
                colorscale=[[0, "#eef5f9"], [1, "#17486e"]],
                showscale=False,
            )
        )
        fig_cm.update_layout(title="Confusion matrix", height=420)
        left.plotly_chart(fig_cm, use_container_width=True)
    with right:
        threshold = st.slider(
            "Simulasi probability threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.50,
            step=0.05,
        )
        simulated = (evaluation["probability"] >= threshold).astype(int)
        threshold_recall = recall_score(evaluation["actual"], simulated)
        threshold_precision = precision_score(
            evaluation["actual"], simulated, zero_division=0
        )
        flagged = int(simulated.sum())
        a, b, c = st.columns(3)
        a.metric("Recall", f"{threshold_recall:.1%}")
        b.metric("Precision", f"{threshold_precision:.1%}")
        c.metric("Booking ditandai", f"{flagged:,}")
        st.write(
            "Threshold rendah menangkap lebih banyak pembatalan, tetapi menambah beban "
            "kontak tim. Threshold operasional perlu mengikuti kapasitas intervensi."
        )
        threshold_rows = []
        for value in np.arange(0.1, 0.91, 0.05):
            pred = (evaluation["probability"] >= value).astype(int)
            threshold_rows.append(
                {
                    "Threshold": value,
                    "Recall": recall_score(evaluation["actual"], pred),
                    "Precision": precision_score(
                        evaluation["actual"], pred, zero_division=0
                    ),
                }
            )
        threshold_df = pd.DataFrame(threshold_rows)
        fig_threshold = px.line(
            threshold_df,
            x="Threshold",
            y=["Recall", "Precision"],
            markers=True,
            title="Trade-off threshold",
        )
        right.plotly_chart(fig_threshold, use_container_width=True)

with prediction_tab:
    st.subheader("Prediksi risiko satu booking")
    st.caption("Isi karakteristik booking lalu tekan tombol prediksi.")
    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        country = c1.selectbox("Country", sorted(data["country"].unique()))
        market_segment = c2.selectbox(
            "Market segment", sorted(data["market_segment"].unique())
        )
        deposit_type = c3.selectbox(
            "Deposit type", sorted(data["deposit_type"].unique())
        )
        customer_type = c1.selectbox(
            "Customer type", sorted(data["customer_type"].unique())
        )
        reserved_room_type = c2.selectbox(
            "Reserved room type", sorted(data["reserved_room_type"].unique())
        )
        previous_cancellations = c3.number_input(
            "Previous cancellations", min_value=0, max_value=30, value=0
        )
        booking_changes = c1.number_input(
            "Booking changes", min_value=0, max_value=25, value=0
        )
        days_in_waiting_list = c2.number_input(
            "Days in waiting list", min_value=0, max_value=400, value=0
        )
        parking = c3.number_input(
            "Required car parking spaces", min_value=0, max_value=8, value=0
        )
        special_requests = c1.number_input(
            "Total special requests", min_value=0, max_value=5, value=0
        )
        submitted = st.form_submit_button(
            "Prediksi risiko", use_container_width=True
        )

    if submitted:
        booking = pd.DataFrame(
            [
                {
                    "country": country,
                    "market_segment": market_segment,
                    "previous_cancellations": previous_cancellations,
                    "booking_changes": booking_changes,
                    "deposit_type": deposit_type,
                    "days_in_waiting_list": days_in_waiting_list,
                    "customer_type": customer_type,
                    "reserved_room_type": reserved_room_type,
                    "required_car_parking_spaces": parking,
                    "total_of_special_requests": special_requests,
                }
            ]
        )
        probability = model.predict_proba(make_features(booking))[0, 1]
        st.metric("Probabilitas pembatalan", f"{probability:.1%}")
        if probability >= 0.7:
            st.error("Risiko tinggi: prioritaskan konfirmasi personal.")
        elif probability >= 0.5:
            st.warning("Risiko menengah: pertimbangkan konfirmasi otomatis.")
        else:
            st.success("Risiko rendah: cukup dipantau.")
        st.caption(
            "Skor adalah keluaran model, bukan kepastian. Gunakan sebagai alat "
            "prioritisasi dan tetap lakukan verifikasi manusia."
        )

with data_tab:
    st.subheader("Data booking terfilter")
    display_columns = [
        "country",
        "market_segment",
        "previous_cancellations",
        "booking_changes",
        "deposit_type",
        "days_in_waiting_list",
        "customer_type",
        "reserved_room_type",
        "required_car_parking_spaces",
        "total_of_special_requests",
        "status",
    ]
    st.dataframe(
        filtered[display_columns],
        use_container_width=True,
        hide_index=True,
        height=480,
    )
    st.download_button(
        "Download data terfilter",
        data=filtered[display_columns].to_csv(index=False).encode("utf-8"),
        file_name="hotel_booking_filtered.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "Capstone Project Module 3 · Purwadhika Digital Technology School · "
    "Model digunakan untuk pembelajaran dan bukan keputusan operasional otomatis."
)

