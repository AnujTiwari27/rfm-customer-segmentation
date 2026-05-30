import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

st.set_page_config(
    page_title="RFM Customer Segmentation",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    section[data-testid="stSidebar"] { background-color: #161b22; }
    .rec-card {
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 10px;
        border-left: 5px solid;
    }
    .rec-tag {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ── Segment config — single source of truth ───────────────────────────────────
SEGMENT_CONFIG = {
    "Champions": {
        "color": "#e94560",
        "icon": "🏆",
        "desc": "Buy often, recently, spend most",
        "priority": "HIGH VALUE",
        "priority_color": "#e94560",
        "actions": [
            "Upsell premium & new product lines",
            "Invite to VIP loyalty programme",
            "Offer early access to launches",
            "Ask for reviews & referrals",
            "Send personalised thank-you gifts",
        ],
        "email_subject": "You're one of our VIPs — exclusive access inside 🎁",
        "campaign_type": "Loyalty & Upsell",
        "expected_roi": "High — best converting segment",
        "discount": "No discount needed — they already love you",
    },
    "Loyal Customers": {
        "color": "#4fc3f7",
        "icon": "💙",
        "desc": "Regular buyers, good value",
        "priority": "RETAIN",
        "priority_color": "#4fc3f7",
        "actions": [
            "Upsell to higher-value products",
            "Enrol in points/rewards programme",
            "Request product reviews",
            "Send personalised recommendations",
            "Offer bundle deals",
        ],
        "email_subject": "We appreciate you — here's something special 💙",
        "campaign_type": "Retention & Upsell",
        "expected_roi": "High — low acquisition cost",
        "discount": "5-10% loyalty discount",
    },
    "Potential Loyalists": {
        "color": "#81c784",
        "icon": "🌱",
        "desc": "Recent buyers with potential",
        "priority": "NURTURE",
        "priority_color": "#81c784",
        "actions": [
            "Offer membership or loyalty programme",
            "Send onboarding email sequence",
            "Share product education content",
            "Provide personalised picks based on history",
            "Offer first-repeat-purchase incentive",
        ],
        "email_subject": "We think you'll love these picks just for you 🌱",
        "campaign_type": "Nurture & Convert",
        "expected_roi": "Medium-High — high growth potential",
        "discount": "10-15% second purchase discount",
    },
    "At Risk": {
        "color": "#f5a623",
        "icon": "⚠️",
        "desc": "Used to buy, now inactive",
        "priority": "WIN BACK",
        "priority_color": "#f5a623",
        "actions": [
            "Send discount voucher immediately",
            "Launch win-back email campaign",
            "Ask for feedback — why did they stop?",
            "Showcase new arrivals they missed",
            "Offer free shipping on next order",
        ],
        "email_subject": "We miss you! Here's 20% off — just for you ⚠️",
        "campaign_type": "Win-Back",
        "expected_roi": "Medium — act fast before they churn fully",
        "discount": "20-25% win-back discount",
    },
    "Need Attention": {
        "color": "#7b68ee",
        "icon": "🔮",
        "desc": "Average on all RFM metrics",
        "priority": "ENGAGE",
        "priority_color": "#7b68ee",
        "actions": [
            "Send limited-time flash sale offer",
            "Share bestseller / trending products",
            "Trigger browsing-based re-engagement",
            "Offer free gift with next purchase",
            "Use social proof (reviews, ratings)",
        ],
        "email_subject": "Don't miss out — trending picks this week 🔥",
        "campaign_type": "Re-engagement",
        "expected_roi": "Medium — needs compelling offer",
        "discount": "15% time-limited offer",
    },
    "Lost": {
        "color": "#8b949e",
        "icon": "💤",
        "desc": "Long inactive, low value",
        "priority": "LAST CHANCE",
        "priority_color": "#8b949e",
        "actions": [
            "Send one last-chance re-engagement email",
            "Offer largest discount (30%+)",
            "Ask if they want to unsubscribe (cleans list)",
            "Suppress from regular campaigns to save cost",
            "Accept graceful churn for low-value lost customers",
        ],
        "email_subject": "Last chance — we want you back 💤",
        "campaign_type": "Last-Chance",
        "expected_roi": "Low — minimal spend justified",
        "discount": "30%+ last-chance discount",
    },
}

# ── Sidebar nav ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 RFM Segmentation")
    st.markdown("---")
    page = st.selectbox("Menu", [
        "Business Understanding",
        "Data Understanding",
        "Data Preparation",
        "Modeling & Evaluation",
        "Predict",
        "Email Campaign"
    ])
    st.markdown("---")
    uploaded = st.file_uploader("Upload your CSV", type=["csv"])
    st.caption("Columns needed: InvoiceNo, InvoiceDate, CustomerID, Quantity, UnitPrice")

# ── Data helpers ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

@st.cache_data
def compute_rfm(df):
    df = df.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df = df.dropna(subset=["CustomerID"])
    df["CustomerID"] = df["CustomerID"].astype(int)
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg(
        Recency   =("InvoiceDate",  lambda x: (snapshot - x.max()).days),
        Frequency =("InvoiceNo",    "nunique"),
        Monetary  =("TotalPrice",   "sum")
    ).reset_index()
    rfm["Monetary"] = rfm["Monetary"].round(2)
    return rfm

@st.cache_data
def elbow_data(rfm, max_k=19):
    X = StandardScaler().fit_transform(rfm[["Recency","Frequency","Monetary"]].values)
    inertias = []
    for k in range(1, max_k + 1):
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        km.fit(X)
        inertias.append(km.inertia_)
    arr = np.array(inertias)
    return list(range(1, max_k + 1)), (arr / arr[0]).tolist()

@st.cache_data
def run_kmeans(rfm, k):
    X = rfm[["Recency","Frequency","Monetary"]].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = KMeans(n_clusters=k, init="k-means++", n_init=15, random_state=42)
    labels = model.fit_predict(Xs)
    sil = silhouette_score(Xs, labels)
    return labels, scaler, model, sil

def seg_label(rec, freq, mon, rm, fm, mm):
    if rec <= rm * 0.5 and freq >= fm * 1.5 and mon >= mm * 1.5:
        return "Champions", SEGMENT_CONFIG["Champions"]["color"]
    elif rec <= rm and freq >= fm:
        return "Loyal Customers", SEGMENT_CONFIG["Loyal Customers"]["color"]
    elif rec <= rm * 0.7 and freq < fm:
        return "Potential Loyalists", SEGMENT_CONFIG["Potential Loyalists"]["color"]
    elif rec > rm * 1.5 and freq >= fm:
        return "At Risk", SEGMENT_CONFIG["At Risk"]["color"]
    elif rec > rm * 1.5 and freq < fm and mon < mm:
        return "Lost", SEGMENT_CONFIG["Lost"]["color"]
    else:
        return "Need Attention", SEGMENT_CONFIG["Need Attention"]["color"]

def to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")

def marketing_rec_card(seg_name):
    """Render a full marketing recommendation card for a segment."""
    cfg = SEGMENT_CONFIG.get(seg_name, {})
    if not cfg:
        return
    color   = cfg["color"]
    icon    = cfg["icon"]
    actions = cfg["actions"]
    st.markdown(f"""
    <div class="rec-card" style="background:#1c2333;border-color:{color};">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">
            <span style="font-size:18px;font-weight:700;color:{color}">{icon} {seg_name}</span>
            <span class="rec-tag" style="background:{color}22;color:{color};border:1px solid {color}">
                {cfg['priority']}
            </span>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:10px;font-size:13px;">
            <div><span style="color:#8b949e">Campaign type:</span>
                 <span style="color:#e0e0e0;margin-left:4px">{cfg['campaign_type']}</span></div>
            <div><span style="color:#8b949e">Expected ROI:</span>
                 <span style="color:#e0e0e0;margin-left:4px">{cfg['expected_roi']}</span></div>
            <div><span style="color:#8b949e">Discount offer:</span>
                 <span style="color:#e0e0e0;margin-left:4px">{cfg['discount']}</span></div>
            <div><span style="color:#8b949e">Email subject:</span>
                 <span style="color:#e0e0e0;margin-left:4px;font-style:italic">"{cfg['email_subject']}"</span></div>
        </div>
        <div style="font-size:13px;color:#8b949e;margin-bottom:4px;font-weight:600;">Action checklist:</div>
        {''.join(f'<div style="font-size:13px;color:#c9d1d9;padding:2px 0">✅ {a}</div>' for a in actions)}
    </div>""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
if uploaded:
    raw_df = load_data(uploaded)
else:
    try:
        raw_df = pd.read_csv("online_retail_sample.csv")
    except:
        raw_df = None

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Business Understanding
# ════════════════════════════════════════════════════════════════════════════════
if page == "Business Understanding":
    st.title("Business Understanding")
    st.caption("Why customer segmentation matters")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Project Goal")
        st.write("""
        This project segments customers using the **RFM model** — a proven marketing framework.
        By grouping customers into segments, businesses can:
        - Identify their most valuable customers
        - Detect at-risk customers before they churn
        - Personalise campaigns for each group
        - Allocate marketing budget efficiently
        """)

        st.subheader("What is RFM?")
        st.table(pd.DataFrame({
            "Metric": ["Recency (R)", "Frequency (F)", "Monetary (M)"],
            "Definition": [
                "How recently did the customer purchase?",
                "How often do they purchase?",
                "How much do they spend in total?"
            ],
            "Business Insight": [
                "Recent buyers are more likely to buy again",
                "Frequent buyers are engaged & loyal",
                "High spenders are most valuable"
            ]
        }))

        st.subheader("Methodology — CRISP-DM")
        st.write("""
        1. **Business Understanding** — Define goals
        2. **Data Understanding** — Explore raw transactions
        3. **Data Preparation** — Clean data, compute RFM
        4. **Modeling & Evaluation** — K-Means + Elbow Method
        5. **Predict** — Classify new customers
        """)

    with col2:
        st.subheader("Segments")
        for seg, cfg in SEGMENT_CONFIG.items():
            st.markdown(f"""<div style="background:#1c2333;border-left:4px solid {cfg['color']};
                border-radius:8px;padding:10px 14px;margin-bottom:8px;">
                <b style="color:{cfg['color']}">{cfg['icon']} {seg}</b><br>
                <span style="font-size:13px;color:#8b949e">{cfg['desc']}</span>
            </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Data Understanding
# ════════════════════════════════════════════════════════════════════════════════
elif page == "Data Understanding":
    st.title("Data Understanding")
    st.caption("Explore the raw transaction dataset")

    if raw_df is None:
        st.warning("Please upload your CSV using the sidebar.")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows",       f"{len(raw_df):,}")
    c2.metric("Columns",          len(raw_df.columns))
    c3.metric("Unique Customers", f"{raw_df['CustomerID'].nunique():,}")
    date_col = raw_df["InvoiceDate"].astype(str)
    c4.metric("Date Range", f"{date_col.min()[:10]} → {date_col.max()[:10]}")

    st.markdown("---")
    st.subheader("Sample Data (first 20 rows)")
    st.dataframe(raw_df.head(20), use_container_width=True)

    st.subheader("Column Info")
    st.dataframe(pd.DataFrame({
        "Column":   raw_df.columns,
        "Non-Null": [raw_df[c].notna().sum() for c in raw_df.columns],
        "Null":     [raw_df[c].isna().sum()  for c in raw_df.columns],
        "Dtype":    [str(raw_df[c].dtype)    for c in raw_df.columns],
        "Sample":   [str(raw_df[c].dropna().iloc[0]) if raw_df[c].notna().any() else "" for c in raw_df.columns],
    }), use_container_width=True, hide_index=True)

    st.subheader("Monthly Revenue")
    tmp = raw_df.copy()
    tmp["InvoiceDate"] = pd.to_datetime(tmp["InvoiceDate"])
    tmp["Revenue"] = tmp["Quantity"] * tmp["UnitPrice"]
    monthly = tmp.groupby(tmp["InvoiceDate"].dt.to_period("M"))["Revenue"].sum().reset_index()
    monthly["InvoiceDate"] = monthly["InvoiceDate"].astype(str)
    fig = px.line(monthly, x="InvoiceDate", y="Revenue", template="plotly_dark")
    fig.update_traces(line_color="#e94560")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        top_c = tmp.groupby("Country")["Revenue"].sum().nlargest(8).reset_index()
        st.plotly_chart(px.bar(top_c, x="Revenue", y="Country", orientation="h",
            title="Top Countries by Revenue", template="plotly_dark",
            color="Revenue", color_continuous_scale="Reds"), use_container_width=True)
    with col2:
        top_p = tmp.groupby("Description")["Quantity"].sum().nlargest(8).reset_index()
        st.plotly_chart(px.bar(top_p, x="Quantity", y="Description", orientation="h",
            title="Top Products by Quantity", template="plotly_dark",
            color="Quantity", color_continuous_scale="Blues"), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Data Preparation
# ════════════════════════════════════════════════════════════════════════════════
elif page == "Data Preparation":
    st.title("Data Preparation")
    st.caption("Clean data and compute RFM features")

    if raw_df is None:
        st.warning("Please upload your CSV using the sidebar.")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Cleaning Steps")
        df2 = raw_df.copy()
        n0 = len(df2)
        df2 = df2[df2["Quantity"] > 0];  n1 = len(df2)
        df2 = df2[df2["UnitPrice"] > 0]; n2 = len(df2)
        df2 = df2.dropna(subset=["CustomerID"]); n3 = len(df2)
        st.info(f"Raw rows: **{n0:,}**")
        st.write(f"- Removed **{n0-n1}** rows where Quantity ≤ 0 (returns/cancellations)")
        st.write(f"- Removed **{n1-n2}** rows where UnitPrice ≤ 0")
        st.write(f"- Removed **{n2-n3}** rows with missing CustomerID")
        st.success(f"Clean rows: **{n3:,}**")

    with col2:
        st.subheader("RFM Formula")
        st.table(pd.DataFrame({
            "Feature": ["Recency", "Frequency", "Monetary"],
            "Formula": ["Days since last invoice", "Count of unique invoices", "Sum of Quantity × UnitPrice"],
        }))

    rfm = compute_rfm(raw_df)
    st.subheader("RFM Table (first 20 rows)")
    st.dataframe(rfm.head(20), use_container_width=True, hide_index=True)

    st.subheader("RFM Distributions")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.plotly_chart(px.histogram(rfm, x="Recency",   nbins=40, template="plotly_dark",
            color_discrete_sequence=["#e94560"], title="Recency"), use_container_width=True)
    with c2:
        st.plotly_chart(px.histogram(rfm, x="Frequency", nbins=40, template="plotly_dark",
            color_discrete_sequence=["#4fc3f7"], title="Frequency"), use_container_width=True)
    with c3:
        st.plotly_chart(px.histogram(rfm, x="Monetary",  nbins=40, template="plotly_dark",
            color_discrete_sequence=["#81c784"], title="Monetary"), use_container_width=True)

    st.subheader("RFM Statistics")
    st.dataframe(rfm[["Recency","Frequency","Monetary"]].describe().round(2), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Modeling & Evaluation
# ════════════════════════════════════════════════════════════════════════════════
elif page == "Modeling & Evaluation":
    st.title("Modeling & Evaluation")
    st.caption("K-Means clustering with Elbow Method")

    if raw_df is None:
        st.warning("Please upload your CSV using the sidebar.")
        st.stop()

    rfm = compute_rfm(raw_df)

    # Elbow chart
    st.subheader("Elbow Method — Choose Optimal K")
    ks, inertias = elbow_data(rfm)
    fig_e = go.Figure(go.Scatter(x=ks, y=inertias, mode="lines+markers",
        line=dict(color="#4fc3f7", width=2.5), marker=dict(size=7)))
    fig_e.update_layout(template="plotly_dark", height=380,
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Normalised Inertia (Total Squared Distance)")
    st.plotly_chart(fig_e, use_container_width=True)

    k = st.slider("Select number of clusters (k)", 2, 12, 6)
    st.info(f"You have chosen to segment into **{k} clusters**.")

    labels, scaler, model, sil = run_kmeans(rfm, k)
    rfm = rfm.copy()
    rfm["Cluster"] = labels

    rm, fm, mm = rfm["Recency"].median(), rfm["Frequency"].median(), rfm["Monetary"].median()
    rfm["Segment"] = rfm.apply(
        lambda r: seg_label(r.Recency, r.Frequency, r.Monetary, rm, fm, mm)[0], axis=1)
    rfm["MarketingAction"] = rfm["Segment"].map({
        seg: cfg["actions"][0] for seg, cfg in SEGMENT_CONFIG.items()
    })
    rfm["CampaignType"] = rfm["Segment"].map({
        seg: cfg["campaign_type"] for seg, cfg in SEGMENT_CONFIG.items()
    })
    rfm["Discount"]     = rfm["Segment"].map({
        seg: cfg["discount"] for seg, cfg in SEGMENT_CONFIG.items()
    })

    st.metric("Silhouette Score", f"{sil:.3f}", help="Closer to 1.0 = better clusters")

    # Cluster stats
    st.subheader("Cluster Statistics")
    summary = rfm.groupby("Cluster").agg(
        RecencyMean   =("Recency",    "mean"),
        FrequencyMean =("Frequency",  "mean"),
        MonetaryMean  =("Monetary",   "mean"),
        Count         =("CustomerID", "count")
    ).round(2).reset_index()
    summary["Percent"] = (summary["Count"] / summary["Count"].sum() * 100).round(2)
    summary.insert(0, "Cluster Label", summary["Cluster"].apply(lambda x: f"Cluster {x}"))
    st.dataframe(summary, use_container_width=True, hide_index=True)

    # Charts
    st.subheader("Segmentation Charts")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.scatter(rfm, x="Recency", y="Frequency", color="Segment",
            size="Monetary", size_max=18, opacity=0.75, template="plotly_dark",
            title="Recency vs Frequency",
            color_discrete_sequence=px.colors.qualitative.Vivid), use_container_width=True)
    with c2:
        st.plotly_chart(px.scatter(rfm, x="Frequency", y="Monetary", color="Segment",
            size="Recency", size_max=18, opacity=0.75, template="plotly_dark",
            title="Frequency vs Monetary",
            color_discrete_sequence=px.colors.qualitative.Vivid), use_container_width=True)

    seg_c = rfm["Segment"].value_counts().reset_index()
    seg_c.columns = ["Segment", "Count"]
    st.plotly_chart(px.pie(seg_c, names="Segment", values="Count", hole=0.4,
        template="plotly_dark", title="Segment Distribution",
        color_discrete_sequence=px.colors.qualitative.Vivid), use_container_width=True)

    st.subheader("3D RFM View")
    fig3d = px.scatter_3d(rfm, x="Recency", y="Frequency", z="Monetary",
        color="Segment", opacity=0.7, template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Vivid)
    fig3d.update_traces(marker=dict(size=3))
    st.plotly_chart(fig3d, use_container_width=True)

    # ── MARKETING RECOMMENDATIONS ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📣 Marketing Recommendations per Segment")
    st.caption("Full action plan, campaign type, discount offer and email subject for each segment")

    seg_filter = st.multiselect(
        "Filter segments to display",
        options=list(SEGMENT_CONFIG.keys()),
        default=list(SEGMENT_CONFIG.keys())
    )

    active_segs = rfm["Segment"].unique().tolist()
    col_a, col_b = st.columns(2)
    for i, seg in enumerate(seg_filter):
        count = len(rfm[rfm["Segment"] == seg])
        cfg   = SEGMENT_CONFIG[seg]
        card_html = f"""
        <div class="rec-card" style="background:#1c2333;border-color:{cfg['color']};">
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">
                <span style="font-size:17px;font-weight:700;color:{cfg['color']}">{cfg['icon']} {seg}</span>
                <span class="rec-tag" style="background:{cfg['color']}22;color:{cfg['color']};border:1px solid {cfg['color']}">
                    {cfg['priority']}
                </span>
            </div>
            <div style="font-size:12px;color:#8b949e;margin-bottom:8px">
                👥 {count} customers &nbsp;|&nbsp; {cfg['campaign_type']} &nbsp;|&nbsp; {cfg['discount']}
            </div>
            <div style="font-size:12px;color:#8b949e;margin-bottom:6px;font-style:italic">
                📧 "{cfg['email_subject']}"
            </div>
            <div style="font-size:12px;color:#8b949e;font-weight:600;margin-bottom:4px">Action checklist:</div>
            {''.join(f'<div style="font-size:12px;color:#c9d1d9;padding:1px 0">✅ {a}</div>' for a in cfg['actions'])}
            <div style="font-size:12px;color:#8b949e;margin-top:8px">
                📈 <b style="color:#e0e0e0">Expected ROI:</b> {cfg['expected_roi']}
            </div>
        </div>"""
        if i % 2 == 0:
            col_a.markdown(card_html, unsafe_allow_html=True)
        else:
            col_b.markdown(card_html, unsafe_allow_html=True)

    # ── EXPORT CSV ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📥 Export Segmented Customer List")

    export_cols = ["CustomerID","Recency","Frequency","Monetary","Cluster","Segment","CampaignType","Discount","MarketingAction"]
    export_df   = rfm[export_cols].copy()
    export_df.columns = ["CustomerID","Recency (days)","Frequency (orders)","Monetary ($)",
                          "Cluster","Segment","Campaign Type","Discount Offer","Recommended Action"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Customers",   f"{len(export_df):,}")
    c2.metric("Segments",          export_df["Segment"].nunique())
    c3.metric("Avg Monetary ($)",  f"${export_df['Monetary ($)'].mean():,.2f}")

    # Filter by segment before download
    export_seg_filter = st.multiselect(
        "Export specific segments only (leave empty = export all)",
        options=list(SEGMENT_CONFIG.keys()),
        default=[],
        key="export_seg_filter"
    )
    if export_seg_filter:
        export_df = export_df[export_df["Segment"].isin(export_seg_filter)]

    st.dataframe(export_df.head(20), use_container_width=True, hide_index=True)
    if len(export_df) > 20:
        st.caption(f"Showing first 20 of {len(export_df):,} rows — full list in downloaded CSV.")

    st.download_button(
        label=f"⬇️ Download Segmented Customers CSV ({len(export_df):,} rows)",
        data=to_csv_bytes(export_df),
        file_name="rfm_segmented_customers.csv",
        mime="text/csv",
        use_container_width=True,
        type="primary"
    )

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Predict
# ════════════════════════════════════════════════════════════════════════════════
elif page == "Predict":
    st.title("Predict Customer Segment")
    st.caption("Enter RFM values to find which segment a customer belongs to")

    if raw_df is None:
        st.warning("Please upload your CSV using the sidebar.")
        st.stop()

    rfm = compute_rfm(raw_df)
    k = st.sidebar.slider("Number of clusters (k)", 2, 12, 6)
    labels, scaler, model, sil = run_kmeans(rfm, k)
    rfm = rfm.copy()
    rfm["Cluster"] = labels
    rm, fm, mm = rfm["Recency"].median(), rfm["Frequency"].median(), rfm["Monetary"].median()

    st.subheader("Enter Customer RFM Values")
    c1, c2, c3 = st.columns(3)
    with c1:
        recency   = st.number_input("Recency (days since last purchase)", 1, 1000, 30)
    with c2:
        frequency = st.number_input("Frequency (number of orders)", 1, 500, 5)
    with c3:
        monetary  = st.number_input("Monetary (total spend $)", 1.0, 100000.0, 250.0, step=10.0)

    if st.button("Find Segment", type="primary", use_container_width=True):
        pt_scaled = scaler.transform([[recency, frequency, monetary]])
        cid       = model.predict(pt_scaled)[0]
        cpts      = rfm[rfm["Cluster"] == cid]
        avg_r, avg_f, avg_m = cpts["Recency"].mean(), cpts["Frequency"].mean(), cpts["Monetary"].mean()
        seg_name, color = seg_label(avg_r, avg_f, avg_m, rm, fm, mm)
        size = len(cpts)
        pct  = round(size / len(rfm) * 100, 1)
        cfg  = SEGMENT_CONFIG.get(seg_name, {})

        st.markdown("---")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Segment",      seg_name)
        mc2.metric("Cluster",      f"Cluster {cid}")
        mc3.metric("Cluster Size", f"{size} ({pct}%)")
        mc4.metric("Silhouette",   f"{sil:.3f}")

        # Full recommendation card
        st.markdown("### 📣 Marketing Recommendation")
        marketing_rec_card(seg_name)

        # Comparison chart
        st.subheader("Customer vs Cluster Average")
        cmp = pd.DataFrame({
            "Metric":        ["Recency (days)", "Frequency (orders)", "Monetary ($)"],
            "Your Customer": [recency, frequency, monetary],
            "Cluster Avg":   [round(avg_r,1), round(avg_f,1), round(avg_m,2)]
        })
        st.plotly_chart(px.bar(cmp, x="Metric", y=["Your Customer","Cluster Avg"],
            barmode="group", template="plotly_dark",
            color_discrete_sequence=["#e94560","#4fc3f7"]), use_container_width=True)

        # Export single customer card
        st.markdown("---")
        single_row = pd.DataFrame([{
            "CustomerID":         "New Customer",
            "Recency (days)":     recency,
            "Frequency (orders)": frequency,
            "Monetary ($)":       monetary,
            "Cluster":            cid,
            "Segment":            seg_name,
            "Campaign Type":      cfg.get("campaign_type",""),
            "Discount Offer":     cfg.get("discount",""),
            "Recommended Action": cfg.get("actions",[""])[0],
        }])
        st.download_button(
            label="⬇️ Download this customer's profile as CSV",
            data=to_csv_bytes(single_row),
            file_name=f"customer_profile_{seg_name.replace(' ','_')}.csv",
            mime="text/csv",
        )

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 6 — Email Campaign
# ════════════════════════════════════════════════════════════════════════════════
elif page == "Email Campaign":
    st.title("📧 Email Campaign")
    st.caption("Send personalised emails to customers by segment")

    if raw_df is None:
        st.warning("Please upload your CSV using the sidebar.")
        st.stop()

    # ── Build RFM + segments ──────────────────────────────────────────────────
    rfm = compute_rfm(raw_df)
    k   = st.sidebar.slider("Number of clusters (k)", 2, 12, 6)
    labels, scaler, model, sil = run_kmeans(rfm, k)
    rfm = rfm.copy()
    rfm["Cluster"] = labels
    rm, fm, mm = rfm["Recency"].median(), rfm["Frequency"].median(), rfm["Monetary"].median()
    rfm["Segment"] = rfm.apply(
        lambda r: seg_label(r.Recency, r.Frequency, r.Monetary, rm, fm, mm)[0], axis=1)

    # ── Pre-written email templates per segment ───────────────────────────────
    EMAIL_TEMPLATES = {
        "Champions": {
            "subject": "You're one of our VIP customers — exclusive offer inside!",
            "body": """Dear Valued Customer,

We are delighted to recognise you as one of our most valued Champions!

Your loyalty and continued support mean everything to us. As a token of our appreciation, we would like to offer you:

  ★ Early access to our newest product launches
  ★ An exclusive VIP discount on your next order
  ★ Priority customer support — just for you

You are the reason we do what we do. Thank you for being an amazing customer.

Warm regards,
The Customer Team"""
        },
        "Loyal Customers": {
            "subject": "We appreciate you — here's a special loyalty reward!",
            "body": """Dear Loyal Customer,

Thank you for your continued support and trust in us!

As one of our most loyal customers, we want to make sure you always feel valued. Here is what we have for you:

  ★ A 5-10% loyalty discount on your next purchase
  ★ Access to our exclusive bundle deals
  ★ Personalised product recommendations just for you

We truly appreciate your loyalty. We look forward to serving you again soon.

Warm regards,
The Customer Team"""
        },
        "Potential Loyalists": {
            "subject": "We think you'll love these picks — just for you!",
            "body": """Dear Customer,

Thank you for shopping with us recently!

We noticed you have great taste, and we think you are going to love what we have in store. Here is a special offer to welcome you back:

  ★ 10-15% off your next purchase
  ★ Free membership to our loyalty programme
  ★ Handpicked product recommendations based on your history

We would love to make you a regular — and we are going to make it worth your while!

Warm regards,
The Customer Team"""
        },
        "At Risk": {
            "subject": "We miss you! Here's 20% off — come back today",
            "body": """Dear Valued Customer,

We have noticed that it has been a while since your last visit, and we truly miss you!

We want to win you back with something special:

  ★ An exclusive 20-25% discount — just for you
  ★ Free shipping on your next order
  ★ New arrivals we think you will love

We would love to hear from you. If there is anything we can do better, please let us know — your feedback means the world to us.

We hope to see you again soon!

Warm regards,
The Customer Team"""
        },
        "Need Attention": {
            "subject": "Don't miss out — trending picks this week!",
            "body": """Dear Customer,

We have some exciting offers we do not want you to miss!

This week only, we are sharing our top trending products with a special incentive just for you:

  ★ 15% off — limited time only
  ★ Free gift with your next purchase
  ★ Our bestselling products, hand-picked for you

Hurry — these offers will not last long!

Warm regards,
The Customer Team"""
        },
        "Lost": {
            "subject": "Last chance — we want you back!",
            "body": """Dear Customer,

We have not seen you in a while, and we genuinely miss having you as a customer.

This is our last-chance offer especially for you:

  ★ A 30%+ exclusive discount — our biggest offer yet
  ★ Free shipping, no minimum order
  ★ We have improved a lot since your last visit!

If there is anything that made you leave, we would love the chance to make it right. 

We hope to welcome you back!

Warm regards,
The Customer Team"""
        },
    }

    st.markdown("---")

    # ── STEP 1: Select Segment ────────────────────────────────────────────────
    st.subheader("Step 1 — Select Customer Segment")
    available_segs = sorted(rfm["Segment"].unique().tolist())
    selected_seg   = st.selectbox("Choose segment to email", available_segs)
    seg_customers  = rfm[rfm["Segment"] == selected_seg]
    cfg            = SEGMENT_CONFIG.get(selected_seg, {})

    # Segment summary card
    st.markdown(f"""
    <div style="background:#1c2333;border-left:5px solid {cfg.get('color','#4fc3f7')};
                border-radius:10px;padding:12px 16px;margin:10px 0;">
        <b style="color:{cfg.get('color','#4fc3f7')};font-size:16px">
            {cfg.get('icon','')} {selected_seg}
        </b> &nbsp;
        <span style="color:#8b949e">
            {len(seg_customers)} customers &nbsp;|&nbsp;
            {cfg.get('campaign_type','')} &nbsp;|&nbsp;
            {cfg.get('discount','')}
        </span>
    </div>""", unsafe_allow_html=True)

    # ── STEP 2: Enter Customer Emails ─────────────────────────────────────────
    st.markdown("---")
    st.subheader("Step 2 — Enter Customer Emails")
    st.caption(f"Enter the email addresses of {selected_seg} customers (one per line)")

    # Show CustomerIDs for reference
    with st.expander(f"View {selected_seg} Customer IDs ({len(seg_customers)} customers)"):
        st.dataframe(
            seg_customers[["CustomerID","Recency","Frequency","Monetary","Segment"]],
            use_container_width=True, hide_index=True
        )

    email_input = st.text_area(
        "Customer email addresses (one per line)",
        placeholder="customer1@example.com\ncustomer2@example.com\ncustomer3@example.com",
        height=150
    )

    # Parse and validate emails
    raw_emails    = [e.strip() for e in email_input.strip().split("\n") if e.strip()]
    valid_emails  = [e for e in raw_emails if "@" in e and "." in e.split("@")[-1]]
    invalid_emails= [e for e in raw_emails if e not in valid_emails]

    if raw_emails:
        c1, c2, c3 = st.columns(3)
        c1.metric("Emails entered",  len(raw_emails))
        c2.metric("Valid emails",    len(valid_emails), delta=None)
        c3.metric("Invalid emails",  len(invalid_emails),
                  delta=f"-{len(invalid_emails)}" if invalid_emails else None,
                  delta_color="inverse")
        if invalid_emails:
            st.warning(f"Invalid email addresses (will be skipped): {', '.join(invalid_emails)}")

    # ── STEP 3: Compose Email ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Step 3 — Compose Email")

    tmpl = EMAIL_TEMPLATES.get(selected_seg, {
        "subject": f"Special offer for {selected_seg} customers",
        "body": f"Dear Customer,\n\nThank you for being a {selected_seg} customer.\n\nWarm regards,\nThe Team"
    })

    email_subject = st.text_input("Email Subject", value=tmpl["subject"])
    email_body    = st.text_area("Email Body (editable)", value=tmpl["body"], height=300)

    # Preview
    with st.expander("Preview Email"):
        st.markdown(f"""
        <div style="background:#1c2333;border-radius:10px;padding:20px;
                    font-family:Georgia,serif;line-height:1.7;">
            <div style="color:#8b949e;font-size:12px;margin-bottom:4px">
                <b>To:</b> {valid_emails[0] if valid_emails else 'customer@example.com'}
                {f' + {len(valid_emails)-1} more' if len(valid_emails)>1 else ''}
            </div>
            <div style="color:#8b949e;font-size:12px;margin-bottom:12px">
                <b>Subject:</b> {email_subject}
            </div>
            <hr style="border-color:#30363d;margin-bottom:12px">
            <div style="color:#c9d1d9;white-space:pre-line;font-size:14px">
                {email_body}
            </div>
        </div>""", unsafe_allow_html=True)

    # ── STEP 4: Gmail Setup ───────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Step 4 — Gmail Configuration")

    st.info("""**How to get Gmail App Password:**
1. Go to your Google Account → Security
2. Enable 2-Step Verification
3. Go to App Passwords → Select app: Mail → Generate
4. Copy the 16-character password and paste below""")

    col1, col2 = st.columns(2)
    with col1:
        sender_email = st.text_input("Your Gmail Address",
                                      placeholder="yourname@gmail.com")
    with col2:
        app_password = st.text_input("Gmail App Password (16 chars)",
                                      type="password",
                                      placeholder="xxxx xxxx xxxx xxxx")

    # ── STEP 5: Send Emails ───────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Step 5 — Send Campaign")

    # Summary before sending
    if valid_emails and sender_email and app_password and email_subject and email_body:
        st.markdown(f"""
        <div style="background:#0f3460;border-radius:10px;padding:14px 18px;">
            <b style="color:#4fc3f7">Campaign Summary</b><br>
            <span style="color:#c9d1d9">
            Segment: <b>{selected_seg}</b> &nbsp;|&nbsp;
            Recipients: <b>{len(valid_emails)}</b> &nbsp;|&nbsp;
            From: <b>{sender_email}</b>
            </span>
        </div>""", unsafe_allow_html=True)
        st.markdown("")

    send_btn = st.button(
        f"🚀 Send Email to {len(valid_emails)} {selected_seg} Customers",
        type="primary",
        use_container_width=True,
        disabled=not (valid_emails and sender_email and app_password and email_subject and email_body)
    )

    if send_btn:
        if not sender_email or not app_password:
            st.error("Please enter your Gmail address and App Password.")
        elif not valid_emails:
            st.error("Please enter at least one valid email address.")
        else:
            success_list = []
            failed_list  = []
            progress_bar = st.progress(0, text="Sending emails...")
            status_box   = st.empty()

            for i, recipient in enumerate(valid_emails):
                try:
                    # Build MIME email
                    msg = MIMEMultipart("alternative")
                    msg["Subject"] = email_subject
                    msg["From"]    = sender_email
                    msg["To"]      = recipient

                    # Plain text part
                    msg.attach(MIMEText(email_body, "plain"))

                    # HTML part — styled version
                    html_body = f"""
                    <html><body style="font-family:Georgia,serif;
                                       background:#f8f9fa;padding:30px;">
                    <div style="max-width:600px;margin:auto;background:#ffffff;
                                border-radius:12px;padding:30px;
                                border-top:5px solid {cfg.get('color','#2E75B6')};">
                        <h2 style="color:{cfg.get('color','#2E75B6')}">
                            {cfg.get('icon','')} {selected_seg} Customer
                        </h2>
                        <pre style="font-family:Georgia,serif;font-size:15px;
                                    line-height:1.7;white-space:pre-wrap;
                                    color:#333333">{email_body}</pre>
                        <hr style="border-color:#eeeeee;margin:20px 0">
                        <p style="font-size:12px;color:#999999">
                            This email was sent as part of our
                            {cfg.get('campaign_type','marketing')} campaign.
                        </p>
                    </div>
                    </body></html>"""
                    msg.attach(MIMEText(html_body, "html"))

                    # Send via Gmail SMTP
                    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                        server.login(sender_email, app_password.replace(" ", ""))
                        server.sendmail(sender_email, recipient, msg.as_string())

                    success_list.append(recipient)
                    status_box.success(f"✅ Sent to {recipient}")

                except smtplib.SMTPAuthenticationError:
                    st.error("Gmail authentication failed. Please check your email and App Password.")
                    break
                except Exception as e:
                    failed_list.append(recipient)
                    status_box.warning(f"⚠️ Failed: {recipient} — {str(e)}")

                progress_bar.progress(
                    (i + 1) / len(valid_emails),
                    text=f"Sending... {i+1}/{len(valid_emails)}"
                )

            # Final report
            progress_bar.empty()
            status_box.empty()
            st.markdown("---")
            st.subheader("Campaign Report")
            r1, r2, r3 = st.columns(3)
            r1.metric("Total Sent",   len(success_list))
            r2.metric("Failed",       len(failed_list))
            r3.metric("Success Rate", f"{round(len(success_list)/len(valid_emails)*100)}%")

            if success_list:
                st.success(f"Campaign sent successfully to {len(success_list)} {selected_seg} customers!")
                with st.expander("Successfully sent to"):
                    for e in success_list:
                        st.write(f"✅ {e}")

            if failed_list:
                with st.expander("Failed emails"):
                    for e in failed_list:
                        st.write(f"❌ {e}")

            # Download campaign report
            report_df = pd.DataFrame({
                "Email":  success_list + failed_list,
                "Status": ["Sent"] * len(success_list) + ["Failed"] * len(failed_list),
                "Segment": selected_seg,
                "Subject": email_subject,
            })
            st.download_button(
                label="⬇️ Download Campaign Report CSV",
                data=report_df.to_csv(index=False).encode("utf-8"),
                file_name=f"campaign_report_{selected_seg.replace(' ','_')}.csv",
                mime="text/csv"
            )
    else:
        if not valid_emails:
            st.info("Enter customer email addresses above to enable sending.")
        elif not sender_email or not app_password:
            st.info("Enter your Gmail credentials above to enable sending.")
