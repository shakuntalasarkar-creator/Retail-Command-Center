import streamlit as st
import pandas as pd
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Retail Intelligence Command Center",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Density Dark Theme Styling
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .metric-card {
        background-color: #1E222D;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #2B313E;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    </style>
""", unsafe_allow_html=True)

st.title("🏪 Retail Intelligence Command Center")
st.caption("AI-Powered Retail Operations & Volumetric Customer Dynamics")

# -----------------------------------------------------------------------------
# 1. Data Ingestion & Processing
# -----------------------------------------------------------------------------
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file, encoding='latin1', low_memory=False)
    else:
        df = pd.read_excel(uploaded_file)
    
    mapping = {
        'Mobile Number': 'customer_phone',
        'Bill Date': 'bill_date',
        'Bill No': 'bill_no',
        'Item Name': 'item_name',
        'qty': 'qty',
        'amount': 'amount',
        'Vertical': 'vertical'
    }
    df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})
    
    df['customer_phone'] = df['customer_phone'].astype(str).str.split('.').str[0]
    df = df[df['customer_phone'].str.len() >= 10]
    df['bill_date'] = pd.to_datetime(df['bill_date'], errors='coerce')
    df['qty'] = pd.to_numeric(df.get('qty', 1), errors='coerce').fillna(1)
    df['amount'] = pd.to_numeric(df.get('amount', 0), errors='coerce').fillna(0)
    
    return df.dropna(subset=['bill_date', 'customer_phone'])

# Sidebar File Uploader
st.sidebar.header("📥 Data Source")
uploaded_file = st.sidebar.file_uploader("Upload Sales Ledger (.csv / .xlsx)", type=["csv", "xlsx"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    max_date = df['bill_date'].max()

    # Dynamic Threshold Calculations
    cust_visits = df[['customer_phone', 'bill_no', 'bill_date']].drop_duplicates().sort_values(['customer_phone', 'bill_date'])
    cust_visits['prev_visit'] = cust_visits.groupby('customer_phone')['bill_date'].shift(1)
    cust_visits['gap_days'] = (cust_visits['bill_date'] - cust_visits['prev_visit']).dt.days

    all_gaps = cust_visits['gap_days'].dropna()
    reg_cutoff = int(np.percentile(all_gaps, 60)) if len(all_gaps) > 0 else 7
    dorm_cutoff = int(np.percentile(all_gaps, 90)) if len(all_gaps) > 0 else 25

    cust_summary = cust_visits.groupby('customer_phone').agg(
        last_visit=('bill_date', 'max'),
        total_visits=('bill_no', 'nunique'),
        avg_gap=('gap_days', 'mean')
    ).reset_index()

    cust_summary['days_inactive'] = (max_date - cust_summary['last_visit']).dt.days

    def assign_status(days):
        if days <= reg_cutoff: return 'Active Regular'
        elif days <= dorm_cutoff: return 'Slipping / Irregular'
        else: return 'Dormant / Lost'

    cust_summary['status'] = cust_summary['days_inactive'].apply(assign_status)

    # -----------------------------------------------------------------------------
    # 2. Top Macro Metrics
    # -----------------------------------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    active_cnt = (cust_summary['status'] == 'Active Regular').sum()
    slipping_cnt = (cust_summary['status'] == 'Slipping / Irregular').sum()
    dormant_cnt = (cust_summary['status'] == 'Dormant / Lost').sum()
    total_cust = len(cust_summary)

    with c1: st.metric("🟢 Active Regulars", f"{active_cnt}", f"{(active_cnt/total_cust)*100:.1f}%")
    with c2: st.metric("🟡 Slipping", f"{slipping_cnt}", f"{(slipping_cnt/total_cust)*100:.1f}%")
    with c3: st.metric("🔴 Dormant", f"{dormant_cnt}", f"{(dormant_cnt/total_cust)*100:.1f}%")
    with c4: st.metric("⚙️ Benchmark", "60th / 90th %", f"≤ {reg_cutoff}d cycle")

    st.markdown("---")

    # -----------------------------------------------------------------------------
    # 3. Interactive Master Table & Volumetric Views
    # -----------------------------------------------------------------------------
    tab1, tab2 = st.tabs(["📊 Command Center Analytics", "🤖 Talk to Store Agent Copilot"])

    with tab1:
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("🔄 Customer Lifecycle Breakdown")
            st.bar_chart(cust_summary['status'].value_counts())

        with col_right:
            st.subheader("🛒 Top Volumetric Categories")
            if 'vertical' in df.columns:
                v_counts = df.groupby('vertical')['qty'].sum().nlargest(5)
                st.bar_chart(v_counts)

        st.markdown("### 🚨 Master Action Queue")
        status_filter = st.multiselect("Filter Status:", ['Active Regular', 'Slipping / Irregular', 'Dormant / Lost'], default=['Slipping / Irregular'])
        filtered_df = cust_summary[cust_summary['status'].isin(status_filter)]
        st.dataframe(filtered_df.head(20), use_container_width=True)

    # -----------------------------------------------------------------------------
    # 4. Interactive AI Agent Chatbot
    # -----------------------------------------------------------------------------
    with tab2:
        st.subheader("💬 Ask Your Store AI Agent")
        st.write("Ask questions about your store metrics, dormant customer counts, or top product volume!")

        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your Retail Intelligence Copilot. Ask me anything about your loaded store data."}]

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ex: How many dormant customers do we have?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Basic rule-based AI agent responses grounded in the loaded dataset
            prompt_lower = prompt.lower()
            if "dormant" in prompt_lower:
                response = f"There are currently **{dormant_cnt} dormant customers** (lapsed past {dorm_cutoff} days), representing about {(dormant_cnt/total_cust)*100:.1f}% of your total customer base."
            elif "regular" in prompt_lower or "active" in prompt_lower:
                response = f"You have **{active_cnt} active regular customers** visiting within your auto-calculated {reg_cutoff}-day cycle."
            elif "slipping" in prompt_lower or "irregular" in prompt_lower:
                response = f"There are **{slipping_cnt} slipping customers** currently drifting between the regular and dormant windows."
            else:
                response = f"Based on your loaded data of {total_cust} unique customers, your store is operating on a 60th percentile visit cycle of {reg_cutoff} days. Feel free to ask about active, slipping, or dormant metrics!"

            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

else:
    st.warning("👈 Please upload your sales ledger (`.csv` or `.xlsx`) in the sidebar to initialize the Command Center and AI Agent.")
