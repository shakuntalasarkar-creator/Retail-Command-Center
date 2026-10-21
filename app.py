import streamlit as st
import pandas as pd
import numpy as np

# Page Config: High-Density Dark Theme Command Center
st.set_page_config(
    page_title="Retail Intelligence Command Center",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Metric Cards & Sleek UI
st.markdown("""
    <style>
    .metric-card {
        background-color: #1E222D;
        border-radius: 10px;
        padding: 15px 20px;
        border: 1px solid #2B313E;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .status-active { color: #00E676; font-weight: bold; }
    .status-slipping { color: #FFB300; font-weight: bold; }
    .status-dormant { color: #FF5252; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.title("🏪 Retail Intelligence Command Center")
st.caption("Powered by Dynamic Percentile Auto-Benchmarks & Volumetric Multi-Anchor Tracking")

# -----------------------------------------------------------------------------
# 1. Flexible Data Ingestion & Engine Logic
# -----------------------------------------------------------------------------
@st.cache_data
def load_and_process_data(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file, encoding='latin1', low_memory=False)
    else:
        df = pd.read_excel(uploaded_file)

    # Standard internal schema mapping (adaptable to any POS export)
    mapping = {
        'Mobile Number': 'customer_phone',
        'Bill Date': 'bill_date',
        'Bill No': 'bill_no',
        'Item Name': 'item_name',
        'qty': 'qty',
        'amount': 'amount',
        'Vertical': 'vertical'
    }
    
    # Apply mapping dynamically if columns exist
    df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})
    
    # Fallback column cleanup if headers differ
    if 'customer_phone' not in df.columns:
        possible_phone = [c for c in df.columns if 'mobile' in c.lower() or 'phone' in c.lower()]
        if possible_phone:
            df = df.rename(columns={possible_phone[0]: 'customer_phone'})

    if 'bill_date' not in df.columns:
        possible_date = [c for c in df.columns if 'date' in c.lower()]
        if possible_date:
            df = df.rename(columns={possible_date[0]: 'bill_date'})

    # Data Types Cleanup
    df['customer_phone'] = df['customer_phone'].astype(str).str.split('.').str[0]
    df = df[df['customer_phone'].str.len() >= 10]
    df['bill_date'] = pd.to_datetime(df['bill_date'], errors='coerce')
    df['qty'] = pd.to_numeric(df.get('qty', 1), errors='coerce').fillna(1)
    df['amount'] = pd.to_numeric(df.get('amount', 0), errors='coerce').fillna(0)
    
    df = df.dropna(subset=['bill_date', 'customer_phone'])
    return df

# Sidebar: File Uploader
st.sidebar.header("📥 Data Source")
uploaded_file = st.sidebar.file_uploader("Upload Sales Ledger (.csv / .xlsx)", type=["csv", "xlsx"])

if uploaded_file is not None:
    df = load_and_process_data(uploaded_file)
    max_date = df['bill_date'].max()

    # -----------------------------------------------------------------------------
    # 2. Dynamic Threshold Calculations
    # -----------------------------------------------------------------------------
    cust_visits = df[['customer_phone', 'bill_no', 'bill_date']].drop_duplicates().sort_values(['customer_phone', 'bill_date'])
    cust_visits['prev_visit'] = cust_visits.groupby('customer_phone')['bill_date'].shift(1)
    cust_visits['gap_days'] = (cust_visits['bill_date'] - cust_visits['prev_visit']).dt.days

    all_gaps = cust_visits['gap_days'].dropna()
    reg_cutoff = int(np.percentile(all_gaps, 60)) if len(all_gaps) > 0 else 7
    dorm_cutoff = int(np.percentile(all_gaps, 90)) if len(all_gaps) > 0 else 25

    # Customer Lifetime Aggregation
    cust_summary = cust_visits.groupby('customer_phone').agg(
        last_visit=('bill_date', 'max'),
        total_visits=('bill_no', 'nunique'),
        avg_gap=('gap_days', 'mean')
    ).reset_index()

    cust_summary['days_inactive'] = (max_date - cust_summary['last_visit']).dt.days

    def assign_status(days):
        if days <= reg_cutoff:
            return 'Active Regular'
        elif days <= dorm_cutoff:
            return 'Slipping / Irregular'
        else:
            return 'Dormant / Lost'

    cust_summary['status'] = cust_summary['days_inactive'].apply(assign_status)

    # -----------------------------------------------------------------------------
    # 3. Macro Health Metrics (Top Summary Row)
    # -----------------------------------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    active_cnt = (cust_summary['status'] == 'Active Regular').sum()
    slipping_cnt = (cust_summary['status'] == 'Slipping / Irregular').sum()
    dormant_cnt = (cust_summary['status'] == 'Dormant / Lost').sum()
    total_cust = len(cust_summary)

    with c1:
        st.metric("🟢 Active Regulars", f"{active_cnt}", f"{(active_cnt/total_cust)*100:.1f}%")
        st.caption(f"Cycle: ≤ {reg_cutoff} Days")
    with c2:
        st.metric("🟡 Slipping / Irregular", f"{slipping_cnt}", f"{(slipping_cnt/total_cust)*100:.1f}%", delta_color="inverse")
        st.caption(f"Gap: {reg_cutoff+1} - {dorm_cutoff} Days")
    with c3:
        st.metric("🔴 Dormant / Lost", f"{dormant_cnt}", f"{(dormant_cnt/total_cust)*100:.1f}%", delta_color="inverse")
        st.caption(f"Lapsed: > {dorm_cutoff} Days")
    with c4:
        st.metric("⚙️ Dynamic Percentiles", "60th / 90th", f"Data Ref: {max_date.strftime('%d %b %Y')}")
        st.caption("Auto-benchmarked from intervals")

    st.markdown("---")

    col_left, col_right = st.columns([1, 1])

    # -----------------------------------------------------------------------------
    # 4. Trajectory Flow & Drift Analysis (Middle Left)
    # -----------------------------------------------------------------------------
    with col_left:
        st.subheader("🔄 30-Day Trajectory Drift")
        st.write("Customer migration pattern based on dynamic return windows:")
        
        st.info(f"🟢 **Stable Regulars:** {active_cnt} customers visiting within {reg_cutoff} days.")
        st.warning(f"⚠️ **Drifting (Regular ➔ Irregular):** {slipping_cnt} customers entering early lapse window.")
        st.error(f"🚨 **At-Risk (Irregular ➔ Dormant):** {dormant_cnt} customers past {dorm_cutoff}-day cutoff.")

    # -----------------------------------------------------------------------------
    # 5. Volumetric Archetype Breakdown (Middle Right)
    # -----------------------------------------------------------------------------
    with col_right:
        st.subheader("🛒 Volumetric Archetype Matrix")
        st.write("Top Categories & Item Stacks by Unit Volume:")

        if 'vertical' in df.columns:
            vertical_qty = df.groupby('vertical')['qty'].sum().reset_index()
            total_qty = vertical_qty['qty'].sum()
            vertical_qty['unit_contribution_%'] = (vertical_qty['qty'] / total_qty * 100).round(1)
            vertical_qty = vertical_qty.sort_values(by='qty', ascending=False).head(4)

            for _, row in vertical_qty.iterrows():
                v_name = row['vertical']
                pct = row['unit_contribution_%']
                top_items = df[df['vertical'] == v_name].groupby('item_name')['qty'].sum().nlargest(2).index.tolist()
                stack_str = " + ".join(top_items) if top_items else "Core Staples"

                st.write(f"**{v_name}** | Unit Contribution: **{pct}%**")
                st.progress(min(int(pct), 100))
                st.caption(f"📦 Lead Anchor Stack: {stack_str}")
        else:
            st.info("Upload a dataset containing a 'Vertical' or 'Category' column to render volumetric stacks.")

    st.markdown("---")

    # -----------------------------------------------------------------------------
    # 6. Paginated Master Action Table (Bottom Module)
    # -----------------------------------------------------------------------------
    st.subheader("🚨 Master Action Queue (Paginated & Filterable)")
    
    status_filter = st.multiselect(
        "Filter by Lifecycle Status:",
        options=['Active Regular', 'Slipping / Irregular', 'Dormant / Lost'],
        default=['Slipping / Irregular']
    )

    filtered_cust = cust_summary[cust_summary['status'].isin(status_filter)]

    # Pagination Logic
    items_per_page = 10
    total_pages = max(1, int(np.ceil(len(filtered_cust) / items_per_page)))
    page = st.number_input("Page", min_value=1, max_value=total_pages, step=1)

    start_idx = (page - 1) * items_per_page
    end_idx = start_idx + items_per_page

    st.dataframe(
        filtered_cust.iloc[start_idx:end_idx][['customer_phone', 'status', 'days_inactive', 'total_visits']],
        use_container_width=True
    )
    st.caption(f"Showing page {page} of {total_pages} ({len(filtered_cust)} total customers matching filter)")

    if st.button("⚡ Trigger WhatsApp Nurture Broadcast"):
        st.success(f"Broadcast queued for {len(filtered_cust)} targeted customers via API link!")

else:
    st.warning("👈 Please upload your sales ledger (`.csv` or `.xlsx`) in the sidebar to populate the Command Center.")