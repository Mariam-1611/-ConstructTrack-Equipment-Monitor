import os
import time
import logging
import psycopg2
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from kafka_consumer import EquipmentKafkaConsumer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#  Configuration 
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:29092")
DB_HOST      = os.getenv("DB_HOST",      "localhost")
DB_USER      = os.getenv("DB_USER",      "eagle")
DB_PASSWORD  = os.getenv("DB_PASSWORD",  "eagle123")
DB_NAME      = os.getenv("DB_NAME",      "equipment_db")

#  Page Config 
st.set_page_config(
    page_title="ConstructTrack — Equipment Monitor",
    page_icon="🏗️",
    layout="wide"
)

#  Custom CSS 
st.markdown("""
<style>
    .active-badge {
        background-color: #00c853;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 14px;
    }
    .inactive-badge {
        background-color: #d50000;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 14px;
    }
    .metric-card {
        background-color: #1e1e2e;
        padding: 16px;
        border-radius: 10px;
        border: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)

#  Session State 
# Session state persists across Streamlit reruns
if "equipment_data" not in st.session_state:
    st.session_state.equipment_data = {}  # equipment_id → latest message

if "consumer" not in st.session_state:
    try:
        st.session_state.consumer = EquipmentKafkaConsumer(
            bootstrap_servers=KAFKA_BROKER
        )
    except Exception as e:
        st.session_state.consumer = None
        logger.error(f"Kafka connection failed: {e}")

#  Database Helper 
def save_to_db(msg):
    """Save a detection message to PostgreSQL."""
    try:
        conn = psycopg2.connect(
            host=DB_HOST, user=DB_USER,
            password=DB_PASSWORD, dbname=DB_NAME
        )
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO equipment_detections (
                frame_id, equipment_id, equipment_class,
                current_state, current_activity, motion_source,
                total_tracked_sec, total_active_sec,
                total_idle_sec, utilization_percent
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            msg.get("frame_id"),
            msg.get("equipment_id"),
            msg.get("equipment_class"),
            msg["utilization"]["current_state"],
            msg["utilization"]["current_activity"],
            msg["utilization"]["motion_source"],
            msg["time_analytics"]["total_tracked_seconds"],
            msg["time_analytics"]["total_active_seconds"],
            msg["time_analytics"]["total_idle_seconds"],
            msg["time_analytics"]["utilization_percent"]
        ))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"DB insert failed: {e}")


def get_history_from_db(equipment_id, limit=100):
    """Fetch recent history for an equipment from PostgreSQL."""
    try:
        conn = psycopg2.connect(
            host=DB_HOST, user=DB_USER,
            password=DB_PASSWORD, dbname=DB_NAME
        )
        df = pd.read_sql(f"""
            SELECT time, current_state, current_activity,
                   utilization_percent, total_active_sec, total_idle_sec
            FROM equipment_detections
            WHERE equipment_id = %s
            ORDER BY time DESC
            LIMIT %s
        """, conn, params=(equipment_id, limit))
        conn.close()
        return df
    except Exception as e:
        logger.error(f"DB query failed: {e}")
        return pd.DataFrame()


#  Poll Kafka 
def poll_kafka():
    """
    Read latest messages from Kafka and update session state.
    Called every time the dashboard refreshes.
    """
    if st.session_state.consumer is None:
        return

    messages = st.session_state.consumer.get_messages(max_messages=50)

    for msg in messages:
        eq_id = msg.get("equipment_id")
        if eq_id:
            # Keep only the latest message per equipment
            st.session_state.equipment_data[eq_id] = msg
            # Save to database
            save_to_db(msg)


#  UI Components 
def render_equipment_card(eq_id, data):
    """Renders a status card for one piece of equipment."""
    util    = data["utilization"]
    analytics = data["time_analytics"]

    state    = util["current_state"]
    activity = util["current_activity"]
    source   = util["motion_source"]

    badge_class = "active-badge" if state == "ACTIVE" else "inactive-badge"

    with st.container():
        st.markdown(f"### {eq_id} — {data.get('equipment_class','').upper()}")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(
                f'<span class="{badge_class}">{state}</span>',
                unsafe_allow_html=True
            )
            st.caption(f"Motion: {source}")

        with col2:
            st.metric("Activity", activity)

        with col3:
            st.metric(
                "Utilization",
                f"{analytics['utilization_percent']}%"
            )

        with col4:
            st.metric(
                "Active Time",
                f"{analytics['total_active_seconds']:.1f}s"
            )

        # Utilization gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=analytics["utilization_percent"],
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Utilization %"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#00c853" if state == "ACTIVE" else "#d50000"},
                "steps": [
                    {"range": [0, 40],  "color": "#2d2d2d"},
                    {"range": [40, 70], "color": "#1a1a2e"},
                    {"range": [70, 100],"color": "#0f3460"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 2},
                    "thickness": 0.75,
                    "value": analytics["utilization_percent"]
                }
            }
        ))
        fig.update_layout(
            height=220,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Time breakdown bar
        total = analytics["total_tracked_seconds"]
        if total > 0:
            active_pct = analytics["total_active_seconds"] / total
            idle_pct   = analytics["total_idle_seconds"]   / total

            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                name="Active",
                x=[active_pct * 100],
                y=[eq_id],
                orientation="h",
                marker_color="#00c853"
            ))
            fig2.add_trace(go.Bar(
                name="Idle",
                x=[idle_pct * 100],
                y=[eq_id],
                orientation="h",
                marker_color="#d50000"
            ))
            fig2.update_layout(
                barmode="stack",
                height=80,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                showlegend=True,
                xaxis=dict(range=[0, 100], showticklabels=False),
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.divider()


# Main Dashboard Layout 
def main():
    # Header
    st.title("🏗️ ConstructTrack — Equipment Monitor")
    st.caption("Real-time equipment utilization tracking powered by CV + Kafka")

    # Status bar
    col1, col2, col3 = st.columns(3)
    with col1:
        kafka_status = "🟢 Connected" if st.session_state.consumer else "🔴 Disconnected"
        st.metric("Kafka", kafka_status)
    with col2:
        eq_count = len(st.session_state.equipment_data)
        st.metric("Equipment Tracked", eq_count)
    with col3:
        active_count = sum(
            1 for d in st.session_state.equipment_data.values()
            if d["utilization"]["current_state"] == "ACTIVE"
        )
        st.metric("Currently Active", active_count)

    st.divider()

    # Poll Kafka for new messages
    poll_kafka()

    # Render equipment cards
    if not st.session_state.equipment_data:
        st.info("⏳ Waiting for detections... Make sure the CV service is running.")
    else:
        for eq_id, data in st.session_state.equipment_data.items():
            render_equipment_card(eq_id, data)

    # Auto-refresh every 2 seconds
    time.sleep(2)
    st.rerun()


if __name__ == "__main__":
    main()