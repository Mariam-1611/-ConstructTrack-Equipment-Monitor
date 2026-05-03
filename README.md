# 🏗️ ConstructTrack — Real-Time Equipment Utilization Monitor

> A production-grade, real-time computer vision pipeline that tracks construction equipment activity, classifies work states, and streams live analytics through Apache Kafka to an interactive dashboard.

---

## 📽️ Demo

> **Watch the live demo below — equipment detected, classified, and tracked in real time:**

<!-- Replace the link below with your actual screen recording link after uploading to YouTube/Google Drive -->
[![ConstructTrack Demo](https://img.shields.io/badge/▶%20Watch%20Demo-Click%20Here-red?style=for-the-badge&logo=youtube)]([YOUR_DEMO_VIDEO_LINK_HERE](https://drive.google.com/file/d/1dmuoG0wt7OKR9O7KXyb22ry8hdTXvDwi/view?usp=sharing))

## 🧠 What It Does

ConstructTrack processes video footage of construction sites and for each detected piece of equipment it:

- Classifies it as **ACTIVE** or **INACTIVE** in real time
- Detects **arm-only motion** (e.g. excavator digging while tracks are stationary)
- Classifies the **current activity**: `DIGGING`, `SWINGING`, `DUMPING`, or `WAITING`
- Tracks **total active time**, **idle time**, and **utilization percentage**
- Streams all results through **Apache Kafka** to a **live Streamlit dashboard**
- Persists all data to **TimescaleDB** (PostgreSQL with time-series extension)

---

## 🏛️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     docker-compose network                   │
│                                                              │
│  ┌──────────────┐    ┌─────────────┐    ┌────────────────┐  │
│  │  CV Service  │───▶│    Kafka    │───▶│   Dashboard    │  │
│  │              │    │   Broker    │    │   (Streamlit)  │  │
│  │ YOLO detect  │    │             │    │                │  │
│  │ Motion anal. │    │ topic:      │    │ Live badges    │  │
│  │ Activity cls │    │ equipment-  │    │ Gauge charts   │  │
│  │ Time tracker │    │ detections  │    │ Time bars      │  │
│  └──────────────┘    └─────────────┘    └───────┬────────┘  │
│                                                  │           │
│  ┌───────────┐   ┌────────────┐                 │           │
│  │ Zookeeper │   │ TimescaleDB│◀────────────────┘           │
│  │ (manages  │   │ PostgreSQL │                             │
│  │  Kafka)   │   │ time-series│                             │
│  └───────────┘   └────────────┘                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔬 Technical Deep Dive

### The Articulated Motion Problem (Key Innovation)

Standard motion detection fails for construction equipment because an excavator arm can be **actively digging while the tracks remain completely stationary**. A naive system would classify this as INACTIVE — which is wrong.

**Our solution: Region-Based Motion Analysis**

We split each detected bounding box into two regions:

```
┌─────────────────┐
│   UPPER REGION  │  ← ARM & BUCKET (optical flow here)
│   (top 50%)     │
├─────────────────┤
│   LOWER REGION  │  ← TRACKS & BODY (optical flow here)
│   (bottom 50%)  │
└─────────────────┘
```

Decision logic:
| Upper Region | Lower Region | Result |
|---|---|---|
| Moving | Moving | `ACTIVE` — `full_body` |
| Moving | Still | `ACTIVE` — `arm_only` ✅ key case |
| Still | Still | `INACTIVE` |

### Activity Classification (Optical Flow)

We use **Lucas-Kanade Optical Flow** on the upper region to track corner points between frames. By analyzing the average motion vector `(dx, dy)` over an 8-frame history window:

| Dominant Direction | Activity |
|---|---|
| `dy > 0` (downward) | `DIGGING` |
| `dy < 0` (upward) | `DUMPING` |
| `dx` dominant (sideways) | `SWINGING` |
| No significant movement | `WAITING` |

**Why optical flow instead of a trained classifier?**
No labeled activity dataset was available. Optical flow works purely from physics — no training required. This is mentioned as a key design trade-off.

### Kafka Payload Format

Every processed frame produces a JSON message to the `equipment-detections` topic:

```json
{
  "frame_id": 450,
  "equipment_id": "EX-001",
  "equipment_class": "excavator",
  "utilization": {
    "current_state": "ACTIVE",
    "current_activity": "DIGGING",
    "motion_source": "arm_only"
  },
  "time_analytics": {
    "total_tracked_seconds": 15.0,
    "total_active_seconds": 12.5,
    "total_idle_seconds": 2.5,
    "utilization_percent": 83.3
  }
}
```

---

## 🗂️ Project Structure

```
ConstructTracker-Equipment/
├── cv_service/
│   ├── main.py                 # Main pipeline — reads video, runs YOLO, sends to Kafka
│   ├── motion_analyzer.py      # Region-based motion analysis (ACTIVE/INACTIVE)
│   ├── activity_classifier.py  # Optical flow activity classification
│   ├── kafka_producer.py       # Kafka producer — sends JSON payloads
│   ├── Dockerfile
│   └── requirements.txt
├── dashboard/
│   ├── app.py                  # Streamlit live dashboard
│   ├── kafka_consumer.py       # Kafka consumer — reads from topic
│   ├── Dockerfile
│   └── requirements.txt
├── db/
│   └── init.sql                # TimescaleDB schema + hypertable setup
├── videos/                     # Place your video files here
├── docker-compose.yml
├── .env
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- [Anaconda](https://www.anaconda.com/) or Python 3.10+
- Git

### Step 1 — Clone the repository

```bash
git clone https://github.com/Mariam-1611/ConstructTrack.git
cd ConstructTrack
```

### Step 2 — Create conda environment

```bash
conda create -n track-equ python=3.10 -y
conda activate track-equ
pip install -r cv_service/requirements.txt
pip install -r dashboard/requirements.txt
```

### Step 3 — Add your video files

Place your construction site video files in the `videos/` folder:
```
videos/
├── video1.mp4
└── video2.mp4
```

### Step 4 — Start infrastructure (Kafka + PostgreSQL)

```bash
docker-compose up -d zookeeper kafka postgres
```

Wait ~30 seconds for all services to start, then verify:
```bash
docker-compose ps
```

### Step 5 — Run the CV Service

```bash
# Set video path
set VIDEO_PATH=videos/video1.mp4        # Windows
export VIDEO_PATH=videos/video1.mp4     # Linux/Mac

python cv_service/main.py
```

### Step 6 — Run the Dashboard

Open a new terminal:
```bash
conda activate eagle-vision
cd dashboard
streamlit run app.py
```

Open your browser at **http://localhost:8501** 🚀

---

## 🐳 Run Everything with Docker

To run the complete system including CV service and dashboard:

```bash
# Add your video to videos/ folder first, then:
docker-compose up --build
```

Open **http://localhost:8501** in your browser.

---

## 📊 Dashboard Features

- **🟢 ACTIVE / 🔴 INACTIVE** — Live badge per machine
- **Activity label** — DIGGING / SWINGING / DUMPING / WAITING
- **Utilization gauge** — Real-time percentage dial
- **Active vs Idle bar** — Visual time breakdown per machine
- **Equipment count** — Total tracked and currently active
- **Auto-refresh** — Updates every 2 seconds

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Object Detection | YOLOv8 (Ultralytics) |
| Motion Analysis | OpenCV Optical Flow |
| Message Broker | Apache Kafka + Zookeeper |
| Database | TimescaleDB (PostgreSQL) |
| Dashboard | Streamlit + Plotly |
| Containerization | Docker + Docker Compose |
| Language | Python 3.10 |

---

## 📝 Design Decisions & Trade-offs

**1. Optical Flow over Deep Learning for activity classification**
With no labeled dataset for construction equipment activities, optical flow gives reliable results based purely on motion physics. A trained classifier would require thousands of labeled frames.

**2. Region-based motion over whole-frame motion**
Splitting the bounding box into upper/lower regions was essential to correctly detect arm-only motion — the key challenge in this domain.

**3. Kafka over direct API calls**
Kafka decouples the CV service from the dashboard completely. The CV service can run at full video speed without waiting for the dashboard to consume data. This makes the system scalable — multiple consumers can read the same topic independently.

**4. TimescaleDB over plain PostgreSQL**
TimescaleDB's hypertables automatically partition time-series data for fast range queries — ideal for querying equipment utilization over time windows.

**5. YOLOv8n (nano) for prototype speed**
The nano model runs fast enough for near-real-time processing on CPU. For production, YOLOv8m or a fine-tuned model on construction equipment would improve detection accuracy.

---

## 👩‍💻 Author

**Mariam Goda**
Data Science & AI Student — Zewail City

[![GitHub](https://img.shields.io/badge/GitHub-Mariam--1611-black?logo=github)](https://github.com/Mariam-1611)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](YOUR_LINKEDIN_LINK_HERE) 
