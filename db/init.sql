 CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS equipment_detections (
    time        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    frame_id    INTEGER,
    equipment_id        VARCHAR(20),
    equipment_class     VARCHAR(50),
    current_state       VARCHAR(20),
    current_activity    VARCHAR(50),
    motion_source       VARCHAR(30),
    total_tracked_sec   FLOAT,
    total_active_sec    FLOAT,
    total_idle_sec      FLOAT,
    utilization_percent FLOAT
);

SELECT create_hypertable('equipment_detections', 'time', if_not_exists => TRUE);

