import json
import logging
from kafka import KafkaProducer
from kafka.errors import KafkaError

logger = logging.getLogger(__name__)


class EquipmentKafkaProducer:
    """
    Handles sending detection results from the CV service into Kafka.

    Every time a frame is processed, we package the results as a JSON
    message and send it to the 'equipment-detections' topic.

    The dashboard (consumer) will pick these messages up in real-time.
    """

    def __init__(self, bootstrap_servers='kafka:9092', topic='equipment-detections'):
        self.topic = topic
        self.producer = None
        self.bootstrap_servers = bootstrap_servers
        self._connect()

    def _connect(self):
        """
        Connect to Kafka broker.
        Retries are handled by kafka-python automatically.
        """
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,

                # Serialize Python dict → JSON bytes automatically
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),

                # Wait up to 10 seconds for broker acknowledgment
                request_timeout_ms=10000,

                # Retry up to 3 times on failure
                retries=3
            )
            logger.info(f"Connected to Kafka at {self.bootstrap_servers}")

        except KafkaError as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            raise

    def send_detection(self,
                       frame_id,
                       equipment_id,
                       equipment_class,
                       current_state,
                       current_activity,
                       motion_source,
                       time_analytics):
        """
        Sends one detection result to Kafka.
        """
        payload = {
            "frame_id": frame_id,
            "equipment_id": equipment_id,
            "equipment_class": equipment_class,
            "utilization": {
                "current_state": current_state,
                "current_activity": current_activity,
                "motion_source": motion_source
            },
            "time_analytics": time_analytics
        }

        try:
            # Send to Kafka — non-blocking (fire and forget)
            future = self.producer.send(
                self.topic,
                key=equipment_id.encode('utf-8'),
                value=payload
            )

            # Optional: wait for confirmation (makes it blocking)

            logger.debug(f"Sent frame {frame_id} for {equipment_id} → {current_state} / {current_activity}")

        except KafkaError as e:
            logger.error(f"Failed to send message to Kafka: {e}")

    def flush(self):
        """
        Force send any buffered messages immediately.
        Call this at the end of processing or before shutdown.
        """
        if self.producer:
            self.producer.flush()

    def close(self):
        """Cleanly close the Kafka connection."""
        if self.producer:
            self.producer.flush()
            self.producer.close()
            logger.info("Kafka producer closed.") 
