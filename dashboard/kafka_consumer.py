import json
import logging
from kafka import KafkaConsumer
from kafka.errors import KafkaError

logger = logging.getLogger(__name__)


class EquipmentKafkaConsumer:
    """
    Reads detection results from Kafka and yields them one by one.

    This is the receiving end of the pipeline:
        CV Service (producer) → Kafka → Dashboard (consumer)

    We use 'latest' offset so the dashboard always shows
    the most recent data, not old messages from the past.
    """

    def __init__(self, bootstrap_servers='kafka:9092', topic='equipment-detections'):
        self.topic = topic
        self.consumer = None
        self.bootstrap_servers = bootstrap_servers
        self._connect()

    def _connect(self):
        try:
            self.consumer = KafkaConsumer(
                self.topic,
                bootstrap_servers=self.bootstrap_servers,

                # Start from latest message — we want live data
                auto_offset_reset='latest',

                # Deserialize JSON bytes → Python dict automatically
                value_deserializer=lambda v: json.loads(v.decode('utf-8')),

                # Consumer group — allows multiple consumers if needed
                group_id='dashboard-group',

                # Don't wait long for messages — keep dashboard responsive
                consumer_timeout_ms=1000
            )
            logger.info(f"Connected to Kafka at {self.bootstrap_servers}")

        except KafkaError as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            raise

    def get_messages(self, max_messages=50):
        """
        Reads up to max_messages from Kafka and returns them as a list.
        Returns empty list if no messages are available.
        """
        messages = []
        try:
            for msg in self.consumer:
                messages.append(msg.value)
                if len(messages) >= max_messages:
                    break
        except Exception as e:
            logger.debug(f"No messages or timeout: {e}")
        return messages

    def close(self):
        if self.consumer:
            self.consumer.close()
            logger.info("Kafka consumer closed.") 
