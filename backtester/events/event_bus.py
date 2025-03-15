"""
Event bus for event-based communication in the backtesting system.
"""

from typing import Dict, Any, Callable, List, Optional
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class EventBus:
    """
    Event bus for event-based communication.
    
    This class provides a simple event bus implementation for
    communication between modules and with a separate frontend.
    """
    
    def __init__(self, use_redis: bool = False, redis_url: Optional[str] = None):
        """
        Initialize the event bus.
        
        Args:
            use_redis: Whether to use Redis for event distribution
            redis_url: Redis URL if using Redis
        """
        self.subscribers = {}
        self.use_redis = use_redis
        self.redis_client = None
        
        if use_redis:
            try:
                import redis
                self.redis_client = redis.from_url(redis_url or "redis://localhost:6379/0")
                logger.info("Connected to Redis for event distribution")
            except (ImportError, Exception) as e:
                logger.warning(f"Failed to connect to Redis: {e}. Falling back to in-memory events.")
                self.use_redis = False
    
    def subscribe(self, event_type: str, callback: Callable[[Dict[str, Any]], None]):
        """
        Subscribe to an event type.
        
        Args:
            event_type: Type of event to subscribe to
            callback: Function to call when the event occurs
        """
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
        
        logger.debug(f"Subscribed to event type: {event_type}")
    
    def emit(self, event_type: str, data: Dict[str, Any]):
        """
        Emit an event.
        
        Args:
            event_type: Type of event to emit
            data: Data associated with the event
        """
        # Add timestamp to event data
        event_data = {
            "type": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        
        # Log the event
        logger.debug(f"Event emitted: {event_type}")
        
        # Notify in-memory subscribers
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                try:
                    callback(event_data)
                except Exception as e:
                    logger.error(f"Error in event subscriber: {e}")
        
        # Publish to Redis if enabled
        if self.use_redis and self.redis_client:
            try:
                self.redis_client.publish(
                    f"backtester:events:{event_type}",
                    json.dumps(event_data, default=str)
                )
            except Exception as e:
                logger.error(f"Error publishing event to Redis: {e}")
    
    def start_listening(self, event_types: List[str] = None):
        """
        Start listening for events from Redis.
        
        Args:
            event_types: List of event types to listen for (None for all)
        """
        if not self.use_redis or not self.redis_client:
            logger.warning("Redis not configured, cannot start listening")
            return
        
        try:
            import threading
            
            def listen_for_events():
                pubsub = self.redis_client.pubsub()
                
                # Subscribe to specified channels or all backtester events
                channels = [f"backtester:events:{event_type}" for event_type in event_types] if event_types else ["backtester:events:*"]
                pubsub.psubscribe(*channels)
                
                logger.info(f"Started listening for events on channels: {channels}")
                
                for message in pubsub.listen():
                    if message["type"] == "pmessage":
                        try:
                            event_data = json.loads(message["data"])
                            event_type = event_data["type"]
                            
                            # Notify subscribers
                            if event_type in self.subscribers:
                                for callback in self.subscribers[event_type]:
                                    callback(event_data)
                        except Exception as e:
                            logger.error(f"Error processing Redis event: {e}")
            
            # Start listening in a background thread
            thread = threading.Thread(target=listen_for_events, daemon=True)
            thread.start()
            
        except Exception as e:
            logger.error(f"Error starting Redis listener: {e}")
    
    def stop(self):
        """Stop the event bus and clean up resources."""
        if self.use_redis and self.redis_client:
            try:
                self.redis_client.close()
                logger.info("Closed Redis connection")
            except Exception as e:
                logger.error(f"Error closing Redis connection: {e}")
        
        # Clear subscribers
        self.subscribers = {} 