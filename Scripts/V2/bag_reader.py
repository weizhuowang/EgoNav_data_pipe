"""
Bag Reader - Unified reader for ROS1 bag and ROS2 mcap formats
"""

import os
from abc import ABC, abstractmethod
from typing import Iterator, Tuple, Any, List, Dict, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class TopicInfo:
    """Topic information"""
    name: str
    msg_type: str
    msg_count: int


class BaseBagReader(ABC):
    """Base class for bag readers"""

    def __init__(self, bag_path: str):
        self.bag_path = bag_path
        if not os.path.exists(bag_path):
            raise FileNotFoundError(f"Bag file not found: {bag_path}")

    @abstractmethod
    def read_messages(self, topics: List[str] = None) -> Iterator[Tuple[str, Any, float]]:
        """
        Read messages from bag

        Yields:
            (topic, msg, timestamp) - topic name, message object, header.stamp timestamp
        """
        pass

    @abstractmethod
    def get_topic_info(self) -> Dict[str, TopicInfo]:
        """Get all topic information"""
        pass

    @abstractmethod
    def get_message_count(self) -> int:
        """Get total message count"""
        pass

    @abstractmethod
    def close(self):
        """Close bag file"""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class ROS1BagReader(BaseBagReader):
    """ROS1 bag file reader"""

    def __init__(self, bag_path: str):
        super().__init__(bag_path)
        try:
            import rosbag
            import yaml
        except ImportError:
            raise ImportError("rosbag not found. Install with: pip install rosbag")

        self.bag = rosbag.Bag(bag_path)
        self._info_dict = yaml.safe_load(self.bag._get_yaml_info())

    def read_messages(self, topics: List[str] = None) -> Iterator[Tuple[str, Any, float]]:
        """
        Read messages using header.stamp as timestamp (lag compensate)
        """
        for topic, msg, t in self.bag.read_messages(topics=topics):
            # Use header.stamp if available, otherwise use bag time
            if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
                timestamp = msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9
            else:
                timestamp = t.to_sec()
            yield topic, msg, timestamp

    def get_topic_info(self) -> Dict[str, TopicInfo]:
        """Get all topic information"""
        info = {}
        topics = self._info_dict.get('topics', [])
        for t in topics:
            info[t['topic']] = TopicInfo(
                name=t['topic'],
                msg_type=t['type'],
                msg_count=t['messages']
            )
        return info

    def get_message_count(self) -> int:
        """Get total message count"""
        return self.bag.get_message_count()

    def close(self):
        """Close bag file"""
        self.bag.close()


class McapBagReader(BaseBagReader):
    """ROS2 mcap file reader"""

    def __init__(self, bag_path: str):
        # mcap can be a directory or single file
        if os.path.isdir(bag_path):
            # Find .mcap file
            mcap_files = [f for f in os.listdir(bag_path) if f.endswith('.mcap')]
            if not mcap_files:
                raise FileNotFoundError(f"No .mcap files found in {bag_path}")
            bag_path = os.path.join(bag_path, mcap_files[0])

        super().__init__(bag_path)

        try:
            from mcap.reader import make_reader
            from mcap_ros2.reader import read_ros2_messages
        except ImportError:
            raise ImportError(
                "mcap not found. Install with: pip install mcap mcap-ros2-support"
            )

        self._mcap_path = bag_path
        self._reader = None
        self._topic_info = None

    def _get_reader(self):
        """Get mcap reader"""
        from mcap.reader import make_reader
        if self._reader is None:
            self._reader = make_reader(open(self._mcap_path, 'rb'))
        return self._reader

    def read_messages(self, topics: List[str] = None) -> Iterator[Tuple[str, Any, float]]:
        """Read messages"""
        from mcap_ros2.reader import read_ros2_messages

        for msg in read_ros2_messages(self._mcap_path, topics=topics):
            topic = msg.channel.topic
            ros_msg = msg.ros_msg

            # Get timestamp
            if hasattr(ros_msg, 'header') and hasattr(ros_msg.header, 'stamp'):
                stamp = ros_msg.header.stamp
                timestamp = stamp.sec + stamp.nanosec * 1e-9
            else:
                # Use log_time (nanoseconds)
                timestamp = msg.log_time * 1e-9

            yield topic, ros_msg, timestamp

    def get_topic_info(self) -> Dict[str, TopicInfo]:
        """Get all topic information"""
        if self._topic_info is not None:
            return self._topic_info

        reader = self._get_reader()
        summary = reader.get_summary()

        self._topic_info = {}
        for channel_id, channel in summary.channels.items():
            schema = summary.schemas.get(channel.schema_id)
            msg_count = summary.statistics.channel_message_counts.get(channel_id, 0)
            self._topic_info[channel.topic] = TopicInfo(
                name=channel.topic,
                msg_type=schema.name if schema else 'unknown',
                msg_count=msg_count
            )

        return self._topic_info

    def get_message_count(self) -> int:
        """Get total message count"""
        reader = self._get_reader()
        summary = reader.get_summary()
        return sum(summary.statistics.channel_message_counts.values())

    def close(self):
        """Close mcap file"""
        if self._reader is not None:
            self._reader = None


class BagReader:
    """
    Unified bag reader interface

    Auto-detects file format and uses the appropriate reader
    """

    def __init__(self, bag_path: str):
        self.bag_path = bag_path
        self.format = self.detect_format(bag_path)

        if self.format == 'ros1':
            self._reader = ROS1BagReader(bag_path)
        elif self.format == 'mcap':
            self._reader = McapBagReader(bag_path)
        else:
            raise ValueError(f"Unknown bag format: {bag_path}")

    @staticmethod
    def detect_format(path: str) -> str:
        """
        Detect bag file format

        Returns:
            'ros1' or 'mcap'
        """
        if os.path.isdir(path):
            # Check directory contents
            files = os.listdir(path)
            if any(f.endswith('.mcap') for f in files):
                return 'mcap'
            if any(f.endswith('.db3') for f in files):
                return 'mcap'  # db3 is also ROS2 format

        # Check file extension
        _, ext = os.path.splitext(path)
        if ext == '.mcap':
            return 'mcap'
        if ext == '.bag':
            return 'ros1'
        if ext == '.db3':
            return 'mcap'

        # Try reading file header
        if os.path.isfile(path):
            with open(path, 'rb') as f:
                header = f.read(10)
                # mcap files start with magic bytes
                if header.startswith(b'\x89MCAP'):
                    return 'mcap'
                # ROS1 bag files start with #ROSBAG
                if header.startswith(b'#ROSBAG'):
                    return 'ros1'

        raise ValueError(f"Cannot detect bag format: {path}")

    def read_messages(self, topics: List[str] = None) -> Iterator[Tuple[str, Any, float]]:
        """Read messages"""
        return self._reader.read_messages(topics)

    def get_topic_info(self) -> Dict[str, TopicInfo]:
        """Get topic information"""
        return self._reader.get_topic_info()

    def get_message_count(self) -> int:
        """Get total message count"""
        return self._reader.get_message_count()

    def close(self):
        """Close file"""
        self._reader.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
