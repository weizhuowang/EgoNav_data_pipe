"""
Fix corrupted mcap files by rebuilding them, skipping bad chunks.

Uses StreamReader with emit_chunks=True to get raw Chunk records,
then manually decompresses each chunk with error handling.

Usage:
    python fix_mcap.py input.mcap output_fixed.mcap
"""

import os
import sys
from tqdm import tqdm
from mcap.stream_reader import StreamReader, breakup_chunk
from mcap.writer import Writer, CompressionType
from mcap.records import Chunk, Schema, Channel, Message, Header


def fix_mcap(input_path, output_path):
    file_size = os.path.getsize(input_path)
    print(f"Fixing: {input_path} ({file_size / 1e9:.2f} GB)")
    print(f"Output: {output_path}")

    count = 0
    skipped_chunks = 0
    t_origin = None  # first message time, for relative timestamps
    profile = ""

    # Read with emit_chunks=True so we get raw Chunk records
    # and can handle decompression errors ourselves
    with open(input_path, "rb") as f_in:
        stream_reader = StreamReader(f_in, emit_chunks=True)

        # ID mappings: old -> new
        schema_map = {}  # old_schema_id -> new_schema_id
        channel_map = {}  # old_channel_id -> new_channel_id
        channel_topics = {}  # old_channel_id -> topic name
        last_odom_time = None  # last t265 odom timestamp (nanoseconds)

        pbar = tqdm(
            total=file_size, unit="B", unit_scale=True, desc="Fixing", mininterval=0.1
        )

        with open(output_path, "wb") as f_out:
            writer = Writer(f_out, compression=CompressionType.LZ4)

            for record in stream_reader.records:
                # Update progress bar based on file position
                new_pos = f_in.tell()
                pbar.update(new_pos - pbar.n)

                if isinstance(record, Header):
                    profile = record.profile
                    writer.start(profile=profile, library=record.library)

                elif isinstance(record, Schema):
                    # Schema outside of chunks (rare but possible)
                    if record.id not in schema_map:
                        new_id = writer.register_schema(
                            record.name, record.encoding, record.data
                        )
                        schema_map[record.id] = new_id

                elif isinstance(record, Channel):
                    # Channel outside of chunks
                    channel_topics[record.id] = record.topic
                    if record.id not in channel_map:
                        new_schema_id = schema_map.get(record.schema_id, 0)
                        new_id = writer.register_channel(
                            record.topic,
                            record.message_encoding,
                            new_schema_id,
                        )
                        channel_map[record.id] = new_id

                elif isinstance(record, Message):
                    # Check t265 odom timestamp gap
                    topic = channel_topics.get(record.channel_id, "")
                    if "T265" in topic:
                        if last_odom_time is not None:
                            gap = (record.log_time - last_odom_time) / 1e9
                            if gap > 0.1:
                                rel_t = record.log_time / 1e9 - (t_origin or 0)
                                print(f"  [t265 gap] t={rel_t:.1f}s  gap={gap:.3f}s")
                        last_odom_time = record.log_time
                    # Message outside of chunks
                    if record.channel_id in channel_map:
                        writer.add_message(
                            channel_map[record.channel_id],
                            record.log_time,
                            record.data,
                            record.publish_time,
                            record.sequence,
                        )
                        count += 1

                elif isinstance(record, Chunk):
                    if t_origin is None:
                        t_origin = record.message_start_time / 1e9
                    # Try to decompress and extract records from chunk
                    try:
                        inner_records = breakup_chunk(record, validate_crc=False)
                    except Exception as e:
                        skipped_chunks += 1
                        rel_start = record.message_start_time / 1e9 - t_origin
                        rel_end = record.message_end_time / 1e9 - t_origin
                        duration = rel_end - rel_start
                        compressed_sz = len(record.data) if record.data else 0
                        print(
                            f"  [skip chunk #{skipped_chunks}] "
                            f"t={rel_start:.1f}s~{rel_end:.1f}s ({duration:.3f}s) "
                            f"size={compressed_sz / 1024:.0f}KB"
                        )
                        continue

                    for inner in inner_records:
                        if isinstance(inner, Schema):
                            if inner.id not in schema_map:
                                new_id = writer.register_schema(
                                    inner.name, inner.encoding, inner.data
                                )
                                schema_map[inner.id] = new_id

                        elif isinstance(inner, Channel):
                            channel_topics[inner.id] = inner.topic
                            if inner.id not in channel_map:
                                new_schema_id = schema_map.get(inner.schema_id, 0)
                                new_id = writer.register_channel(
                                    inner.topic,
                                    inner.message_encoding,
                                    new_schema_id,
                                )
                                channel_map[inner.id] = new_id

                        elif isinstance(inner, Message):
                            topic = channel_topics.get(inner.channel_id, "")
                            if "T265" in topic:
                                if last_odom_time is not None:
                                    gap = (inner.log_time - last_odom_time) / 1e9
                                    if gap > 0.1:
                                        rel_t = inner.log_time / 1e9 - (t_origin or 0)
                                        print(f"  [t265 gap] t={rel_t:.1f}s  gap={gap:.3f}s")
                                last_odom_time = inner.log_time
                            if inner.channel_id in channel_map:
                                writer.add_message(
                                    channel_map[inner.channel_id],
                                    inner.log_time,
                                    inner.data,
                                    inner.publish_time,
                                    inner.sequence,
                                )
                                count += 1

            writer.finish()

        pbar.n = file_size
        pbar.refresh()
        pbar.close()

    print(f"Done! Messages saved: {count}, Chunks skipped: {skipped_chunks}")
    return count, skipped_chunks


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python fix_mcap.py input.mcap output_fixed.mcap")
        sys.exit(1)
    fix_mcap(sys.argv[1], sys.argv[2])
