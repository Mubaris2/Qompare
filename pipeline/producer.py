import json
import time
import argparse
import pandas as pd
from kafka import KafkaProducer

def produce_stream(task, parquet_path, topic, rate_limit):
    producer = KafkaProducer(
        bootstrap_servers='localhost:9092',
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    df = pd.read_parquet(parquet_path)
    
    print(f"Starting producer for {task} on topic {topic}. Total rows: {len(df)}")
    
    for i, row in df.iterrows():
        payload = row.to_dict()
        payload['step'] = i
        payload['task'] = task
        payload['kafka_timestamp'] = time.time()
        
        producer.send(topic, value=payload)
        
        if rate_limit > 0:
            time.sleep(1.0 / rate_limit)
            
        if i > 0 and i % 1000 == 0:
            print(f"Sent {i} records...")
            
    producer.flush()
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True, type=str)
    parser.add_argument('--data', required=True, type=str)
    parser.add_argument('--topic', default='stream_data', type=str)
    parser.add_argument('--rate', default=0, type=int, help="Records per second. 0 is max speed.")
    args = parser.parse_args()
    
    produce_stream(args.task, args.data, args.topic, args.rate)