import traceback
import pika
import os

from callback import callback

if __name__ == '__main__':
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(
            host=os.environ['RABBITMQ_SERVER'],
            credentials = pika.PlainCredentials(os.environ['RABBITMQ_USER'], os.environ['RABBITMQ_PASSWORD']),
            heartbeat=0
        )
    )

    channel = connection.channel()

    channel.queue_declare(queue=os.environ['RABBITMQ_QUEUE_NAME'], durable=True)

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=os.environ['RABBITMQ_QUEUE_NAME'], on_message_callback=callback)

    try:
        print('Starting worker.')
        channel.start_consuming()
    except:
        print(traceback.format_exc())
        channel.close()
        print('Channel closed.')