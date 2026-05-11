import time
from flwr.common import Context, Message, RecordDict
from flwr.server import ServerApp, Grid

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    print("--- Server running ---")

    node_ids = list(grid.get_node_ids())
    print(f"Server sees {len(node_ids)} clients.")

    if len(node_ids) < 4:
        print("Not enough clients.")
        return

    target_nodes = node_ids[:4]

    job_messages = [
        Message(
            content=RecordDict(),
            dst_node_id=node_id,
            message_type="train",
            group_id="0"
        )
        for node_id in target_nodes
    ]

    print(f"Sending requests to: {target_nodes}")
    outbound_message_ids = grid.push_messages(messages=job_messages)

    all_replies = []
    while len(all_replies) < len(target_nodes):
        fetched_replies = grid.pull_messages(message_ids=outbound_message_ids)
        all_replies.extend(fetched_replies)

        if len(all_replies) < len(target_nodes):
            received_ids = {reply.metadata.reply_to_message_id for reply in all_replies}
            outbound_message_ids = [m_id for m_id in outbound_message_ids if m_id not in received_ids]
            time.sleep(0.5)

    print(f"Success. All {len(all_replies)} replies received.")
    for reply in all_replies:
        print(f"--- Reply from {reply.metadata.src_node_id} ---")