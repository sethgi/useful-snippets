import argparse
import tqdm
import rosbag

parser = argparse.ArgumentParser()
parser.add_argument("bags", nargs="+")
parser.add_argument("--out_path")
args = parser.parse_args()

bag_files = args.bags


out_bag = rosbag.Bag(args.out_path, 'w')

for bag_path in bag_files:
    with rosbag.Bag(bag_path) as bag:
        for topic, msg, ts in tqdm.tqdm(bag.read_messages(), total=bag.get_message_count()):
            out_bag.write(topic, msg, ts)

out_bag.close()
