import argparse
import glob
import re
import os

import tensorflow as tf
import dateutil.parser


date_re = r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}'
log_re = '\[(' + date_re + ')\] Step (\d+) stats, ([^:]+): loss = ([\d\.]+)'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs', nargs='+')
    args = parser.parse_args()

    for path in args.inputs:
        for existing in glob.glob(os.path.join(path, 'events.out.tfevents*')):
            os.unlink(existing)
        writer = tf.summary.FileWriter(path)
        for line in open(os.path.join(path, 'log.txt')):
            m = re.search(log_re, line)
            if m is None:
                continue
            timestamp, step, section, loss = m.groups()
            step = int(step)
            loss = float(loss)
            timestamp = dateutil.parser.parse(timestamp).timestamp()

            writer.add_event(
                tf.Event(
                  wall_time=timestamp,
                  step=step,
                  summary=tf.Summary(
                          value=[
                              tf.Summary.Value(
                                tag='loss/{}'.format(section),
                                simple_value=loss)])))

        writer.close()
        print(path)


if __name__ == '__main__':
    main()
