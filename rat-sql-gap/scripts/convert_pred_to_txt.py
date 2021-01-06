import argparse
import json

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--input_file")
  parser.add_argument("--output_file")
  args = parser.parse_args()

  with open(args.input_file) as fin:
    data = json.load(fin)

  fout = open(args.output_file, "w")
  for item in data["per_item"]:
    fout.write(item["predicted"] + "\n")
