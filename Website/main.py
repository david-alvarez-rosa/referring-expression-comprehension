import argparse


parser = argparse.ArgumentParser(description="asdf")
parser.add_argument("--file", help="name of file to test. TODO")

args = parser.parse_args()

print(args.file)
print("hey there buddy")
