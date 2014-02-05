#!/usr/bin/env python

import sys


def get_result_name(line):
    return line.split('/')[-1].strip()


def get_time_secs(line):
    return int(line.split()[-1])


def get_performances(file_path):
    measures = []
    this_measure = {}

    with open(file_path) as dataFile:

        for line in dataFile.readlines():
            if "** Output path" in line:
                this_measure["result_name"] = get_result_name(line)

            if "** Time required to denoise (seconds)" in line:
                this_measure["time_secs"] = get_time_secs(line)
                measures.append(this_measure)
                this_measure = {}

    return measures


def print_performances(performances):
    header  = ""
    header += "result"
    header += ", time(secs)"
    print(header)

    for performance in performances:
        row  = ""
        row += "{:s}".format(performance["result_name"])
        row += ", {:5d}".format(performance["time_secs"])

        print(row)


def main():
    filepath = sys.argv[1]
    performances = get_performances(filepath)
    print_performances(performances)

if __name__ == "__main__":
    main()
