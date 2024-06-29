import json


def get_annotations(annotations_fp: str):
    with open(annotations_fp, "r") as f:
        annotations = json.load(f)

    replace_str = "C:/Users/Levi/Desktop/quantitative-benchmark/test-set\\"
    with_str = "/playpen-storage/levlevi/nba-positions-videos-dataset/testing/quantitative-benchmark/assets/test-set/"
    keys = list(annotations.keys())
    for k in keys:
        new_key = k.replace(replace_str, with_str)
        if new_key != k:
            annotations[new_key] = annotations[k]
            del annotations[k]

    return annotations


def get_timestamps(timestamps_fp: str):
    with open(timestamps_fp, "r") as f:
        timestamps = json.load(f)

    replace_str = "/mnt/opr/"
    with_str = "/playpen-storage/"

    for k in list(timestamps.keys()):
        new_key = k.replace(replace_str, with_str)
        if new_key != k:
            timestamps[new_key] = timestamps[k]
            del timestamps[k]

    return timestamps
