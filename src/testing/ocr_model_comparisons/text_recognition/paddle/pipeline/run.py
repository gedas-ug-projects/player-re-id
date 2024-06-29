import cProfile
import pstats
import io

import json
from pipeline import process_dir

def main():

    # test_set_fp = "/playpen-storage/levlevi/nba-positions-videos-dataset/testing/quantitative-benchmark/assets/test-set"

    test_set_fp = "/mnt/sun/levlevi/data-sources"
    timestamps_out_fp = "/playpen-storage/levlevi/nba-positions-videos-dataset/testing/quantitative-benchmark/data/nba_15_16_timestamps.json"
    timestamps = process_dir(test_set_fp, data_out_path=timestamps_out_fp)
    with open(timestamps_out_fp, "w") as f:
        json.dump(timestamps, f, indent=4)


def profile_main():
    pr = cProfile.Profile()
    pr.enable()
    main()
    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()

    with open('/playpen-storage/levlevi/nba-positions-videos-dataset/testing/quantitative-benchmark/profile_results.txt', 'w') as f:
        f.write(s.getvalue())
    print(s.getvalue())

if __name__ == "__main__":
    main()