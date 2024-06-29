import re

from typing import List


def find_time_remaining_from_results(results: List[str]):
    """
    Matches any string showing a valid time remaining of 20 minutes or less
    assumes brodcasts use MM:SS for times > 1 minute, and SS.S for times < 1 minute
    """

    if results is None:
        return None
    time_remaining_regex = r"(20:00)|(0[0-9]?:[0-9][0-9](\.[0-9])?)|([1-9]:[0-5][0-9])|(1[0-9]:[0-5][0-9](\.[0-9])?)|([0-9]\.[0-9])|([1-5][0-9]\.[0-9])"
    for result in results:
        if result is None:
            continue
        result = result.replace(" ", "")
        match = re.match(time_remaining_regex, result)
        if match is not None and match[0] == result:
            return result
    return None


def convert_time_to_float(time_remaining):
    """
    Coverts valid time-remaining str
    to float value representation.
    Return None if time-remaining is invalid.

    Ex: '1:30' -> 90.
    """

    if time_remaining is None:
        return None
    minutes, seconds = 0.0, 0.0
    if ":" in time_remaining:
        time_arr = time_remaining.split(":")
        minutes = float(time_arr[0])
        seconds = float(time_arr[1])
    elif "." in time_remaining:
        seconds = float(time_remaining)
    else:
        return None
    return (60.0 * minutes) + seconds
