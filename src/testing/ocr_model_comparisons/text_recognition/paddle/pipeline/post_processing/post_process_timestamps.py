import numpy as np
import sys
import os


def update_timestamps(timestamps, time_remaining):
    for k, v in enumerate(time_remaining):
        timestamps[str(k)]["time_remaining"] = v
    return timestamps


def get_time_remaining_from_timestamps(timestamps):
    return np.array(
        [
            timestamps[k]["time_remaining"] if timestamps[k]["time_remaining"] is not None else None
            for k in timestamps
        ]
    )

def extend_timestamps(timestamps):
    """
    Interpolate timestamps in-place.
    """

    last_quarter, last_time = None, None

    for key in timestamps:
        quarter, time_remaining = (
            timestamps[key]["quarter"],
            timestamps[key]["time_remaining"],
        )
        if quarter:
            last_quarter = quarter
        else:
            timestamps[key]["quarter"] = last_quarter
        if time_remaining:
            last_time = time_remaining
        else:
            timestamps[key]["time_remaining"] = last_time

    return timestamps


def interpolate_time_remaining(time_remaining):

    fps = 30
    multiplier = 0
    decreasing = False
    for i in range(len(time_remaining) - 1):
        current, next_value = time_remaining[i], time_remaining[i + 1]
        if current == None or next_value == None:
            continue
        peak_value = time_remaining[min(i + fps, len(time_remaining) - 1)]
        if current == 0:
            continue
        decreasing = peak_value < current
        if decreasing:
            if multiplier > 30:
                multiplier, decreasing = 0, False
                continue
            time_remaining[i] -= round((1 / 30) * multiplier, 2)
            multiplier = 0 if next_value < current else multiplier + 1

    return time_remaining


def post_process_timestamps(timestamps):

    timestamps = timestamps.copy()

    def extend_timestamps(time_remaining):
        """
        Interpolate timestamps in-place.
        """
        _time_remaining = []
        last_time = 0
        for val in time_remaining:
            if val != None and val > 1:
                last_time = val
            _time_remaining.append(last_time)
        return _time_remaining

    def interpolate(time_remaining):

        time_remaining = time_remaining.copy()
        fps = 30
        multiplier = 0
        decreasing = False
        for i in range(len(time_remaining) - 1):
            current, next_value = time_remaining[i], time_remaining[i + 1]
            peak_value = time_remaining[min(i + fps, len(time_remaining) - 1)]
            if current == 0:
                continue
            decreasing = peak_value < current
            if decreasing:
                if multiplier > 30:
                    multiplier, decreasing = 0, False
                    continue
                time_remaining[i] -= round((1 / 30) * multiplier, 2)
                multiplier = 0 if next_value < current else multiplier + 1
        return time_remaining

    def moving_average(x, window):
        return np.convolve(x, np.ones(window), "valid") / window

    def normalize(arr):
        _min, _max = arr.min(), arr.max()
        return (arr - _min) / (_max - _min)

    def denoise_time_remaining(time_remaining):

        def update_time_remaining(remove_indices, time_remaining):
            valid_indices = np.where(remove_indices == 0)[0]
            for idx in np.where(remove_indices)[0]:
                nearest_valid_index = valid_indices[
                    np.argmin(np.abs(valid_indices - idx))
                ]
                time_remaining[idx] = time_remaining[nearest_valid_index]

        # remove values that deviate too far from expected values
        time_remaining = np.array(time_remaining)
        time_remaining_og = time_remaining.copy()
        expected = np.linspace(100, 720, len(time_remaining), endpoint=False)[::-1]
        norm_expected_diff = normalize(np.abs(expected - time_remaining_og))
        remove_indices = (norm_expected_diff > 0.5).astype(int)
        update_time_remaining(remove_indices, time_remaining)

        # convolve with shrinking window
        for window in [1000, 500]:
            if len(time_remaining) > window:
                mvg_avg = moving_average(time_remaining, window)
                padded_avg = np.pad(
                    mvg_avg, (window // 2, window - window // 2 - 1), mode="edge"
                )
                norm_diff = normalize(np.abs(time_remaining - padded_avg))
                remove_indices = (norm_diff > 0.5).astype(int)
                update_time_remaining(remove_indices, time_remaining)

        # convolve with shrinking window
        for window in [50, 10, 5]:
            if len(time_remaining) > window:
                mvg_avg = moving_average(time_remaining, window)
                padded_avg = np.pad(
                    mvg_avg, (window // 2, window - window // 2 - 1), mode="edge"
                )
                norm_diff = normalize(np.abs(time_remaining - padded_avg))
                remove_indices = (norm_diff > 0.5).astype(int)
                update_time_remaining(remove_indices, time_remaining)

        temp_interpolated = interpolate(time_remaining)
        delta = np.gradient(temp_interpolated)
        delta_inter = normalize(moving_average(abs(delta), 7))
        remove_indices = (delta_inter > 0.1).astype(int)
        update_time_remaining(remove_indices, time_remaining)
        return time_remaining

    def remove_delta_zero(a, b):
        if len(a) != len(b):
            raise ValueError("The arrays 'a' and 'b' must be of equal length.")
        # Iterate through the arrays
        for i in range(len(a)):
            if b[i] == 0:
                a[i] = None
        return a

    time_remaining = get_time_remaining_from_timestamps(timestamps)
    extended_time_remaining = extend_timestamps(time_remaining)
    denoised_time_remaining = denoise_time_remaining(extended_time_remaining)
    interpolated_time_remaining = interpolate(denoised_time_remaining)

    # remove values where delta = 0
    delta_time_remaining = np.gradient(interpolated_time_remaining)
    remove_delta_zero(interpolated_time_remaining, delta_time_remaining)

    timestamps = update_timestamps(
        timestamps=timestamps, time_remaining=interpolated_time_remaining
    )
    return timestamps
