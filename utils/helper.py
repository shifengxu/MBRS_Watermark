import time
import datetime

def log_info(*args):
    ds = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{ds}]", *args)

def get_time_ttl_and_eta(time_start, elapsed_iter, total_iter):
    """
    Get estimated total time and ETA time.
    :param time_start:
    :param elapsed_iter:
    :param total_iter:
    :return: string of elapsed time, string of ETA
    """

    def sec_to_str(sec):
        val = int(sec)  # seconds in int type
        s = val % 60
        val = val // 60  # minutes
        m = val % 60
        val = val // 60  # hours
        h = val % 24
        d = val // 24  # days
        return f"{d}-{h:02d}:{m:02d}:{s:02d}"

    elapsed_time = time.time() - time_start  # seconds elapsed
    elp = sec_to_str(elapsed_time)
    if elapsed_iter == 0:
        eta = 'NA'
    else:
        # seconds
        eta = elapsed_time * (total_iter - elapsed_iter) / elapsed_iter
        eta = sec_to_str(eta)
    return elp, eta
