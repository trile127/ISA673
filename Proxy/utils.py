#! /usr/bin/python

import ColorizePython

def colorizeLog(shouldColorize, log_level, msg):
    ## Higher is the log_level in the log() argument, the lower is its priority.
    colorize_log = {
        "NORMAL": ColorizePython.pycolors.ENDC,
        "WARNING": ColorizePython.pycolors.WARNING,
        "SUCCESS": ColorizePython.pycolors.OKGREEN,
        "FAIL": ColorizePython.pycolors.FAIL,
        "RESET": ColorizePython.pycolors.ENDC
    }

    if shouldColorize.lower() == "true":
        if log_level in colorize_log:
            return colorize_log[str(log_level)] + msg + colorize_log['RESET']
        return colorize_log["NORMAL"] + msg + colorize_log["RESET"]
    return msg
