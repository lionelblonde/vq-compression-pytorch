import time
from contextlib import contextmanager


def prettify_time(seconds):
    """Print the number of seconds in human-readable format.

    Examples:
        '2 days', '2 hours and 37 minutes', 'less than a minute'.
    """
    minutes = seconds // 60
    seconds %= 60
    hours = minutes // 60
    minutes %= 60
    days = hours // 24
    hours %= 24

    def helper(count, name):
        return "{} {}{}".format(str(count), name, ('s' if count > 1 else ''))

    # Display only the two greater units (days and hours, hours and minutes, minutes and seconds)
    if days > 0:
        message = helper(days, 'day')
        if hours > 0:
            message += ' and ' + helper(hours, 'hour')
        return message
    if hours > 0:
        message = helper(hours, 'hour')
        if minutes > 0:
            message += ' and ' + helper(minutes, 'minute')
        return message
    if minutes > 0:
        return helper(minutes, 'minute')
    # Finally, if none of the previous conditions is valid
    return 'less than a minute'


def colorize(string, color, bold=False, highlight=False):
    color2num = dict(gray=30, red=31, green=32, yellow=33, blue=34,
                     magenta=35, cyan=36, white=37, crimson=38)
    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


def log_module_info(logger, name, model):

    def _fmt(n):
        if n // 10 ** 6 > 0:
            return str(round(n / 10 ** 6, 2)) + ' M'
        elif n // 10 ** 3:
            return str(round(n / 10 ** 3, 2)) + ' k'
        else:
            return str(n)

    logger.info(4 * ">" + " logging {} specs".format(name))
    logger.info(model)
    num_params = [p.numel() for (_, p) in model.named_parameters() if p.requires_grad]
    logger.info("total trainable params: {}.".format(_fmt(sum(num_params))))


def timed_cm_wrapper(logger, use, color_message='yellow', color_elapsed_time='cyan'):
    """Wraps a context manager that records the time taken by encapsulated ops"""
    assert isinstance(use, bool)

    @contextmanager
    def _timed(message):
        """Display the time it took for the mpi master
        to perform the task within the context manager
        """
        if use:
            logger.info(colorize(">>>> {}".format(message).ljust(50, '.'), color=color_message))
            tstart = time.time()
            yield
            logger.info(colorize("[done in {:.3f} seconds]".format(time.time() - tstart).rjust(50, '.'),
                                 color=color_elapsed_time))
        else:
            yield
    return _timed


def log_epoch_info(logger, epochs_so_far, epochs, tstart):
    """Display the current epoch and elapsed time"""
    elapsed = prettify_time(time.time() - tstart)
    fmtstr = " epoch [{}/{}] | elapsed time: {}"
    logger.info(colorize(fmtstr.format(epochs_so_far, epochs, elapsed).rjust(75, '>'),
                         color='blue'))
