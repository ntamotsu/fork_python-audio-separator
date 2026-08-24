"""Helpers for preserving exception context in remote API responses."""


def format_remote_error(error):
    """Format an exception, batch members, and explicit causes without duplicates."""
    message = str(error)
    pending = [error]
    seen = set()

    while pending:
        current = pending.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))

        if current is not error:
            current_message = str(current)
            if current_message and current_message not in message:
                message = f"{message} (caused by: {current_message})"

        cause = getattr(current, "__cause__", None)
        if cause is not None:
            pending.append(cause)

        failures = getattr(current, "failures", ())
        failure_items = failures.items() if isinstance(failures, dict) else failures
        for _path, failure in failure_items:
            pending.append(failure)

    return message
