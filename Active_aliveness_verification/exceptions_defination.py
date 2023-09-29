"""
some user defined exceptions
"""


class Validation(Exception):
    """
    Our Custom Validation Exception
    Raise this when you want to return 400 validation error,
    and pass the internal code of the error, the code must be defined in errors.json
    """

    ...


class InternalServerError(Exception):
    """
    Our Custom Internal Server Error Exception
    Raise this when you want to return 500 internal server error,
    Try to not to use this until you exhausted all error handling because the return message of this is vague.
    """

    ...


class SomethingWrongHappened(Exception):
    """
    Our Custom Something Wrong Happened Error Exception
    Raise this when you want to return 449 insufficient information was provided.
    """

    ...
