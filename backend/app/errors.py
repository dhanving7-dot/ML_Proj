"""Domain errors mapped to HTTP responses in ``main.py`` (no stack traces or paths are exposed)."""


class AppError(Exception):
    status_code = 400

    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


class NotFound(AppError):
    status_code = 404


class ModelUnavailable(AppError):
    status_code = 503


class JobConflict(AppError):
    status_code = 409


class PayloadTooLarge(AppError):
    status_code = 413


class InvalidInput(AppError):
    status_code = 422
