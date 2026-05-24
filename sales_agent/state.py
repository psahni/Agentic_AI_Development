from typing import TypedDict, Annotated
from operator import add


class SalesState(TypedDict):
    raw_data: str
    metrics: dict
    anomalies: Annotated[list, add]
    insights: str
    report: str
    status: str
    error_message: str


def get_initial_state() -> SalesState:
    return SalesState(
        raw_data      = "",
        metrics       = {},
        anomalies     = [],
        insights      = "",
        report        = "",
        status        = "ok",
        error_message = ""
    )