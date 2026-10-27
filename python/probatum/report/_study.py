"""A search's or a validation's report: what it searched and found, as its summary says
it; each trial's or each path's score; a search's parameter importance, its
overfitting probability and, where a null was given, its deflated Sharpe; a
validation's windows in sample against out of sample, with their decay; and the looks
taken at the held-out bars (ADR 0041; Specification, Search; Validation; Across a
search)."""

from .._errors import ProbatumRuntimeError, ProbatumTypeError, ProbatumValueError
from . import _figures, _page


def page(study, null):
    """The report of ``study`` as one HTML page, its deflated Sharpe read against
    ``null`` where one is given."""
    figures = _page.Figures()
    said    = study.summary(null=null)

    body = f"<h1>{_page.escape(said.splitlines()[0])}</h1>\n"
    body += _page.section("Summary", _page.preformatted(said))

    if study.kind == "trials":
        body += _page.section("Trials", figures.held(_figures.trials, study))
        body += _page.section("Parameter importance",
                              _told(study.importance)
                              + (figures.held(_figures.importance, study)
                                 if _holds(study.importance) else ""))
        body += _page.section("Overfitting", _told(study.overfitting))
        if null is not None:
            body += _page.section("Deflated Sharpe",
                                  _told(lambda: study.deflated_sharpe(null)))
    else:
        body += _page.section("Paths", figures.held(_figures.paths, study))
        body += _page.section("In sample against out of sample",
                              figures.held(_figures.windows, study) + _told(study.decay)
                              + _told(study.consistency))

    looks = study.looks
    body += _page.section("The held-out bars", _page.paragraph(
        f"{looks} look{'s' * (looks != 1)} taken at the held-out bars"))

    return _page.document(said.splitlines()[0], body)


def _told(call):
    """What ``call`` gives, as its text, or why it gives nothing."""
    try:
        return _page.paragraph(str(call()))
    except (ProbatumRuntimeError, ProbatumTypeError, ProbatumValueError) as error:
        return _page.paragraph(f"none: {error}")


def _holds(call):
    try:
        call()
    except (ProbatumRuntimeError, ProbatumTypeError, ProbatumValueError):
        return False
    return True
