"""Stable public diagnostics and validation reports."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Severity(StrEnum):
    """Diagnostic severity."""

    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """A machine-readable diagnostic with optional repair guidance."""

    code: str
    severity: Severity
    pointer: str
    message: str
    suggestion: str | None = None


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """The complete public result of validating one input or conversion phase."""

    diagnostics: tuple[Diagnostic, ...] = ()

    @property
    def is_valid(self) -> bool:
        """Whether the report contains no errors."""

        return not self.errors

    @property
    def errors(self) -> tuple[Diagnostic, ...]:
        """Error diagnostics in deterministic order."""

        return tuple(item for item in self.diagnostics if item.severity is Severity.ERROR)

    @property
    def warnings(self) -> tuple[Diagnostic, ...]:
        """Warning diagnostics in deterministic order."""

        return tuple(item for item in self.diagnostics if item.severity is Severity.WARNING)


class PygucError(Exception):
    """A conversion failure described by a structured validation report."""

    def __init__(self, message: str, report: ValidationReport) -> None:
        super().__init__(message)
        self.report = report


class DiagnosticBag:
    """Bounded collector that never lets warnings hide input errors."""

    __slots__ = (
        "_error_limit",
        "_errors",
        "_errors_truncated",
        "_warning_limit",
        "_warnings",
        "_warnings_truncated",
    )

    def __init__(self, *, error_limit: int = 100, warning_limit: int = 100) -> None:
        self._errors: list[Diagnostic] = []
        self._warnings: list[Diagnostic] = []
        self._error_limit = error_limit
        self._warning_limit = warning_limit
        self._errors_truncated = False
        self._warnings_truncated = False

    @property
    def report(self) -> ValidationReport:
        """Build an immutable, deterministically ordered report."""

        items = [*self._errors, *self._warnings]
        if self._errors_truncated:
            items.append(
                Diagnostic(
                    code="PG001",
                    severity=Severity.ERROR,
                    pointer="",
                    message=f"error diagnostic limit of {self._error_limit} reached",
                    suggestion="Fix the reported errors, then validate the asset again.",
                )
            )
        if self._warnings_truncated:
            items.append(
                Diagnostic(
                    code="PG002",
                    severity=Severity.WARNING,
                    pointer="",
                    message=f"warning diagnostic limit of {self._warning_limit} reached",
                    suggestion="Review the reported warnings, then validate the asset again.",
                )
            )
        return ValidationReport(tuple(sorted(items, key=_diagnostic_sort_key)))

    @property
    def errors(self) -> tuple[Diagnostic, ...]:
        return self.report.errors

    @property
    def warnings(self) -> tuple[Diagnostic, ...]:
        return self.report.warnings

    def error(
        self,
        code: str,
        message: str,
        pointer: str = "",
        suggestion: str | None = None,
    ) -> None:
        if len(self._errors) < self._error_limit:
            self._errors.append(Diagnostic(code, Severity.ERROR, pointer, message, suggestion))
        else:
            self._errors_truncated = True

    def warning(
        self,
        code: str,
        message: str,
        pointer: str = "",
        suggestion: str | None = None,
    ) -> None:
        if len(self._warnings) < self._warning_limit:
            self._warnings.append(Diagnostic(code, Severity.WARNING, pointer, message, suggestion))
        else:
            self._warnings_truncated = True

    def extend(self, diagnostics: tuple[Diagnostic, ...]) -> None:
        """Merge diagnostics while preserving the independent caps."""

        for diagnostic in diagnostics:
            if diagnostic.severity is Severity.ERROR:
                self.error(
                    diagnostic.code,
                    diagnostic.message,
                    diagnostic.pointer,
                    diagnostic.suggestion,
                )
            else:
                self.warning(
                    diagnostic.code,
                    diagnostic.message,
                    diagnostic.pointer,
                    diagnostic.suggestion,
                )

    def raise_if_errors(self, message: str = "glTF validation failed") -> None:
        report = self.report
        if not report.is_valid:
            raise PygucError(message, report)


def merge_reports(*reports: ValidationReport) -> ValidationReport:
    """Combine reports using the same deterministic ordering and caps."""

    bag = DiagnosticBag()
    for report in reports:
        bag.extend(report.diagnostics)
    return bag.report


def error_report(
    code: str,
    message: str,
    *,
    pointer: str = "",
    suggestion: str | None = None,
) -> ValidationReport:
    """Create a one-error report for a failed conversion phase."""

    return ValidationReport((Diagnostic(code, Severity.ERROR, pointer, message, suggestion),))


def _diagnostic_sort_key(diagnostic: Diagnostic) -> tuple[int, str, str, str]:
    severity = 0 if diagnostic.severity is Severity.ERROR else 1
    return severity, diagnostic.pointer, diagnostic.code, diagnostic.message
