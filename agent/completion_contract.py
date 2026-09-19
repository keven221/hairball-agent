"""Hairball-native completion contracts and direct-evidence assessment.

This module is deliberately independent of the conversation loops.  The loops
only adapt real turn/tool facts into :class:`CompletionContractLedger`; parsing,
matching, and serializable snapshots live here so product surfaces cannot
silently grow different definitions of evidence.

It is passive observability: it never changes model input, chooses a tool,
blocks a stop, or schedules another turn. Cancellation, budgets and provider
failures are terminal facts, never prompts for an extra model turn.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Optional, Sequence


_PARSER_VERSION = 4
_MAX_CLAUSES = 64
_MAX_EVIDENCE_SUMMARY = 1_200
_MAX_PROMPT_EVIDENCE = 240
_HEADING_RE = re.compile(r"^\s{0,3}(?:#{1,6}\s+|\*{1,2})(.+?)(?:\*{1,2})?\s*$")
_PLAIN_HEADING_RE = re.compile(r"^\s*([^:：\n]{1,80})\s*[:：]\s*$")
_FENCE_RE = re.compile(r"^(`{3,}|~{3,})")
_LIST_RE = re.compile(
    r"^\s*(?:"
    r"[-+*•]\s+(?:\[(?: |x|X|-)\]\s*)?"
    r"|\[(?: |x|X|-)\]\s*"
    r"|[☐☑]\s*"
    r"|\d+[.)]\s+"
    r"|[A-Za-z][.)]\s+"
    r")(.+?)\s*$"
)
_MARKDOWN_PREFIX_RE = re.compile(r"^(?:[-+*•]|\d+[.)]|[A-Za-z][.)]|[☐☑])\s+")
_SPACE_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{1,}|[\u4e00-\u9fff]{2,}")

_CONTRACT_HEADINGS = (
    "acceptance criteria",
    "acceptance",
    "requirements",
    "required behavior",
    "success criteria",
    "definition of done",
    "验收标准",
    "验收条件",
    "需求",
    "功能要求",
    "必须满足",
    "完成标准",
)
_NORMATIVE_RE = re.compile(
    r"\b(?:must|shall|required|need to|needs to|do not|don't|never|always)\b|"
    r"(?:必须|不得|需要|需|应当|禁止|确保|保留|完整)",
    re.IGNORECASE,
)
_QUESTION_RE = re.compile(r"(?:\?|？)\s*$")
_ERROR_TERMS = ("error", "exception", "failure", "fail", "错误", "异常", "失败", "伪造")
_CANCEL_TERMS = ("cancel", "abort", "interrupt", "取消", "中断", "终止")
_RECOVERY_TERMS = ("retry", "recover", "resume", "fallback", "重试", "恢复", "回退")
_VERIFY_TERMS = (
    "test", "verify", "verification", "validation", "check", "diagnostic",
    "测试", "验证", "校验", "检查", "诊断",
)
_STOP_WORDS = frozenset(
    {
        "must", "shall", "required", "need", "needs", "to", "the", "a", "an",
        "and", "or", "be", "is", "are", "that", "this", "with", "for", "of",
        "不得", "必须", "需要", "应当", "确保", "保留", "覆盖", "行为", "路径",
    }
)
_FILE_MUTATION_TOOLS = frozenset({"ast_edit", "patch", "write_file"})


class ClauseKind(str, Enum):
    BEHAVIOR = "behavior"
    ERROR = "error"
    CANCELLATION = "cancellation"
    RECOVERY = "recovery"
    COMPATIBILITY = "compatibility"
    VERIFICATION = "verification"
    REPORTING = "reporting"


class EvidencePolicy(str, Enum):
    DIRECT_TOOL = "direct_tool"
    DIRECT_VERIFICATION = "direct_verification"
    EXPLANATION_ALLOWED = "explanation_allowed"


class EvidenceKind(str, Enum):
    TOOL_STARTED = "tool_started"
    TOOL_COMPLETED = "tool_completed"
    TOOL_FAILED = "tool_failed"
    TOOL_CANCELLED = "tool_cancelled"
    TERMINAL_COMMAND = "terminal_command"
    VERIFICATION = "verification"
    WORKSPACE_MUTATION = "workspace_mutation"
    DIAGNOSTIC = "diagnostic"
    MODEL_CLAIM = "model_claim"
    TURN_TERMINAL = "turn_terminal"


class ClauseStatus(str, Enum):
    PENDING = "pending"
    EVIDENCED = "evidenced"
    PARTIAL = "partial"
    BLOCKED = "blocked"
    CONTRADICTED = "contradicted"
    WAIVED = "waived"


class AssessmentState(str, Enum):
    READY = "ready"
    NEEDS_AUDIT = "needs_audit"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True)
class CompletionClause:
    clause_id: str
    text: str
    source_line: int
    kind: ClauseKind
    policy: EvidencePolicy
    required: bool = True
    subject_id: str = ""


@dataclass(frozen=True)
class CompletionContract:
    contract_id: str
    session_id: str
    turn_id: str
    digest: str
    parser_version: int
    clauses: tuple[CompletionClause, ...]
    assist_eligible: bool = False


@dataclass(frozen=True)
class CompletionEvidence:
    evidence_id: str
    sequence: int
    kind: EvidenceKind
    summary: str
    ok: bool | None
    tool_name: str = ""
    tool_call_id: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ClauseAssessment:
    clause_id: str
    status: ClauseStatus
    evidence_ids: tuple[str, ...]
    explanation: str = ""
    subject_id: str = ""


@dataclass(frozen=True)
class CompletionAssessment:
    contract_id: str
    evidence_sequence: int
    terminal_reason: str
    state: AssessmentState
    clauses: tuple[ClauseAssessment, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the stable cross-surface projection of an assessment.

        The transcript never receives this object.  API callers, the kernel
        result and UI adapters can use it without importing the ledger's
        private persistence helpers.
        """
        return {
            "contract_id": self.contract_id,
            "evidence_sequence": self.evidence_sequence,
            "terminal_reason": self.terminal_reason,
            "state": self.state.value,
            "clauses": [
                {
                    "clause_id": item.clause_id,
                    "status": item.status.value,
                    "evidence_ids": list(item.evidence_ids),
                    "explanation": item.explanation,
                    **(
                        {"subject_id": item.subject_id}
                        if item.subject_id
                        else {}
                    ),
                }
                for item in self.clauses
            ],
        }

def _normalise(text: str) -> str:
    return _SPACE_RE.sub(" ", str(text or "").strip())


def _contains_any(text: str, terms: Iterable[str]) -> bool:
    lowered = text.casefold()
    return any(term.casefold() in lowered for term in terms)


def _kind_for_clause(text: str) -> ClauseKind:
    if _contains_any(text, _CANCEL_TERMS):
        return ClauseKind.CANCELLATION
    if _contains_any(text, _ERROR_TERMS):
        return ClauseKind.ERROR
    if _contains_any(text, _RECOVERY_TERMS):
        return ClauseKind.RECOVERY
    if _contains_any(text, _VERIFY_TERMS):
        return ClauseKind.VERIFICATION
    if _contains_any(text, ("api", "compatib", "public", "兼容", "接口", "签名")):
        return ClauseKind.COMPATIBILITY
    if _contains_any(text, ("report", "summar", "说明", "报告", "告诉我")):
        return ClauseKind.REPORTING
    return ClauseKind.BEHAVIOR


def _policy_for_kind(kind: ClauseKind) -> EvidencePolicy:
    if kind in {ClauseKind.ERROR, ClauseKind.CANCELLATION, ClauseKind.RECOVERY, ClauseKind.VERIFICATION}:
        return EvidencePolicy.DIRECT_VERIFICATION
    if kind is ClauseKind.REPORTING:
        return EvidencePolicy.EXPLANATION_ALLOWED
    return EvidencePolicy.DIRECT_TOOL


def _is_contract_heading(text: str) -> bool:
    lowered = _normalise(text).casefold().strip("#*_ -:：")
    for marker in _CONTRACT_HEADINGS:
        if lowered == marker:
            return True
        suffix = lowered[len(marker) :] if lowered.startswith(marker) else ""
        if suffix.startswith((" (", "（", " [", "【", " -", " —", " –", " ✅", " ✔")):
            return True
    return False


def _is_explicit_requirement(text: str) -> bool:
    body = _normalise(text)
    if not body or _QUESTION_RE.search(body):
        return False
    return bool(_NORMATIVE_RE.search(body))


def _clause_id(contract_seed: str, position: int, text: str) -> str:
    digest = hashlib.sha256(f"{contract_seed}:{position}:{text}".encode("utf-8")).hexdigest()
    return f"cc_{digest[:16]}"


def _fold_wrapped_list_item(lines: Sequence[str], index: int) -> tuple[str, int]:
    """Return one Markdown list item's wrapped prose and final source index.

    CommonMark continuation prose is indented relative to the list marker.
    Sibling/nested list markers remain independent clauses, while a relative
    four-space block is treated as code and never promoted into a contract.
    """
    raw = lines[index]
    match = _LIST_RE.match(raw)
    if match is None:
        return _normalise(raw), index
    parts = [match.group(1)]
    base_indent = len(raw) - len(raw.lstrip(" "))
    cursor = index + 1
    while cursor < len(lines):
        continuation = lines[cursor]
        stripped = continuation.strip()
        if (
            not stripped
            or stripped.startswith(">")
            or _FENCE_RE.match(stripped)
            or _HEADING_RE.match(continuation)
            or _PLAIN_HEADING_RE.match(continuation)
            or _LIST_RE.match(continuation)
            or (stripped.startswith("`") and stripped.endswith("`") and len(stripped) > 1)
        ):
            break
        continuation_indent = len(continuation) - len(continuation.lstrip(" "))
        relative_indent = continuation_indent - base_indent
        if relative_indent <= 0 or relative_indent >= 4:
            break
        parts.append(stripped)
        cursor += 1
    return _normalise(" ".join(parts)), cursor - 1


def compile_completion_contract(
    user_message: str | None,
    *,
    session_id: str | None,
    turn_id: str | None,
) -> CompletionContract:
    """Compile explicit user requirements without treating normal prose as a gate.

    A contract section makes every list item explicit.  Outside such a section,
    only normative sentences are collected.  Fenced code is intentionally
    ignored: examples and pasted source must not become runtime requirements.
    """
    source = str(user_message or "")
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    session = str(session_id or "")
    turn = str(turn_id or "")
    contract_id = f"contract_{hashlib.sha256(f'{session}:{turn}:{digest}'.encode('utf-8')).hexdigest()[:20]}"
    clauses: list[CompletionClause] = []
    seen: set[str] = set()
    contract_section: bool | None = None
    assist_eligible = False
    fence_marker = ""

    lines = source.splitlines()
    skip_through = -1
    for line_index, raw_line in enumerate(lines):
        if line_index <= skip_through:
            continue
        line_number = line_index + 1
        stripped = raw_line.strip()
        fence = _FENCE_RE.match(stripped)
        if fence:
            marker = fence.group(1)
            if not fence_marker:
                fence_marker = marker
            elif marker[0] == fence_marker[0] and len(marker) >= len(fence_marker):
                fence_marker = ""
            continue
        if (
            fence_marker
            or not stripped
            or stripped.startswith(">")
            or (stripped.startswith("`") and stripped.endswith("`") and len(stripped) > 1)
        ):
            continue
        heading = _HEADING_RE.match(raw_line)
        if heading:
            contract_section = _is_contract_heading(heading.group(1))
            continue
        plain_heading = _PLAIN_HEADING_RE.match(raw_line)
        if plain_heading:
            heading_text = plain_heading.group(1)
            if _is_contract_heading(heading_text) or not _is_explicit_requirement(
                heading_text
            ):
                contract_section = _is_contract_heading(heading_text)
                continue
        list_item = _LIST_RE.match(raw_line)
        if list_item:
            candidate, skip_through = _fold_wrapped_list_item(lines, line_index)
        else:
            candidate = _normalise(stripped)
        if candidate in {"[ ]", "[x]", "[X]", "[-]"}:
            continue
        explicit = contract_section is True and list_item is not None
        if contract_section is False:
            continue
        if not explicit and not _is_explicit_requirement(candidate):
            continue
        if _QUESTION_RE.search(candidate):
            continue
        key = candidate
        if key in seen:
            continue
        seen.add(key)
        if len(clauses) >= _MAX_CLAUSES:
            break
        kind = _kind_for_clause(candidate)
        assist_eligible = assist_eligible or explicit
        clauses.append(
            CompletionClause(
                clause_id=_clause_id(digest, len(clauses), candidate),
                text=candidate,
                source_line=line_number,
                kind=kind,
                policy=_policy_for_kind(kind),
                required=True,
            )
        )

    return CompletionContract(
        contract_id=contract_id,
        session_id=session,
        turn_id=turn,
        digest=digest,
        parser_version=_PARSER_VERSION,
        clauses=tuple(clauses),
        assist_eligible=assist_eligible,
    )


def _keywords(text: str) -> set[str]:
    words = {word.casefold() for word in _WORD_RE.findall(text or "")}
    return {word for word in words if word not in _STOP_WORDS and len(word) > 1}


def _evidence_matches_clause(clause: CompletionClause, evidence: CompletionEvidence) -> bool:
    detail_text = " ".join(
        f"{key}={value}"
        for key, value in evidence.details.items()
        if value is not None and value != "" and value is not False
    )
    text = f"{evidence.summary} {evidence.tool_name} {detail_text}"
    overlap = _keywords(clause.text) & _keywords(text)
    if clause.subject_id:
        raw_subjects = evidence.details.get("plan_step_ids")
        if raw_subjects is None:
            raw_subjects = evidence.details.get("subject_id")
        if isinstance(raw_subjects, (str, bytes)):
            evidence_subjects = {str(raw_subjects).strip()}
        elif isinstance(raw_subjects, Sequence):
            evidence_subjects = {
                str(item).strip() for item in raw_subjects if str(item).strip()
            }
        else:
            evidence_subjects = set()
        subject_matches = (
            clause.subject_id in evidence_subjects
            if evidence_subjects
            else bool(overlap)
        )
        if not subject_matches:
            return False
    if clause.kind is ClauseKind.CANCELLATION:
        return evidence.kind is EvidenceKind.TOOL_CANCELLED or _contains_any(text, _CANCEL_TERMS)
    if clause.kind is ClauseKind.ERROR:
        # A real failed verifier is contradictory evidence, but a *passing*
        # focused verification of an error-handling path is positive evidence.
        # Keeping both facts lets a later assessment distinguish "the test
        # crashed" from "the exception case was tested and handled".
        # An unrelated tool failure is not evidence about every error clause.
        if evidence.kind is EvidenceKind.VERIFICATION:
            # The terminal result envelope always contains an ``error`` key,
            # commonly with a null value.  Treating that serialization detail
            # as focused error-path coverage made every passing pytest command
            # prove every exception requirement.  Structured verification
            # must name the exception/error path in its actual scope, command,
            # or explicit failure detail.  Older durable events without the
            # structured fields retain their bounded summary as a fallback.
            structured_parts = [
                str(evidence.details.get(key) or "")
                for key in ("scope", "canonical_command", "kind", "reason")
            ]
            error_detail = evidence.details.get("error")
            if error_detail is not None and error_detail != "" and error_detail is not False:
                structured_parts.append(f"error {error_detail}")
            structured = " ".join(structured_parts)
            focus = structured if structured.strip() else evidence.summary
            return _contains_any(focus, _ERROR_TERMS)
        # A successful tool envelope may contain source text such as
        # ``FAILED 0`` or a JSON ``error: null`` field.  Only a real failed
        # tool fact can contradict an error clause; successful focused coverage
        # must arrive through the structured VERIFICATION rail above.
        return evidence.kind is EvidenceKind.TOOL_FAILED and bool(overlap)
    if clause.kind is ClauseKind.RECOVERY:
        return _contains_any(text, _RECOVERY_TERMS)
    if clause.kind is ClauseKind.VERIFICATION:
        return evidence.kind in {
            EvidenceKind.DIAGNOSTIC,
            EvidenceKind.VERIFICATION,
            EvidenceKind.TERMINAL_COMMAND,
        }
    return len(overlap) >= 2 or (len(overlap) == 1 and evidence.kind is EvidenceKind.VERIFICATION)


def _is_direct_evidence(evidence: CompletionEvidence) -> bool:
    return evidence.kind in {
        EvidenceKind.TOOL_COMPLETED,
        EvidenceKind.TOOL_FAILED,
        EvidenceKind.TOOL_CANCELLED,
        EvidenceKind.TERMINAL_COMMAND,
        EvidenceKind.VERIFICATION,
        EvidenceKind.DIAGNOSTIC,
    }


class CompletionContractLedger:
    """In-memory event reducer for one immutable contract.

    The class is intentionally usable without a database.  The persistence
    adapter writes its immutable records to ``SessionDB`` after this reducer has
    accepted them; tests and detached tool runners can therefore exercise the
    exact same semantics without opening the user's state database.
    """

    def __init__(self, contract: CompletionContract) -> None:
        self.contract = contract
        self._evidence: list[CompletionEvidence] = []
        self._by_id: dict[str, CompletionEvidence] = {}
        self._claims: dict[str, tuple[str, ...]] = {}

    @property
    def evidence(self) -> tuple[CompletionEvidence, ...]:
        return tuple(self._evidence)

    def record_evidence(
        self,
        *,
        kind: EvidenceKind,
        summary: str,
        ok: bool | None,
        tool_name: str = "",
        tool_call_id: str = "",
        details: Mapping[str, Any] | None = None,
        evidence_id: str | None = None,
    ) -> CompletionEvidence:
        """Append one immutable fact, rejecting conflicting duplicate delivery."""
        identity = str(evidence_id or f"evidence_{uuid.uuid4().hex}")
        body = _normalise(summary)[:_MAX_EVIDENCE_SUMMARY]
        data = dict(details or {})
        prior = self._by_id.get(identity)
        if prior is not None:
            if (
                prior.kind is kind
                and prior.summary == body
                and prior.ok is ok
                and prior.tool_name == str(tool_name or "")
                and prior.tool_call_id == str(tool_call_id or "")
                and dict(prior.details) == data
            ):
                return prior
            raise ValueError(f"conflicting completion evidence id: {identity}")
        event = CompletionEvidence(
            evidence_id=identity,
            sequence=len(self._evidence) + 1,
            kind=kind,
            summary=body,
            ok=ok,
            tool_name=str(tool_name or ""),
            tool_call_id=str(tool_call_id or ""),
            details=data,
        )
        self._evidence.append(event)
        self._by_id[identity] = event
        return event

    def claim_clause(self, clause_id: str, evidence_ids: Sequence[str]) -> bool:
        """Accept an explicit model mapping only when direct evidence supports it.

        This prevents narrative self-certification: a model cannot mark a clause
        done with prose alone, and an inspection-only read cannot stand in for a
        focused verification requirement.
        """
        clause = next((item for item in self.contract.clauses if item.clause_id == clause_id), None)
        if clause is None or not evidence_ids:
            return False
        evidence = [self._by_id.get(str(item)) for item in evidence_ids]
        if any(item is None or not _is_direct_evidence(item) for item in evidence):
            return False
        resolved = [item for item in evidence if item is not None]
        matching = [item for item in resolved if _evidence_matches_clause(clause, item)]
        if not matching:
            return False
        if clause.policy is EvidencePolicy.DIRECT_VERIFICATION and not any(
            item.kind in {
                EvidenceKind.DIAGNOSTIC,
                EvidenceKind.VERIFICATION,
                EvidenceKind.TERMINAL_COMMAND,
            }
            and item.ok is True
            for item in matching
        ):
            return False
        if clause.policy is EvidencePolicy.DIRECT_TOOL and not any(item.ok is True for item in matching):
            return False
        self._claims[clause_id] = tuple(dict.fromkeys(str(item) for item in evidence_ids))
        return True

    def _assess_clause(self, clause: CompletionClause) -> ClauseAssessment:
        matching = [item for item in self._evidence if _evidence_matches_clause(clause, item)]
        latest_mutation = next(
            (
                item
                for item in reversed(self._evidence)
                if item.kind is EvidenceKind.WORKSPACE_MUTATION
            ),
            None,
        )
        mutation_sequence = latest_mutation.sequence if latest_mutation is not None else 0
        current_matching = (
            [item for item in matching if item.sequence > mutation_sequence]
            if mutation_sequence and clause.policy is not EvidencePolicy.EXPLANATION_ALLOWED
            else matching
        )
        ids = tuple(item.evidence_id for item in current_matching)
        claim = self._claims.get(clause.clause_id)
        claimed_events = [self._by_id.get(item) for item in (claim or ())]
        claim_is_stale = bool(
            mutation_sequence
            and clause.policy is not EvidencePolicy.EXPLANATION_ALLOWED
            and any(
                item is not None and item.sequence <= mutation_sequence
                for item in claimed_events
            )
        )
        if claim and not claim_is_stale:
            return ClauseAssessment(
                clause.clause_id,
                ClauseStatus.EVIDENCED,
                claim,
                clause.text,
                clause.subject_id,
            )
        stale_matching = [item for item in matching if item.sequence <= mutation_sequence]
        if (claim_is_stale or stale_matching) and latest_mutation is not None and not current_matching:
            stale_ids = tuple(
                dict.fromkeys(
                    [*(claim or ()), *(item.evidence_id for item in stale_matching), latest_mutation.evidence_id]
                )
            )
            return ClauseAssessment(
                clause.clause_id,
                ClauseStatus.PARTIAL,
                stale_ids,
                f"{clause.text}; prior evidence is stale after a workspace mutation",
                clause.subject_id,
            )
        if not current_matching:
            return ClauseAssessment(
                clause.clause_id,
                ClauseStatus.PENDING,
                (),
                clause.text,
                clause.subject_id,
            )
        direct_verification = [
            item
            for item in current_matching
            if item.kind in {
                EvidenceKind.DIAGNOSTIC,
                EvidenceKind.VERIFICATION,
                EvidenceKind.TERMINAL_COMMAND,
            }
        ]
        # Canonically classified verification is direct evidence in its own
        # right.  Requiring a model to name an internal evidence id would make
        # the contract a hidden second tool protocol and would fail precisely
        # when the model already ran the right test.  This is intentionally
        # narrower than arbitrary exit-zero terminal output: only the existing
        # terminal verifier is promoted to EvidenceKind.VERIFICATION.
        if clause.policy is EvidencePolicy.DIRECT_VERIFICATION and direct_verification:
            latest_direct = direct_verification[-1]
            if latest_direct.ok is True:
                return ClauseAssessment(
                    clause.clause_id,
                    ClauseStatus.EVIDENCED,
                    (latest_direct.evidence_id,),
                    clause.text,
                    clause.subject_id,
                )
            if latest_direct.ok is False:
                return ClauseAssessment(
                    clause.clause_id,
                    ClauseStatus.CONTRADICTED,
                    (latest_direct.evidence_id,),
                    f"{clause.text}; latest direct check failed: {latest_direct.summary}",
                    clause.subject_id,
                )
        if clause.policy is EvidencePolicy.DIRECT_TOOL:
            # A passing verifier alone is never universal proof.  A matching
            # real tool observation plus a fresh structured pass is, however,
            # the complete policy bundle for a direct-tool clause.  Requiring
            # the model to call ``claim_clause`` with internal evidence IDs
            # created an impossible hidden protocol and left correctly tested
            # work permanently partial.
            supporting = [
                item
                for item in current_matching
                if item.kind is EvidenceKind.TOOL_COMPLETED and item.ok is True
            ]
            fresh_verification = [
                item
                for item in self._evidence
                if item.sequence > mutation_sequence
                and item.kind in {
                    EvidenceKind.DIAGNOSTIC,
                    EvidenceKind.VERIFICATION,
                    EvidenceKind.TERMINAL_COMMAND,
                }
                and item.ok is True
            ]
            if supporting and fresh_verification:
                proof_ids = tuple(
                    dict.fromkeys(
                        (
                            supporting[-1].evidence_id,
                            fresh_verification[-1].evidence_id,
                        )
                    )
                )
                return ClauseAssessment(
                    clause.clause_id,
                    ClauseStatus.EVIDENCED,
                    proof_ids,
                    clause.text,
                    clause.subject_id,
                )
        failed_match = next(
            (item for item in reversed(current_matching) if item.ok is False),
            None,
        )
        if clause.kind is ClauseKind.ERROR and failed_match is not None:
            return ClauseAssessment(
                clause.clause_id,
                ClauseStatus.CONTRADICTED,
                ids,
                f"{clause.text}; observed failure: {failed_match.summary}",
                clause.subject_id,
            )
        if clause.kind is ClauseKind.CANCELLATION:
            return ClauseAssessment(
                clause.clause_id,
                ClauseStatus.PARTIAL,
                ids,
                f"{clause.text}; cancellation was observed but not independently verified",
                clause.subject_id,
            )
        return ClauseAssessment(
            clause.clause_id,
            ClauseStatus.PARTIAL,
            ids,
            f"{clause.text}; direct evidence exists but has not been explicitly mapped",
            clause.subject_id,
        )

    def assess(self, *, terminal_reason: str) -> CompletionAssessment:
        """Produce a deterministic snapshot for the current evidence sequence."""
        reason = str(terminal_reason or "natural").casefold()
        clauses = tuple(self._assess_clause(clause) for clause in self.contract.clauses)
        if reason in {"cancelled", "canceled", "interrupted", "keyboard_interrupt"}:
            state = AssessmentState.CANCELLED
        elif reason.startswith("budget"):
            state = AssessmentState.BLOCKED
        elif reason in {"provider_error", "error", "failed"}:
            state = AssessmentState.FAILED
        elif not clauses or all(item.status in {ClauseStatus.EVIDENCED, ClauseStatus.WAIVED} for item in clauses):
            state = AssessmentState.READY
        else:
            state = AssessmentState.NEEDS_AUDIT
        return CompletionAssessment(
            contract_id=self.contract.contract_id,
            evidence_sequence=len(self._evidence),
            terminal_reason=reason,
            state=state,
            clauses=clauses,
        )

def _contract_payload(contract: CompletionContract) -> dict[str, Any]:
    return {
        "contract_id": contract.contract_id,
        "session_id": contract.session_id,
        "turn_id": contract.turn_id,
        "digest": contract.digest,
        "parser_version": contract.parser_version,
        "assist_eligible": contract.assist_eligible,
        "clauses": [
            {
                "clause_id": item.clause_id,
                "text": item.text,
                "source_line": item.source_line,
                "kind": item.kind.value,
                "policy": item.policy.value,
                "required": item.required,
                **({"subject_id": item.subject_id} if item.subject_id else {}),
            }
            for item in contract.clauses
        ],
    }


def _evidence_payload(evidence: CompletionEvidence) -> dict[str, Any]:
    return {
        "evidence_id": evidence.evidence_id,
        "sequence": evidence.sequence,
        "kind": evidence.kind.value,
        "summary": evidence.summary,
        "ok": evidence.ok,
        "tool_name": evidence.tool_name,
        "tool_call_id": evidence.tool_call_id,
        "details": dict(evidence.details),
    }


def _assessment_payload(assessment: CompletionAssessment) -> dict[str, Any]:
    return assessment.to_dict()


def _contract_from_payload(payload: Mapping[str, Any]) -> CompletionContract:
    """Rehydrate the immutable compiled form; never reparse the source prompt."""
    raw_clauses = payload.get("clauses")
    if not isinstance(raw_clauses, Sequence) or isinstance(raw_clauses, (str, bytes)):
        raise ValueError("durable completion contract clauses are malformed")
    clauses: list[CompletionClause] = []
    for raw in raw_clauses:
        if not isinstance(raw, Mapping):
            raise ValueError("durable completion contract clause is malformed")
        clauses.append(
            CompletionClause(
                clause_id=str(raw.get("clause_id") or ""),
                text=str(raw.get("text") or ""),
                source_line=int(raw.get("source_line") or 0),
                kind=ClauseKind(str(raw.get("kind") or "")),
                policy=EvidencePolicy(str(raw.get("policy") or "")),
                required=bool(raw.get("required", True)),
                subject_id=str(raw.get("subject_id") or ""),
            )
        )
    contract = CompletionContract(
        contract_id=str(payload.get("contract_id") or ""),
        session_id=str(payload.get("session_id") or ""),
        turn_id=str(payload.get("turn_id") or ""),
        digest=str(payload.get("digest") or ""),
        parser_version=int(payload.get("parser_version") or 0),
        clauses=tuple(clauses),
        # Payloads written before parser v4 remain safely auditable but do not
        # gain new model-facing guidance after a restart.
        assist_eligible=payload.get("assist_eligible") is True,
    )
    if (
        not contract.contract_id
        or not contract.session_id
        or not contract.turn_id
        or not contract.digest
        or contract.parser_version <= 0
        or any(not clause.clause_id or not clause.text for clause in contract.clauses)
    ):
        raise ValueError("durable completion contract identity is malformed")
    return contract


def _redacted_evidence_summary(value: Any) -> str:
    """Return a bounded fact suitable for the durable evidence ledger.

    Tool results are useful evidence, but they can contain source fragments or
    credentials.  Keep this reduction in the deep completion module so v1 and
    v2 cannot accidentally persist different raw output formats.
    """
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            text = str(value)
    try:
        from agent.redact import redact_sensitive_text

        text = redact_sensitive_text(text)
    except Exception:
        pass
    return _normalise(text)[:_MAX_EVIDENCE_SUMMARY]


def _verification_payload_from_tool_result(tool_result: Any) -> Mapping[str, Any] | None:
    """Accept only the existing structured terminal-verifier rail.

    An exit-zero shell result is not elevated to verification evidence here.
    The terminal tool already classifies canonical verification and returns the
    small ``verification_evidence`` object.  This reducer deliberately rejects
    all other shapes, including a model-written boolean field.
    """
    parsed = _tool_result_mapping(tool_result)
    if parsed is None:
        return None
    payload = parsed.get("verification_evidence")
    return payload if isinstance(payload, Mapping) else None


def _tool_result_mapping(tool_result: Any) -> Mapping[str, Any] | None:
    if isinstance(tool_result, Mapping):
        return tool_result
    if not isinstance(tool_result, str):
        return None
    try:
        parsed = json.loads(tool_result)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, Mapping) else None


def _workspace_mutation_paths(
    tool_name: str, result: Mapping[str, Any] | None
) -> tuple[str, ...]:
    from agent.tool_result_classification import workspace_mutation_paths

    return workspace_mutation_paths(tool_name, result)


def _diagnostic_fact(
    tool_name: str, result: Mapping[str, Any] | None
) -> tuple[str, bool, dict[str, Any]] | None:
    if result is None:
        return None
    name = str(tool_name or "")
    if name == "lsp" and result.get("action") == "diagnostics":
        try:
            count = max(0, int(result.get("count") or 0))
        except (TypeError, ValueError):
            return None
        details = {
            "action": "diagnostics",
            "count": count,
            "file": str(result.get("file") or ""),
        }
        return (
            f"lsp diagnostics reported {count} issue(s)",
            bool(result.get("success")) and count == 0,
            details,
        )
    inline = result.get("lsp_diagnostics")
    if name in _FILE_MUTATION_TOOLS and isinstance(inline, str) and inline.strip():
        return (
            _normalise(inline),
            False,
            {
                "action": "post_edit_diagnostics",
                "file": str(result.get("resolved_path") or result.get("file") or ""),
            },
        )
    return None


class CompletionContractRuntime:
    """One turn's ledger plus the optional ``SessionDB`` persistence adapter.

    Runtime failure to write observability facts must never fail the user's
    actual task. The failure is retained in ``persistence_error`` and the
    in-memory assessment remains available.
    """

    VALID_MODES = frozenset({"off", "observe"})

    def __init__(
        self,
        contract: CompletionContract,
        *,
        mode: str = "off",
        db: Any = None,
    ) -> None:
        selected = str(mode or "off").casefold()
        if selected == "assist":
            selected = "observe"
        self.mode = selected if selected in self.VALID_MODES else "off"
        self.ledger = CompletionContractLedger(contract)
        self._db = db
        self.persistence_error: str = ""
        self._durable = False
        if self.mode != "off":
            self._open_store()

    @classmethod
    def from_durable(
        cls,
        db: Any,
        contract_id: str,
        *,
        mode: str = "observe",
    ) -> "CompletionContractRuntime":
        """Rebuild an exact ledger from committed rows after process restart.

        SQLite transactions prevent half-row tails.  This method additionally
        checks the persisted event sequence against reducer order, so a corrupt
        or manually edited tail fails closed instead of being silently
        renumbered into plausible-looking evidence.
        """
        payload = db.read_completion_contract(str(contract_id or ""))
        if not isinstance(payload, Mapping):
            raise ValueError("unknown durable completion contract")
        runtime = cls(
            _contract_from_payload(payload),
            mode=mode,
            db=db,
        )
        for expected_sequence, raw in enumerate(
            db.read_completion_evidence(runtime.contract.contract_id), start=1
        ):
            if not isinstance(raw, Mapping):
                raise ValueError("durable completion evidence is malformed")
            stored_sequence = int(raw.get("sequence") or 0)
            details = raw.get("details")
            if stored_sequence != expected_sequence or not isinstance(details, Mapping):
                raise ValueError("durable completion evidence sequence is malformed")
            ok = raw.get("ok")
            if ok is not None and not isinstance(ok, bool):
                raise ValueError("durable completion evidence status is malformed")
            event = runtime.ledger.record_evidence(
                kind=EvidenceKind(str(raw.get("kind") or "")),
                summary=str(raw.get("summary") or ""),
                ok=ok,
                tool_name=str(raw.get("tool_name") or ""),
                tool_call_id=str(raw.get("tool_call_id") or ""),
                details=dict(details),
                evidence_id=str(raw.get("evidence_id") or ""),
            )
            if event.sequence != stored_sequence:
                raise ValueError("durable completion evidence replay diverged")
        return runtime

    @property
    def contract(self) -> CompletionContract:
        return self.ledger.contract

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @property
    def durable(self) -> bool:
        return self._durable

    def _open_store(self) -> None:
        db = self._db
        if db is None:
            self.persistence_error = "SessionDB unavailable"
            return
        try:
            db.apply_completion_contract_migration()
            db.upsert_completion_contract(_contract_payload(self.contract))
        except Exception as exc:
            self.persistence_error = f"{type(exc).__name__}: {exc}"
            return
        self._durable = True

    def _tool_evidence_id(self, tool_call_id: str, facet: str) -> str | None:
        call_id = str(tool_call_id or "")
        if not call_id:
            return None
        digest = hashlib.sha256(
            f"{self.contract.contract_id}:{call_id}:{facet}".encode("utf-8")
        ).hexdigest()
        return f"evidence_{digest[:24]}"

    def record_tool_terminal(
        self,
        *,
        tool_name: str,
        tool_call_id: str,
        summary: str,
        failed: bool = False,
        cancelled: bool = False,
        details: Optional[Mapping[str, Any]] = None,
        evidence_id: str | None = None,
    ) -> CompletionEvidence | None:
        if not self.enabled:
            return None
        kind = (
            EvidenceKind.TOOL_CANCELLED if cancelled
            else EvidenceKind.TOOL_FAILED if failed
            else EvidenceKind.TOOL_COMPLETED
        )
        return self._record(
            kind=kind,
            summary=summary,
            ok=False if (failed or cancelled) else True,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            details=details,
            evidence_id=evidence_id,
        )

    def record_verification(
        self,
        *,
        summary: str,
        ok: bool,
        tool_call_id: str = "",
        details: Optional[Mapping[str, Any]] = None,
        evidence_id: str | None = None,
    ) -> CompletionEvidence | None:
        if not self.enabled:
            return None
        return self._record(
            kind=EvidenceKind.VERIFICATION,
            summary=summary,
            ok=bool(ok),
            tool_name="terminal",
            tool_call_id=tool_call_id,
            details=details,
            evidence_id=evidence_id,
        )

    def record_workspace_observation(
        self,
        *,
        root: str,
        changed_paths: Sequence[str],
        state_digest: str,
    ) -> CompletionEvidence | None:
        """Record a real turn-diff observation from the original workspace."""
        paths = tuple(dict.fromkeys(str(path) for path in changed_paths if path))
        if not self.enabled or not paths:
            return None
        identity = hashlib.sha256(
            f"{self.contract.contract_id}:{root}:{state_digest}".encode("utf-8")
        ).hexdigest()[:24]
        return self._record(
            kind=EvidenceKind.WORKSPACE_MUTATION,
            summary=f"workspace diff changed {len(paths)} path(s)",
            ok=True,
            tool_name="workspace_diff",
            details={
                "root": str(root or ""),
                "paths": list(paths),
                "state_digest": str(state_digest or ""),
            },
            evidence_id=f"workspace_{identity}",
        )

    def record_tool_outcome(
        self,
        *,
        tool_name: str,
        tool_call_id: str,
        tool_result: Any,
        failed: bool = False,
        cancelled: bool = False,
        details: Optional[Mapping[str, Any]] = None,
    ) -> tuple[CompletionEvidence, ...]:
        """Reduce one real terminal tool outcome into all applicable facts.

        This is the sole v1/v2 adapter interface.  It records the terminal
        outcome first, promotes only structured terminal verification, and
        never lets an adapter reinterpret a malformed verifier payload as
        success.  The method has no control-flow side effect: model autonomy,
        retries, and the transcript remain owned by the caller.
        """
        if not self.enabled:
            return ()
        name = str(tool_name or "")
        call_id = str(tool_call_id or "")
        evidence_details = dict(details or {})
        plan_subject_ids = tuple(
            clause.subject_id
            for clause in self.contract.clauses
            if clause.subject_id
        )
        # A stepwise Plan turn has one unambiguous evidence owner.  Tag its
        # real tool facts automatically; multi-step turns remain unassigned
        # unless the evidence itself names a step, preventing one broad test
        # from certifying every step.
        if len(plan_subject_ids) == 1 and not evidence_details.get(
            "plan_step_ids"
        ):
            evidence_details["plan_step_ids"] = [plan_subject_ids[0]]
        output_summary = _redacted_evidence_summary(tool_result)
        terminal = self.record_tool_terminal(
            tool_name=name,
            tool_call_id=call_id,
            summary=f"{name}: {output_summary}",
            failed=failed,
            cancelled=cancelled,
            details=evidence_details,
            evidence_id=self._tool_evidence_id(call_id, "terminal"),
        )
        recorded: list[CompletionEvidence] = [terminal] if terminal is not None else []

        parsed_result = _tool_result_mapping(tool_result)
        mutation_paths = (
            ()
            if failed or cancelled
            else _workspace_mutation_paths(name, parsed_result)
        )
        if mutation_paths:
            mutation = self._record(
                kind=EvidenceKind.WORKSPACE_MUTATION,
                summary=f"{name} modified {len(mutation_paths)} workspace path(s)",
                ok=True,
                tool_name=name,
                tool_call_id=call_id,
                details={**evidence_details, "paths": list(mutation_paths)},
                evidence_id=self._tool_evidence_id(call_id, "workspace_mutation"),
            )
            recorded.append(mutation)

        diagnostic = _diagnostic_fact(name, parsed_result)
        if diagnostic is not None and not cancelled:
            summary, diagnostic_ok, diagnostic_details = diagnostic
            fact = self._record(
                kind=EvidenceKind.DIAGNOSTIC,
                summary=summary,
                ok=diagnostic_ok and not failed,
                tool_name=name,
                tool_call_id=call_id,
                details={**evidence_details, **diagnostic_details},
                evidence_id=self._tool_evidence_id(call_id, "diagnostic"),
            )
            recorded.append(fact)

        verification = (
            _verification_payload_from_tool_result(tool_result)
            if name == "terminal"
            else None
        )
        if verification is not None:
            status = str(verification.get("status") or "").casefold()
            kind = str(verification.get("kind") or "verification")
            scope = str(verification.get("scope") or "unknown")
            canonical_command = str(
                verification.get("canonical_command") or "verification command"
            )
            fact = self.record_verification(
                summary=(
                    f"{kind} verification {status or 'unknown'} "
                    f"({scope}): {canonical_command}"
                ),
                ok=status == "passed",
                tool_call_id=call_id,
                details={
                    **evidence_details,
                    "status": status,
                    "kind": kind,
                    "scope": scope,
                    "canonical_command": canonical_command,
                },
                evidence_id=self._tool_evidence_id(call_id, "verification"),
            )
            if fact is not None:
                recorded.append(fact)
        return tuple(recorded)

    def _record(self, **kwargs: Any) -> CompletionEvidence:
        evidence = self.ledger.record_evidence(**kwargs)
        if self._durable:
            try:
                self._db.append_completion_evidence(
                    self.contract.contract_id, _evidence_payload(evidence)
                )
            except Exception as exc:
                self.persistence_error = f"{type(exc).__name__}: {exc}"
                self._durable = False
        return evidence

    def assess(self, *, terminal_reason: str, phase: str) -> CompletionAssessment:
        assessment = self.ledger.assess(terminal_reason=terminal_reason)
        if self._durable:
            try:
                self._db.save_completion_assessment(
                    self.contract.contract_id,
                    assessment.evidence_sequence,
                    phase,
                    _assessment_payload(assessment),
                )
            except Exception as exc:
                self.persistence_error = f"{type(exc).__name__}: {exc}"
                self._durable = False
        return assessment

def completion_contract_runtime_for_turn(
    agent: Any,
    user_message: Any,
) -> CompletionContractRuntime | None:
    """Create the session-pinned runtime from ``config.yaml`` once per session.

    The helper deliberately reads no environment variable.  A mode change can
    affect only a new session, which keeps an existing conversation's prompt
    and behavior stable for cache/replay purposes.
    """
    settings = getattr(agent, "_completion_contract_settings", None)
    if not isinstance(settings, Mapping):
        raw: Mapping[str, Any] | str = {}
        try:
            from hairball_cli.config import load_config

            loaded = ((load_config() or {}).get("agent") or {}).get("completion_contract") or {}
            raw = loaded if isinstance(loaded, (Mapping, str)) else {}
        except Exception:
            raw = {}
        selected = raw if isinstance(raw, str) else raw.get("mode", "off")
        selected = str(selected or "off").casefold()
        if selected == "assist":
            selected = "observe"
        if selected not in CompletionContractRuntime.VALID_MODES:
            selected = "off"
        settings = {"mode": selected}
        try:
            agent._completion_contract_settings = dict(settings)
        except Exception:
            pass
    selected = str(settings.get("mode") or "off").casefold()

    if selected == "off":
        return None
    contract = compile_completion_contract(
        str(user_message or ""),
        session_id=getattr(agent, "session_id", None),
        turn_id=getattr(agent, "_current_turn_id", None),
    )
    return CompletionContractRuntime(
        contract,
        mode=str(selected),
        db=getattr(agent, "_session_db", None),
    )

def completion_assessment_status_line(assessment: Any) -> str:
    """Return one bounded, transport-neutral status line for local surfaces."""
    if not isinstance(assessment, Mapping):
        return ""
    state = str(assessment.get("state") or "").strip().replace("_", " ")
    raw_clauses = assessment.get("clauses")
    if not state or not isinstance(raw_clauses, Sequence) or isinstance(
        raw_clauses, (str, bytes)
    ):
        return ""
    statuses = [
        str(item.get("status") or "")
        for item in raw_clauses
        if isinstance(item, Mapping)
    ]
    total = len(statuses)
    evidenced = sum(status in {"evidenced", "waived"} for status in statuses)
    contradicted = statuses.count("contradicted")
    pending = sum(status in {"pending", "partial", "blocked"} for status in statuses)
    parts = [f"Completion evidence: {state}", f"{evidenced}/{total} evidenced"]
    if contradicted:
        parts.append(f"{contradicted} contradicted")
    if pending:
        parts.append(f"{pending} pending")
    return " · ".join(parts)


__all__ = [
    "AssessmentState",
    "ClauseAssessment",
    "ClauseKind",
    "ClauseStatus",
    "CompletionAssessment",
    "CompletionClause",
    "CompletionContract",
    "CompletionContractLedger",
    "CompletionContractRuntime",
    "CompletionEvidence",
    "EvidenceKind",
    "EvidencePolicy",
    "compile_completion_contract",
    "completion_assessment_status_line",
    "completion_contract_runtime_for_turn",
]
