from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

JSONDict = dict[str, Any]


class EvidenceRoute(str, Enum):
    READY_FOR_RULE = "READY_FOR_RULE"
    NEEDS_CONTEXT = "NEEDS_CONTEXT"
    UNRESOLVABLE = "UNRESOLVABLE"


class EvidenceIssue(str, Enum):
    LOW_RESOLUTION = "LOW_RESOLUTION"
    TARGET_TOO_SMALL = "TARGET_TOO_SMALL"
    LOW_LIGHT = "LOW_LIGHT"
    OCCLUDED = "OCCLUDED"
    RELATION_UNCLEAR = "RELATION_UNCLEAR"
    TEMPORAL_CONTEXT_INSUFFICIENT = "TEMPORAL_CONTEXT_INSUFFICIENT"
    BOUNDARY_STATE_UNCLEAR = "BOUNDARY_STATE_UNCLEAR"
    FRAME_ERROR = "FRAME_ERROR"


class EvidenceReasonCode(str, Enum):
    EVIDENCE_SUFFICIENT = "EVIDENCE_SUFFICIENT"
    MISSING_PPE_STATUS = "MISSING_PPE_STATUS"
    MISSING_REGION_GROUNDING = "MISSING_REGION_GROUNDING"
    MISSING_FRAME_REF = "MISSING_FRAME_REF"
    MISSING_CROP_REF = "MISSING_CROP_REF"
    MISSING_SOURCE_REF = "MISSING_SOURCE_REF"
    TARGET_NOT_IN_DETECTIONS = "TARGET_NOT_IN_DETECTIONS"
    TARGET_ID_MISMATCH = "TARGET_ID_MISMATCH"
    PPE_EVIDENCE_REF_NOT_FOUND = "PPE_EVIDENCE_REF_NOT_FOUND"
    INVALID_DETECTION = "INVALID_DETECTION"
    INVALID_PPE_CONFIDENCE = "INVALID_PPE_CONFIDENCE"
    INVALID_GROUNDING_CONFIDENCE = "INVALID_GROUNDING_CONFIDENCE"
    UNKNOWN_ZONE = "UNKNOWN_ZONE"
    UNSUPPORTED_ZONE_TYPE = "UNSUPPORTED_ZONE_TYPE"
    TARGET_CONFIDENCE_BELOW_THRESHOLD = "TARGET_CONFIDENCE_BELOW_THRESHOLD"
    PPE_CONFIDENCE_BELOW_THRESHOLD = "PPE_CONFIDENCE_BELOW_THRESHOLD"
    GROUNDING_CONFIDENCE_BELOW_THRESHOLD = "GROUNDING_CONFIDENCE_BELOW_THRESHOLD"
    PPE_CONFLICT = "PPE_CONFLICT"
    PPE_STATE_UNCERTAIN = "PPE_STATE_UNCERTAIN"
    VISUAL_AMBIGUITY = "VISUAL_AMBIGUITY"
    RELATION_AMBIGUOUS = "RELATION_AMBIGUOUS"
    TEMPORAL_AMBIGUITY = "TEMPORAL_AMBIGUITY"
    MEDIA_NOT_RECOVERABLE = "MEDIA_NOT_RECOVERABLE"
    CONTEXT_ATTEMPT_BUDGET_EXHAUSTED = "CONTEXT_ATTEMPT_BUDGET_EXHAUSTED"
    CONTEXT_EVIDENCE_DID_NOT_RESOLVE = "CONTEXT_EVIDENCE_DID_NOT_RESOLVE"
    RULE_APPLICATION_FAILED = "RULE_APPLICATION_FAILED"


@dataclass
class Detection:
    object_id: str
    class_label: str
    bbox: list[float]
    confidence: float
    attributes: JSONDict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: JSONDict) -> "Detection":
        return cls(
            object_id=str(data["object_id"]),
            class_label=str(data.get("class", data.get("class_label", ""))),
            bbox=[float(value) for value in data["bbox"]],
            confidence=float(data["confidence"]),
            attributes=dict(data.get("attributes", {})),
        )

    def to_dict(self) -> JSONDict:
        return {
            "object_id": self.object_id,
            "class": self.class_label,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "attributes": self.attributes,
        }


@dataclass
class PPEStatus:
    helmet: str
    confidence: float
    target_id: str = ""
    evidence_detection_ids: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> JSONDict:
        return asdict(self)

@dataclass
class RegionGrounding:
    zone_id: str
    zone_type: str
    spatial_relation: str
    confidence: float
    target_id: str = ""
    rule_source: str = "manual"
    anchor_point: list[float] = field(default_factory=list)

    def to_dict(self) -> JSONDict:
        return asdict(self)


class ContextEvidenceKind(str, Enum):
    VISUAL_QUALITY = "VISUAL_QUALITY"
    LOCAL_RELATION = "LOCAL_RELATION"
    INPUT_CONFIDENCE = "INPUT_CONFIDENCE"


class ContextEvidenceStatus(str, Enum):
    CONFIRMED = "CONFIRMED"
    PROVISIONAL = "PROVISIONAL"


class ContextAction(str, Enum):
    EMIT_CONTEXT_EVIDENCE = "EMIT_CONTEXT_EVIDENCE"
    REQUEST_HIGHER_RESOLUTION_CROP = "REQUEST_HIGHER_RESOLUTION_CROP"
    REQUEST_MORE_FRAMES = "REQUEST_MORE_FRAMES"
    ABSTAIN = "ABSTAIN"


PPE_REQUIRED_ZONE_TYPES = frozenset(
    {"active_work_area", "restricted_zone", "work_at_height"}
)
PPE_EXEMPT_ZONE_TYPES = frozenset({"site_office", "rest_area"})
SUPPORTED_ZONE_TYPES = PPE_REQUIRED_ZONE_TYPES | PPE_EXEMPT_ZONE_TYPES


VISUAL_QUALITY_LABELS = frozenset(
    {
        "CLEAR",
        "LOW_RESOLUTION",
        "TARGET_TOO_SMALL",
        "BLUR",
        "LOW_LIGHT",
        "OCCLUDED",
        "OUT_OF_FRAME",
        "FRAME_ERROR",
        "TEMPORAL_CONTEXT_INSUFFICIENT",
    }
)

LOCAL_RELATION_LABELS = frozenset(
    {
        "NEAR",
        "ADJACENT",
        "OVERLAPPING",
        "OCCLUDED_BY",
        "RELATION_UNCLEAR",
        "BOUNDARY_STATE_UNCLEAR",
    }
)

INPUT_CONFIDENCE_LABELS = frozenset(
    {
        "UPSTREAM_CONFIDENCE_ACCEPTABLE",
        "UPSTREAM_CONFIDENCE_LOW",
        "UPSTREAM_CONFIDENCE_INVALID",
    }
)


@dataclass
class ContextRequest:
    request_id: str
    event_id: str
    worker_id: str
    frame_ref: str
    crop_ref: str
    detections: list[Detection]
    ppe_finding: PPEStatus
    zone_grounding: RegionGrounding
    neighbor_frame_refs: list[str] = field(default_factory=list)
    higher_resolution_source_ref: str | None = None
    evidence_issues: list[EvidenceIssue] = field(default_factory=list)
    context_attempt_count: int = 0
    max_context_attempts: int = 1
    allowed_context_actions: list[ContextAction] = field(default_factory=list)

    def to_dict(self) -> JSONDict:
        return {
            "request_id": self.request_id,
            "event_id": self.event_id,
            "worker_id": self.worker_id,
            "frame_ref": self.frame_ref,
            "crop_ref": self.crop_ref,
            "neighbor_frame_refs": self.neighbor_frame_refs,
            "higher_resolution_source_ref": self.higher_resolution_source_ref,
            "evidence_issues": [issue.value for issue in self.evidence_issues],
            "allowed_context_actions": [
                action.value for action in self.allowed_context_actions
            ],
            "context_attempt_count": self.context_attempt_count,
            "max_context_attempts": self.max_context_attempts,
            "detections": [detection.to_dict() for detection in self.detections],
            "ppe_finding": self.ppe_finding.to_dict(),
            "zone_grounding": self.zone_grounding.to_dict(),
        }


@dataclass
class ContextEvidence:
    evidence_id: str
    kind: ContextEvidenceKind | str
    label: str
    subject_detection_id: str
    frame_ref: str
    crop_ref: str
    zone_ref: str
    confidence: float
    status: ContextEvidenceStatus | str
    reason_code: str
    object_detection_id: str | None = None

    def to_dict(self) -> JSONDict:
        return {
            "evidence_id": self.evidence_id,
            "kind": self.kind.value if isinstance(self.kind, ContextEvidenceKind) else self.kind,
            "label": self.label,
            "subject_detection_id": self.subject_detection_id,
            "object_detection_id": self.object_detection_id,
            "frame_ref": self.frame_ref,
            "crop_ref": self.crop_ref,
            "zone_ref": self.zone_ref,
            "confidence": self.confidence,
            "status": self.status.value
            if isinstance(self.status, ContextEvidenceStatus)
            else self.status,
            "reason_code": self.reason_code,
        }


@dataclass
class ContextProposal:
    evidence: list[ContextEvidence]
    selected_action: ContextAction | str
    action_parameters: JSONDict = field(default_factory=dict)
    context_confidence: float = 0.0
    model_metadata: JSONDict = field(default_factory=dict)


@dataclass
class EmitContextEvidenceParameters:
    evidence_ids: list[str]

    def to_dict(self) -> JSONDict:
        return asdict(self)


@dataclass
class RequestHigherResolutionCropParameters:
    target_detection_id: str
    frame_ref: str
    crop_ref: str
    reason_code: str

    def to_dict(self) -> JSONDict:
        return asdict(self)


@dataclass
class RequestMoreFramesParameters:
    target_detection_id: str
    anchor_frame_ref: str
    frames_before: int
    frames_after: int
    reason_code: str

    def to_dict(self) -> JSONDict:
        return asdict(self)


@dataclass
class AbstainParameters:
    reason_code: str
    invalid_refs: list[str] = field(default_factory=list)
    missing_inputs: list[str] = field(default_factory=list)

    def to_dict(self) -> JSONDict:
        return asdict(self)


ContextActionParameters = (
    EmitContextEvidenceParameters
    | RequestHigherResolutionCropParameters
    | RequestMoreFramesParameters
    | AbstainParameters
)


@dataclass
class ContextResult:
    request_id: str
    event_id: str
    worker_id: str
    evidence: list[ContextEvidence]
    selected_action: ContextAction
    action_parameters: ContextActionParameters
    context_confidence: float
    validation_errors: list[str] = field(default_factory=list)
    model_metadata: JSONDict = field(default_factory=dict)

    def to_dict(self) -> JSONDict:
        return {
            "request_id": self.request_id,
            "event_id": self.event_id,
            "worker_id": self.worker_id,
            "evidence": [item.to_dict() for item in self.evidence],
            "selected_action": self.selected_action.value,
            "action_parameters": self.action_parameters.to_dict(),
            "context_confidence": self.context_confidence,
            "validation_errors": self.validation_errors,
            "model_metadata": self.model_metadata,
        }


@dataclass
class CandidateEvent:
    event_id: str
    worker_id: str
    frame_ref: str | None
    crop_ref: str | None
    source_ref: str | None
    detections: list[Detection]
    ppe_status: PPEStatus | None
    region_grounding: RegionGrounding | None
    higher_resolution_source_ref: str | None = None
    context_evidence: list[ContextEvidence] = field(default_factory=list)
    neighbor_frame_refs: list[str] = field(default_factory=list)
    evidence_issues: list[EvidenceIssue] = field(default_factory=list)
    context_attempt_count: int = 0

    def to_dict(self) -> JSONDict:
        return {
            "event_id": self.event_id,
            "worker_id": self.worker_id,
            "frame_ref": self.frame_ref,
            "crop_ref": self.crop_ref,
            "source_ref": self.source_ref,
            "higher_resolution_source_ref": self.higher_resolution_source_ref,
            "detections": [item.to_dict() for item in self.detections],
            "ppe_status": self.ppe_status.to_dict() if self.ppe_status else None,
            "region_grounding": (
                self.region_grounding.to_dict() if self.region_grounding else None
            ),
            "context_evidence": [item.to_dict() for item in self.context_evidence],
            "neighbor_frame_refs": self.neighbor_frame_refs,
            "evidence_issues": [issue.value for issue in self.evidence_issues],
            "context_attempt_count": self.context_attempt_count,
        }


def is_valid_confirmed_context_evidence(
    candidate: CandidateEvent,
    evidence: ContextEvidence,
) -> bool:
    """Return whether Context evidence is safe to consume for this candidate."""
    status = (
        evidence.status.value
        if isinstance(evidence.status, ContextEvidenceStatus)
        else str(evidence.status)
    )
    if status != ContextEvidenceStatus.CONFIRMED.value:
        return False

    kind = (
        evidence.kind.value
        if isinstance(evidence.kind, ContextEvidenceKind)
        else str(evidence.kind)
    )
    labels_by_kind = {
        ContextEvidenceKind.VISUAL_QUALITY.value: VISUAL_QUALITY_LABELS,
        ContextEvidenceKind.LOCAL_RELATION.value: LOCAL_RELATION_LABELS,
        ContextEvidenceKind.INPUT_CONFIDENCE.value: INPUT_CONFIDENCE_LABELS,
    }
    if kind not in labels_by_kind or evidence.label not in labels_by_kind[kind]:
        return False

    detection_ids = {item.object_id for item in candidate.detections if item.object_id}
    if not isinstance(candidate.frame_ref, str) or not candidate.frame_ref.strip():
        return False
    if not isinstance(candidate.crop_ref, str) or not candidate.crop_ref.strip():
        return False
    if not isinstance(evidence.frame_ref, str) or not evidence.frame_ref.strip():
        return False
    if not isinstance(evidence.crop_ref, str) or not evidence.crop_ref.strip():
        return False
    if not isinstance(evidence.zone_ref, str) or not evidence.zone_ref.strip():
        return False
    if evidence.subject_detection_id != candidate.worker_id:
        return False
    if evidence.subject_detection_id not in detection_ids:
        return False
    if evidence.object_detection_id is not None and evidence.object_detection_id not in detection_ids:
        return False
    if kind == ContextEvidenceKind.LOCAL_RELATION.value and not evidence.object_detection_id:
        return False
    if evidence.frame_ref != candidate.frame_ref or evidence.crop_ref != candidate.crop_ref:
        return False
    if candidate.region_grounding is None or not candidate.region_grounding.zone_id:
        return False
    if evidence.zone_ref != candidate.region_grounding.zone_id:
        return False
    if (
        isinstance(evidence.confidence, bool)
        or not isinstance(evidence.confidence, (int, float))
        or not 0.0 <= float(evidence.confidence) <= 1.0
    ):
        return False
    return bool(evidence.evidence_id and evidence.reason_code)


def validated_confirmed_context_evidence(
    candidate: CandidateEvent,
) -> list[ContextEvidence]:
    return [
        item
        for item in candidate.context_evidence
        if is_valid_confirmed_context_evidence(candidate, item)
    ]


@dataclass
class EvidenceGateResult:
    route: EvidenceRoute
    reason_codes: list[EvidenceReasonCode]
    missing_fields: list[str] = field(default_factory=list)
    recoverable: bool = False
    allowed_context_actions: list[ContextAction] = field(default_factory=list)

    def to_dict(self) -> JSONDict:
        return {
            "route": self.route.value,
            "reason_codes": [code.value for code in self.reason_codes],
            "missing_fields": self.missing_fields,
            "recoverable": self.recoverable,
            "allowed_context_actions": [
                action.value for action in self.allowed_context_actions
            ],
        }

@dataclass
class RuleMatch:
    rule_id: str
    violation: bool
    severity: str
    confidence: float
    recommended_action: str
    reason: str
    uncertainty: str = ""
    reason_codes: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> JSONDict:
        return asdict(self)
