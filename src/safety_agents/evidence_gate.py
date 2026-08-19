from __future__ import annotations

from .contracts import (
    CandidateEvent,
    ContextAction,
    ContextEvidenceKind,
    EvidenceGateResult,
    EvidenceIssue,
    EvidenceReasonCode,
    EvidenceRoute,
    SUPPORTED_ZONE_TYPES,
    validated_confirmed_context_evidence,
)


class EvidenceSufficiencyGate:
    """Deterministically routes normalized evidence without consulting a model."""

    def __init__(
        self,
        min_person_confidence: float = 0.5,
        min_ppe_confidence: float = 0.6,
        min_grounding_confidence: float = 0.8,
        max_context_attempts: int = 1,
    ) -> None:
        if max_context_attempts < 1:
            raise ValueError("max_context_attempts must be at least 1")
        self.min_person_confidence = min_person_confidence
        self.min_ppe_confidence = min_ppe_confidence
        self.min_grounding_confidence = min_grounding_confidence
        self.max_context_attempts = max_context_attempts

    def evaluate(self, candidate: CandidateEvent) -> EvidenceGateResult:
        fatal, missing = self._structural_failures(candidate)
        if fatal:
            return EvidenceGateResult(
                route=EvidenceRoute.UNRESOLVABLE,
                reason_codes=self._unique(fatal),
                missing_fields=list(dict.fromkeys(missing)),
                recoverable=False,
            )

        recoverable, unrecoverable, allowed_actions = self._sufficiency_failures(candidate)
        if unrecoverable:
            return EvidenceGateResult(
                route=EvidenceRoute.UNRESOLVABLE,
                reason_codes=self._unique(unrecoverable),
                missing_fields=[],
                recoverable=False,
                allowed_context_actions=[],
            )

        if recoverable:
            if candidate.context_attempt_count >= self.max_context_attempts:
                return EvidenceGateResult(
                    route=EvidenceRoute.UNRESOLVABLE,
                    reason_codes=self._unique(
                        [
                            *recoverable,
                            EvidenceReasonCode.CONTEXT_ATTEMPT_BUDGET_EXHAUSTED,
                        ]
                    ),
                    missing_fields=[],
                    recoverable=False,
                    allowed_context_actions=[],
                )
            return EvidenceGateResult(
                route=EvidenceRoute.NEEDS_CONTEXT,
                reason_codes=self._unique(recoverable),
                missing_fields=[],
                recoverable=True,
                allowed_context_actions=allowed_actions,
            )

        return EvidenceGateResult(
            route=EvidenceRoute.READY_FOR_RULE,
            reason_codes=[EvidenceReasonCode.EVIDENCE_SUFFICIENT],
            missing_fields=[],
            recoverable=False,
            allowed_context_actions=[],
        )

    def _structural_failures(
        self, candidate: CandidateEvent
    ) -> tuple[list[EvidenceReasonCode], list[str]]:
        reasons: list[EvidenceReasonCode] = []
        missing: list[str] = []

        required_refs = {
            "frame_ref": candidate.frame_ref,
            "crop_ref": candidate.crop_ref,
            "source_ref": candidate.source_ref,
        }
        ref_codes = {
            "frame_ref": EvidenceReasonCode.MISSING_FRAME_REF,
            "crop_ref": EvidenceReasonCode.MISSING_CROP_REF,
            "source_ref": EvidenceReasonCode.MISSING_SOURCE_REF,
        }
        for field_name, value in required_refs.items():
            if not isinstance(value, str) or not value.strip():
                reasons.append(ref_codes[field_name])
                missing.append(field_name)

        detection_ids = {item.object_id for item in candidate.detections if item.object_id}
        if candidate.worker_id not in detection_ids:
            reasons.append(EvidenceReasonCode.TARGET_NOT_IN_DETECTIONS)

        for detection in candidate.detections:
            if (
                not detection.object_id
                or not detection.class_label
                or not self._valid_confidence(detection.confidence)
                or not self._valid_bbox(detection.bbox)
            ):
                reasons.append(EvidenceReasonCode.INVALID_DETECTION)

        ppe = candidate.ppe_status
        grounding = candidate.region_grounding
        if ppe is None:
            reasons.append(EvidenceReasonCode.MISSING_PPE_STATUS)
            missing.append("ppe_status")
        if grounding is None:
            reasons.append(EvidenceReasonCode.MISSING_REGION_GROUNDING)
            missing.append("region_grounding")

        if ppe is not None:
            if ppe.target_id != candidate.worker_id:
                reasons.append(EvidenceReasonCode.TARGET_ID_MISMATCH)
            if not self._valid_confidence(ppe.confidence):
                reasons.append(EvidenceReasonCode.INVALID_PPE_CONFIDENCE)
            if any(ref not in detection_ids for ref in ppe.evidence_detection_ids):
                reasons.append(EvidenceReasonCode.PPE_EVIDENCE_REF_NOT_FOUND)

        if grounding is not None:
            if grounding.target_id != candidate.worker_id:
                reasons.append(EvidenceReasonCode.TARGET_ID_MISMATCH)
            if not self._valid_confidence(grounding.confidence):
                reasons.append(EvidenceReasonCode.INVALID_GROUNDING_CONFIDENCE)
            if not grounding.zone_id:
                reasons.append(EvidenceReasonCode.MISSING_REGION_GROUNDING)
                missing.append("region_grounding.zone_id")

        return self._unique(reasons), list(dict.fromkeys(missing))

    def _sufficiency_failures(
        self, candidate: CandidateEvent
    ) -> tuple[
        list[EvidenceReasonCode],
        list[EvidenceReasonCode],
        list[ContextAction],
    ]:
        ppe = candidate.ppe_status
        grounding = candidate.region_grounding
        if ppe is None or grounding is None:
            return [], [EvidenceReasonCode.MISSING_PPE_STATUS], []

        recoverable: list[EvidenceReasonCode] = []
        unrecoverable: list[EvidenceReasonCode] = []
        confidence_blockers: list[EvidenceReasonCode] = []
        ambiguity_reasons: list[EvidenceReasonCode] = []
        acquisition_actions: set[ContextAction] = set()

        if grounding.zone_id == "UNKNOWN" or grounding.zone_type == "unknown":
            unrecoverable.append(EvidenceReasonCode.UNKNOWN_ZONE)
        elif grounding.zone_type not in SUPPORTED_ZONE_TYPES:
            unrecoverable.append(EvidenceReasonCode.UNSUPPORTED_ZONE_TYPE)

        target_confidence = max(
            (
                item.confidence
                for item in candidate.detections
                if item.object_id == candidate.worker_id
            ),
            default=0.0,
        )
        if target_confidence < self.min_person_confidence:
            confidence_blockers.append(
                EvidenceReasonCode.TARGET_CONFIDENCE_BELOW_THRESHOLD
            )
        if ppe.confidence < self.min_ppe_confidence:
            confidence_blockers.append(EvidenceReasonCode.PPE_CONFIDENCE_BELOW_THRESHOLD)
        if grounding.confidence < self.min_grounding_confidence:
            confidence_blockers.append(
                EvidenceReasonCode.GROUNDING_CONFIDENCE_BELOW_THRESHOLD
            )
        if ppe.conflicts:
            confidence_blockers.append(EvidenceReasonCode.PPE_CONFLICT)
        if ppe.helmet in {"uncertain", "unknown"}:
            confidence_blockers.append(EvidenceReasonCode.PPE_STATE_UNCERTAIN)

        confirmed_relation = any(
            (
                item.kind == ContextEvidenceKind.LOCAL_RELATION.value
                or item.kind is ContextEvidenceKind.LOCAL_RELATION
            )
            and item.label not in {"RELATION_UNCLEAR", "BOUNDARY_STATE_UNCLEAR"}
            for item in validated_confirmed_context_evidence(candidate)
        )

        for issue in candidate.evidence_issues:
            if issue in {EvidenceIssue.LOW_RESOLUTION, EvidenceIssue.TARGET_TOO_SMALL}:
                if candidate.higher_resolution_source_ref:
                    ambiguity_reasons.append(EvidenceReasonCode.VISUAL_AMBIGUITY)
                    acquisition_actions.add(
                        ContextAction.REQUEST_HIGHER_RESOLUTION_CROP
                    )
                else:
                    unrecoverable.append(EvidenceReasonCode.MEDIA_NOT_RECOVERABLE)
            elif issue is EvidenceIssue.OCCLUDED:
                if candidate.neighbor_frame_refs:
                    ambiguity_reasons.append(EvidenceReasonCode.VISUAL_AMBIGUITY)
                    acquisition_actions.add(ContextAction.REQUEST_MORE_FRAMES)
                else:
                    unrecoverable.append(EvidenceReasonCode.MEDIA_NOT_RECOVERABLE)
            elif issue is EvidenceIssue.RELATION_UNCLEAR:
                if not confirmed_relation:
                    if len(candidate.detections) > 1:
                        ambiguity_reasons.append(EvidenceReasonCode.RELATION_AMBIGUOUS)
                    else:
                        unrecoverable.append(EvidenceReasonCode.MEDIA_NOT_RECOVERABLE)
            elif issue in {
                EvidenceIssue.TEMPORAL_CONTEXT_INSUFFICIENT,
                EvidenceIssue.BOUNDARY_STATE_UNCLEAR,
            }:
                if candidate.neighbor_frame_refs:
                    ambiguity_reasons.append(EvidenceReasonCode.TEMPORAL_AMBIGUITY)
                    acquisition_actions.add(ContextAction.REQUEST_MORE_FRAMES)
                else:
                    unrecoverable.append(EvidenceReasonCode.MEDIA_NOT_RECOVERABLE)
            elif issue in {EvidenceIssue.LOW_LIGHT, EvidenceIssue.FRAME_ERROR}:
                unrecoverable.append(EvidenceReasonCode.MEDIA_NOT_RECOVERABLE)

        if confidence_blockers:
            if acquisition_actions:
                recoverable.extend([*confidence_blockers, *ambiguity_reasons])
            else:
                unrecoverable.extend([*confidence_blockers, *ambiguity_reasons])
        else:
            recoverable.extend(ambiguity_reasons)

        allowed_actions: set[ContextAction] = set(acquisition_actions)
        if (
            recoverable == [EvidenceReasonCode.RELATION_AMBIGUOUS]
            and not acquisition_actions
        ):
            allowed_actions.add(ContextAction.EMIT_CONTEXT_EVIDENCE)

        if recoverable and not candidate.source_ref:
            unrecoverable.append(EvidenceReasonCode.MEDIA_NOT_RECOVERABLE)
            recoverable.clear()
            allowed_actions.clear()

        ordered_actions = [
            action
            for action in (
                ContextAction.REQUEST_HIGHER_RESOLUTION_CROP,
                ContextAction.REQUEST_MORE_FRAMES,
                ContextAction.EMIT_CONTEXT_EVIDENCE,
            )
            if action in allowed_actions
        ]
        if ordered_actions:
            ordered_actions.append(ContextAction.ABSTAIN)
        return (
            self._unique(recoverable),
            self._unique(unrecoverable),
            ordered_actions,
        )

    def _valid_confidence(self, value: object) -> bool:
        return (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and 0.0 <= float(value) <= 1.0
        )

    def _valid_bbox(self, bbox: object) -> bool:
        if not isinstance(bbox, list) or len(bbox) != 4:
            return False
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in bbox):
            return False
        x1, y1, x2, y2 = [float(value) for value in bbox]
        return x2 > x1 and y2 > y1

    def _unique(self, values: list[EvidenceReasonCode]) -> list[EvidenceReasonCode]:
        return list(dict.fromkeys(values))
