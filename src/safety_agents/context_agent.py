from __future__ import annotations

from dataclasses import replace
from typing import Protocol

from .contracts import (
    AbstainParameters,
    ContextAction,
    ContextActionParameters,
    ContextEvidence,
    ContextEvidenceKind,
    ContextEvidenceStatus,
    ContextProposal,
    ContextRequest,
    ContextResult,
    EmitContextEvidenceParameters,
    EvidenceIssue,
    INPUT_CONFIDENCE_LABELS,
    LOCAL_RELATION_LABELS,
    RequestHigherResolutionCropParameters,
    RequestMoreFramesParameters,
    VISUAL_QUALITY_LABELS,
)


class ContextModelAdapter(Protocol):
    """Provider-neutral boundary for a future VLM implementation."""

    def analyze(self, request: ContextRequest) -> ContextProposal:
        """Return structured context evidence and one proposed action."""


class UnconfiguredContextModelAdapter:
    """Fail-safe adapter used until a concrete VLM provider is configured."""

    def analyze(self, request: ContextRequest) -> ContextProposal:
        return ContextProposal(
            evidence=[],
            selected_action=ContextAction.ABSTAIN,
            action_parameters={"reason_code": "MODEL_NOT_CONFIGURED"},
            context_confidence=0.0,
            model_metadata={"adapter": "unconfigured"},
        )


class ContextAgent:
    """Validates one candidate event and emits bounded context evidence/actions."""

    CROP_QUALITY_LABELS = {
        "LOW_RESOLUTION",
        "TARGET_TOO_SMALL",
    }
    FRAME_REQUEST_LABELS = {
        "TEMPORAL_CONTEXT_INSUFFICIENT",
        "OCCLUDED",
        "BOUNDARY_STATE_UNCLEAR",
    }
    EMIT_BLOCKING_LABELS = {
        "LOW_RESOLUTION",
        "TARGET_TOO_SMALL",
        "BLUR",
        "LOW_LIGHT",
        "OCCLUDED",
        "OUT_OF_FRAME",
        "FRAME_ERROR",
        "TEMPORAL_CONTEXT_INSUFFICIENT",
        "UPSTREAM_CONFIDENCE_LOW",
        "UPSTREAM_CONFIDENCE_INVALID",
        "RELATION_UNCLEAR",
        "BOUNDARY_STATE_UNCLEAR",
    }
    TERMINAL_VISUAL_LABELS = {"OUT_OF_FRAME", "FRAME_ERROR"}

    def __init__(self, model_adapter: ContextModelAdapter | None = None) -> None:
        self.model_adapter = model_adapter or UnconfiguredContextModelAdapter()

    def analyze(self, request: ContextRequest) -> ContextResult:
        if (
            request.max_context_attempts < 1
            or request.context_attempt_count >= request.max_context_attempts
        ):
            return self._abstain(
                request,
                reason_code="CONTEXT_ATTEMPT_BUDGET_EXHAUSTED",
                validation_errors=["CONTEXT_ATTEMPT_BUDGET_EXHAUSTED"],
            )

        request_errors = self._validate_request(request)
        if request_errors:
            return self._abstain(
                request,
                reason_code="REQUEST_VALIDATION_FAILED",
                validation_errors=request_errors,
            )

        try:
            proposal = self.model_adapter.analyze(request)
        except Exception as error:  # provider adapters are an untrusted boundary
            return self._abstain(
                request,
                reason_code="MODEL_ADAPTER_ERROR",
                validation_errors=[f"MODEL_ADAPTER_ERROR:{type(error).__name__}"],
            )

        if not isinstance(proposal, ContextProposal):
            return self._abstain(
                request,
                reason_code="INVALID_MODEL_PROPOSAL",
                validation_errors=[f"INVALID_MODEL_PROPOSAL:{type(proposal).__name__}"],
            )

        action, action_errors = self._normalize_action(proposal.selected_action)
        evidence, evidence_errors = self._validate_evidence(request, proposal.evidence)
        proposal_errors = [*action_errors, *evidence_errors]

        if (
            action is not None
            and action is not ContextAction.ABSTAIN
            and action not in request.allowed_context_actions
        ):
            proposal_errors.append(f"ACTION_NOT_ALLOWED_FOR_ROUTE:{action.value}")

        if not self._is_confidence(proposal.context_confidence):
            proposal_errors.append("INVALID_CONTEXT_CONFIDENCE")
        if not isinstance(proposal.action_parameters, dict):
            proposal_errors.append("INVALID_ACTION_PARAMETERS")
        if not isinstance(proposal.model_metadata, dict):
            proposal_errors.append("INVALID_MODEL_METADATA")

        if action is None or proposal_errors:
            return self._abstain(
                request,
                reason_code="PROPOSAL_VALIDATION_FAILED",
                validation_errors=proposal_errors,
                model_metadata=proposal.model_metadata
                if isinstance(proposal.model_metadata, dict)
                else {},
            )

        consistency_errors = self._validate_action_consistency(request, action, evidence)
        action_parameters, parameter_errors = self._build_action_parameters(
            request,
            action,
            evidence,
            proposal.action_parameters,
        )
        all_action_errors = [*consistency_errors, *parameter_errors]
        if all_action_errors or action_parameters is None:
            return self._abstain(
                request,
                reason_code="ACTION_VALIDATION_FAILED",
                validation_errors=all_action_errors,
                evidence=evidence,
                model_metadata=proposal.model_metadata,
            )

        if action in {
            ContextAction.REQUEST_HIGHER_RESOLUTION_CROP,
            ContextAction.REQUEST_MORE_FRAMES,
            ContextAction.ABSTAIN,
        }:
            evidence = [
                replace(item, status=ContextEvidenceStatus.PROVISIONAL.value)
                for item in evidence
            ]

        return ContextResult(
            request_id=request.request_id,
            event_id=request.event_id,
            worker_id=request.worker_id,
            evidence=evidence,
            selected_action=action,
            action_parameters=action_parameters,
            context_confidence=round(float(proposal.context_confidence), 4),
            validation_errors=[],
            model_metadata=dict(proposal.model_metadata),
        )

    def _validate_request(self, request: ContextRequest) -> list[str]:
        errors: list[str] = []
        required_text = {
            "REQUEST_ID": request.request_id,
            "EVENT_ID": request.event_id,
            "WORKER_ID": request.worker_id,
            "FRAME_REF": request.frame_ref,
            "CROP_REF": request.crop_ref,
        }
        for name, value in required_text.items():
            if not isinstance(value, str) or not value.strip():
                errors.append(f"MISSING_{name}")

        if not request.detections:
            errors.append("NO_DETECTIONS")
            detection_ids: set[str] = set()
        else:
            detection_ids = {item.object_id for item in request.detections}

        if request.worker_id and request.worker_id not in detection_ids:
            errors.append(f"UNKNOWN_TARGET_DETECTION_REF:{request.worker_id}")

        for detection in request.detections:
            if not detection.object_id:
                errors.append("MISSING_DETECTION_ID")
            if not self._is_confidence(detection.confidence):
                errors.append(f"INVALID_DETECTION_CONFIDENCE:{detection.object_id}")

        for evidence_id in request.ppe_finding.evidence_detection_ids:
            if evidence_id not in detection_ids:
                errors.append(f"UNKNOWN_PPE_DETECTION_REF:{evidence_id}")

        if not self._is_confidence(request.ppe_finding.confidence):
            errors.append("INVALID_PPE_CONFIDENCE")
        if not self._is_confidence(request.zone_grounding.confidence):
            errors.append("INVALID_ZONE_CONFIDENCE")
        if not request.zone_grounding.zone_id:
            errors.append("MISSING_ZONE_REF")
        if request.ppe_finding.target_id != request.worker_id:
            errors.append(f"PPE_TARGET_MISMATCH:{request.ppe_finding.target_id}")
        if request.zone_grounding.target_id != request.worker_id:
            errors.append(f"ZONE_TARGET_MISMATCH:{request.zone_grounding.target_id}")

        if request.higher_resolution_source_ref is not None and (
            not isinstance(request.higher_resolution_source_ref, str)
            or not request.higher_resolution_source_ref.strip()
        ):
            errors.append("INVALID_HIGHER_RESOLUTION_SOURCE_REF")

        if any(not isinstance(ref, str) or not ref.strip() for ref in request.neighbor_frame_refs):
            errors.append("INVALID_NEIGHBOR_FRAME_REF")
        if not isinstance(request.allowed_context_actions, list):
            errors.append("INVALID_ALLOWED_CONTEXT_ACTIONS")
        else:
            for action in request.allowed_context_actions:
                if not isinstance(action, ContextAction):
                    errors.append(f"INVALID_ALLOWED_CONTEXT_ACTION:{action}")
        return list(dict.fromkeys(errors))

    def _normalize_action(
        self, action: ContextAction | str
    ) -> tuple[ContextAction | None, list[str]]:
        raw_action = action.value if isinstance(action, ContextAction) else str(action)
        try:
            return ContextAction(raw_action), []
        except ValueError:
            return None, [f"UNSUPPORTED_ACTION:{raw_action}"]

    def _validate_evidence(
        self,
        request: ContextRequest,
        proposed_evidence: list[ContextEvidence],
    ) -> tuple[list[ContextEvidence], list[str]]:
        if not isinstance(proposed_evidence, list):
            return [], ["INVALID_EVIDENCE_COLLECTION"]

        errors: list[str] = []
        normalized: list[ContextEvidence] = []
        evidence_ids: set[str] = set()
        detection_ids = {item.object_id for item in request.detections}
        for item in proposed_evidence:
            if not isinstance(item, ContextEvidence):
                errors.append("INVALID_EVIDENCE_ITEM")
                continue

            kind = item.kind.value if isinstance(item.kind, ContextEvidenceKind) else str(item.kind)
            status = (
                item.status.value
                if isinstance(item.status, ContextEvidenceStatus)
                else str(item.status)
            )

            if not isinstance(item.evidence_id, str) or not item.evidence_id:
                errors.append("MISSING_EVIDENCE_ID")
            elif item.evidence_id in evidence_ids:
                errors.append(f"DUPLICATE_EVIDENCE_ID:{item.evidence_id}")
            else:
                evidence_ids.add(item.evidence_id)

            if kind not in {value.value for value in ContextEvidenceKind}:
                errors.append(f"UNSUPPORTED_EVIDENCE_KIND:{kind}")
            elif not isinstance(item.label, str) or not self._label_matches_kind(kind, item.label):
                errors.append(f"UNSUPPORTED_EVIDENCE_LABEL:{kind}:{item.label}")

            if status not in {value.value for value in ContextEvidenceStatus}:
                errors.append(f"UNSUPPORTED_EVIDENCE_STATUS:{status}")
            if (
                not isinstance(item.subject_detection_id, str)
                or item.subject_detection_id not in detection_ids
            ):
                errors.append(f"UNKNOWN_SUBJECT_DETECTION_REF:{item.subject_detection_id}")
            if item.subject_detection_id != request.worker_id:
                errors.append(f"SUBJECT_MUST_MATCH_WORKER:{item.subject_detection_id}")
            if item.object_detection_id is not None and (
                not isinstance(item.object_detection_id, str)
                or item.object_detection_id not in detection_ids
            ):
                errors.append(f"UNKNOWN_OBJECT_DETECTION_REF:{item.object_detection_id}")
            if kind == ContextEvidenceKind.LOCAL_RELATION.value and not item.object_detection_id:
                errors.append(f"LOCAL_RELATION_REQUIRES_OBJECT:{item.evidence_id}")
            if not isinstance(item.frame_ref, str) or item.frame_ref != request.frame_ref:
                errors.append(f"FRAME_REF_MISMATCH:{item.frame_ref}")
            if not isinstance(item.crop_ref, str) or item.crop_ref != request.crop_ref:
                errors.append(f"CROP_REF_MISMATCH:{item.crop_ref}")
            if not isinstance(item.zone_ref, str) or item.zone_ref != request.zone_grounding.zone_id:
                errors.append(f"ZONE_REF_MISMATCH:{item.zone_ref}")
            if not self._is_confidence(item.confidence):
                errors.append(f"INVALID_EVIDENCE_CONFIDENCE:{item.evidence_id}")
            if not isinstance(item.reason_code, str) or not item.reason_code:
                errors.append(f"MISSING_EVIDENCE_REASON_CODE:{item.evidence_id}")

            normalized.append(replace(item, kind=kind, status=status))

        return normalized, list(dict.fromkeys(errors))

    def _label_matches_kind(self, kind: str, label: str) -> bool:
        labels_by_kind = {
            ContextEvidenceKind.VISUAL_QUALITY.value: VISUAL_QUALITY_LABELS,
            ContextEvidenceKind.LOCAL_RELATION.value: LOCAL_RELATION_LABELS,
            ContextEvidenceKind.INPUT_CONFIDENCE.value: INPUT_CONFIDENCE_LABELS,
        }
        return label in labels_by_kind.get(kind, frozenset())

    def _validate_action_consistency(
        self,
        request: ContextRequest,
        action: ContextAction,
        evidence: list[ContextEvidence],
    ) -> list[str]:
        errors: list[str] = []
        labels = {item.label for item in evidence}

        terminal_labels = labels & self.TERMINAL_VISUAL_LABELS
        if action is not ContextAction.ABSTAIN and terminal_labels:
            errors.extend(
                f"BLOCKING_VISUAL_QUALITY:{label}" for label in sorted(terminal_labels)
            )

        if action is ContextAction.EMIT_CONTEXT_EVIDENCE:
            confirmed = [
                item
                for item in evidence
                if item.status == ContextEvidenceStatus.CONFIRMED.value
            ]
            if not confirmed:
                errors.append("EMIT_REQUIRES_CONFIRMED_EVIDENCE")
            for label in sorted(labels & self.EMIT_BLOCKING_LABELS):
                error = f"BLOCKING_VISUAL_QUALITY:{label}"
                if error not in errors:
                    errors.append(error)
            resolving_relations = [
                item
                for item in confirmed
                if item.kind == ContextEvidenceKind.LOCAL_RELATION.value
                and item.label not in {"RELATION_UNCLEAR", "BOUNDARY_STATE_UNCLEAR"}
            ]
            if (
                EvidenceIssue.RELATION_UNCLEAR not in request.evidence_issues
                or not resolving_relations
            ):
                errors.append("EMIT_DOES_NOT_RESOLVE_ROUTE")

        if action is ContextAction.REQUEST_HIGHER_RESOLUTION_CROP:
            if not labels & self.CROP_QUALITY_LABELS:
                errors.append("CROP_REQUEST_REQUIRES_RECOVERABLE_QUALITY")
            if not request.higher_resolution_source_ref:
                errors.append("HIGHER_RESOLUTION_SOURCE_UNAVAILABLE")

        if action is ContextAction.REQUEST_MORE_FRAMES:
            if not labels & self.FRAME_REQUEST_LABELS:
                errors.append("FRAME_REQUEST_REQUIRES_RECOVERABLE_EVIDENCE")
            if not request.neighbor_frame_refs:
                errors.append("NEIGHBOR_FRAMES_UNAVAILABLE")

        return errors

    def _build_action_parameters(
        self,
        request: ContextRequest,
        action: ContextAction,
        evidence: list[ContextEvidence],
        raw_parameters: dict[str, object],
    ) -> tuple[ContextActionParameters | None, list[str]]:
        errors: list[str] = []
        provided_target = raw_parameters.get("target_detection_id")
        if provided_target is not None and provided_target != request.worker_id:
            errors.append(f"ACTION_TARGET_MISMATCH:{provided_target}")

        provided_reason = raw_parameters.get("reason_code")
        if provided_reason is not None and (
            not isinstance(provided_reason, str) or not provided_reason.strip()
        ):
            errors.append("INVALID_ACTION_REASON_CODE")

        if action is ContextAction.EMIT_CONTEXT_EVIDENCE:
            confirmed_ids = [
                item.evidence_id
                for item in evidence
                if item.status == ContextEvidenceStatus.CONFIRMED.value
            ]
            raw_ids = raw_parameters.get("evidence_ids", confirmed_ids)
            if not isinstance(raw_ids, list) or any(not isinstance(item, str) for item in raw_ids):
                return None, ["INVALID_EMIT_EVIDENCE_IDS"]
            unknown_ids = [item for item in raw_ids if item not in confirmed_ids]
            if unknown_ids:
                errors.extend(f"UNCONFIRMED_EMIT_EVIDENCE_REF:{item}" for item in unknown_ids)
            if not raw_ids:
                errors.append("EMPTY_EMIT_EVIDENCE_IDS")
            if len(raw_ids) != len(set(raw_ids)):
                errors.append("DUPLICATE_EMIT_EVIDENCE_IDS")
            selected = [item for item in evidence if item.evidence_id in raw_ids]
            if not any(
                item.kind == ContextEvidenceKind.LOCAL_RELATION.value
                and item.label not in {"RELATION_UNCLEAR", "BOUNDARY_STATE_UNCLEAR"}
                for item in selected
            ):
                errors.append("EMIT_SELECTION_DOES_NOT_RESOLVE_ROUTE")
            return EmitContextEvidenceParameters(evidence_ids=list(raw_ids)), errors

        reason_code = self._reason_code(raw_parameters, evidence)
        if action is ContextAction.REQUEST_HIGHER_RESOLUTION_CROP:
            self._validate_parameter_ref(
                raw_parameters,
                "frame_ref",
                request.frame_ref,
                "ACTION_FRAME_REF_MISMATCH",
                errors,
            )
            self._validate_parameter_ref(
                raw_parameters,
                "crop_ref",
                request.crop_ref,
                "ACTION_CROP_REF_MISMATCH",
                errors,
            )
            self._validate_parameter_ref(
                raw_parameters,
                "higher_resolution_source_ref",
                request.higher_resolution_source_ref,
                "ACTION_HIGHER_RESOLUTION_SOURCE_REF_MISMATCH",
                errors,
            )
            return (
                RequestHigherResolutionCropParameters(
                    target_detection_id=request.worker_id,
                    frame_ref=request.frame_ref,
                    crop_ref=request.crop_ref,
                    reason_code=reason_code or "VISUAL_DETAIL_INSUFFICIENT",
                ),
                errors,
            )

        if action is ContextAction.REQUEST_MORE_FRAMES:
            self._validate_parameter_ref(
                raw_parameters,
                "anchor_frame_ref",
                request.frame_ref,
                "ACTION_ANCHOR_FRAME_REF_MISMATCH",
                errors,
            )
            frames_before = self._frame_count(raw_parameters.get("frames_before", 2))
            frames_after = self._frame_count(raw_parameters.get("frames_after", 2))
            if frames_before is None:
                errors.append("INVALID_FRAMES_BEFORE")
                frames_before = 2
            if frames_after is None:
                errors.append("INVALID_FRAMES_AFTER")
                frames_after = 2
            return (
                RequestMoreFramesParameters(
                    target_detection_id=request.worker_id,
                    anchor_frame_ref=request.frame_ref,
                    frames_before=frames_before,
                    frames_after=frames_after,
                    reason_code=reason_code or "TEMPORAL_CONTEXT_INSUFFICIENT",
                ),
                errors,
            )

        return (
            AbstainParameters(reason_code=reason_code or "CONTEXT_MODEL_ABSTAINED"),
            errors,
        )

    def _validate_parameter_ref(
        self,
        raw_parameters: dict[str, object],
        field_name: str,
        expected: str | None,
        error_code: str,
        errors: list[str],
    ) -> None:
        provided = raw_parameters.get(field_name)
        if provided is not None and provided != expected:
            errors.append(f"{error_code}:{provided}")

    def _reason_code(
        self,
        raw_parameters: dict[str, object],
        evidence: list[ContextEvidence],
    ) -> str:
        raw_reason = raw_parameters.get("reason_code")
        if isinstance(raw_reason, str) and raw_reason.strip():
            return raw_reason.strip()
        if evidence:
            return evidence[0].reason_code
        return ""

    def _frame_count(self, value: object) -> int | None:
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        if not 1 <= value <= 10:
            return None
        return value

    def _is_confidence(self, value: object) -> bool:
        return (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and 0.0 <= float(value) <= 1.0
        )

    def _abstain(
        self,
        request: ContextRequest,
        reason_code: str,
        validation_errors: list[str],
        evidence: list[ContextEvidence] | None = None,
        model_metadata: dict[str, object] | None = None,
    ) -> ContextResult:
        invalid_refs = [
            error
            for error in validation_errors
            if "REF" in error or "TARGET_DETECTION" in error
        ]
        missing_inputs = [
            error
            for error in validation_errors
            if error.startswith("MISSING_") or error == "NO_DETECTIONS"
        ]
        return ContextResult(
            request_id=request.request_id,
            event_id=request.event_id,
            worker_id=request.worker_id,
            evidence=[
                replace(item, status=ContextEvidenceStatus.PROVISIONAL.value)
                for item in (evidence or [])
            ],
            selected_action=ContextAction.ABSTAIN,
            action_parameters=AbstainParameters(
                reason_code=reason_code,
                invalid_refs=invalid_refs,
                missing_inputs=missing_inputs,
            ),
            context_confidence=0.0,
            validation_errors=list(dict.fromkeys(validation_errors)),
            model_metadata=dict(model_metadata or {}),
        )
