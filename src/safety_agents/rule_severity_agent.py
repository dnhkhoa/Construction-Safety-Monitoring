from __future__ import annotations

from typing import Any

from .contracts import (
    CandidateEvent,
    ContextEvidenceKind,
    EvidenceGateResult,
    EvidenceRoute,
    PPEStatus,
    PPE_EXEMPT_ZONE_TYPES,
    PPE_REQUIRED_ZONE_TYPES,
    RegionGrounding,
    RuleMatch,
    SUPPORTED_ZONE_TYPES,
    validated_confirmed_context_evidence,
)

EQUIPMENT_CLASSES = {
    "excavator",
    "truck",
    "crane",
    "loader",
    "heavy_equipment",
    "vehicle",
}


class RuleInputNotReadyError(ValueError):
    """Raised when the rule agent is called before the deterministic gate is ready."""


class RuleConfigurationError(RuleInputNotReadyError):
    """Raised when a controlled rule mapping is absent or malformed."""


class _HelmetZoneRulePolicy:
    PPE_REQUIRED_ZONES = set(PPE_REQUIRED_ZONE_TYPES)
    EXEMPT_ZONES = set(PPE_EXEMPT_ZONE_TYPES)

    def __init__(self, rules: list[dict[str, Any]] | dict[str, dict[str, Any]]) -> None:
        source = rules.values() if isinstance(rules, dict) else rules
        self.rules: dict[str, dict[str, Any]] = {}
        for rule in source:
            rule_id = rule.get("rule_id") if isinstance(rule, dict) else None
            if not isinstance(rule_id, str) or not rule_id.strip():
                raise RuleConfigurationError("RULE_NOT_CONFIGURED:MISSING_RULE_ID")
            if rule_id in self.rules:
                raise RuleConfigurationError(f"RULE_NOT_CONFIGURED:DUPLICATE:{rule_id}")
            self.rules[rule_id] = dict(rule)

    def evaluate(
        self,
        ppe_status: PPEStatus,
        grounding: RegionGrounding,
        near_heavy_equipment: bool | None,
        evidence_refs: list[str],
    ) -> RuleMatch:
        if ppe_status.helmet == "visible":
            return self._match(
                "SAFE_HELMET_VISIBLE_001",
                False,
                "none",
                min(1.0, ppe_status.confidence),
                evidence_refs=evidence_refs,
            )

        if ppe_status.helmet == "missing":
            return self._missing_helmet_match(
                ppe_status,
                grounding,
                near_heavy_equipment,
                evidence_refs,
            )

        if ppe_status.helmet in {"uncertain", "unknown"}:
            rule_id = (
                "PPE_UNCERTAIN_REVIEW_001"
                if grounding.zone_type in self.PPE_REQUIRED_ZONES
                else "PPE_UNCERTAIN_LOW_RISK_001"
            )
            return self._match(
                rule_id,
                False,
                "uncertain",
                min(ppe_status.confidence, grounding.confidence),
                uncertainty="PPE state is uncertain or conflicting.",
                reason_codes=["PPE_STATE_UNCERTAIN"],
                evidence_refs=evidence_refs,
                missing_evidence=["DETERMINATE_PPE_STATE"],
            )

        return self._match(
            "NO_MATCH_001",
            False,
            "none",
            0.0,
            reason_codes=["NO_APPLICABLE_RULE"],
            evidence_refs=evidence_refs,
        )

    def _missing_helmet_match(
        self,
        ppe_status: PPEStatus,
        grounding: RegionGrounding,
        near_heavy_equipment: bool | None,
        evidence_refs: list[str],
    ) -> RuleMatch:
        confidence = min(ppe_status.confidence, grounding.confidence)
        if grounding.zone_type in self.EXEMPT_ZONES:
            return self._match(
                "PPE_OFFICE_EXCEPTION_001",
                False,
                "none",
                confidence,
                evidence_refs=evidence_refs,
            )

        high_risk_rules = {
            "restricted_zone": "PPE_RESTRICTED_ZONE_001",
            "work_at_height": "PPE_HEIGHT_ZONE_001",
        }
        if grounding.zone_type in high_risk_rules:
            return self._match(
                high_risk_rules[grounding.zone_type],
                True,
                "critical",
                confidence,
                evidence_refs=evidence_refs,
            )

        if grounding.zone_type == "active_work_area":
            severity = "critical" if near_heavy_equipment is True else "medium"
            missing = [] if near_heavy_equipment is not None else [
                "NEAR_EQUIPMENT_RELATION_NOT_PROVIDED"
            ]
            uncertainty = (
                "Near-equipment relation was not provided; severity was not upgraded."
                if missing
                else ""
            )
            return self._match(
                "PPE_ACTIVE_ZONE_001",
                True,
                severity,
                confidence,
                uncertainty=uncertainty,
                reason_codes=[
                    "MISSING_HELMET_IN_ACTIVE_WORK_AREA",
                    *( ["NEAR_HEAVY_EQUIPMENT_CONFIRMED"] if severity == "critical" else [] ),
                ],
                evidence_refs=evidence_refs,
                missing_evidence=missing,
            )

        return self._match(
            "PPE_UNKNOWN_ZONE_001",
            False,
            "uncertain",
            ppe_status.confidence,
            uncertainty="Zone grounding is missing.",
            reason_codes=["UNKNOWN_ZONE"],
            evidence_refs=evidence_refs,
            missing_evidence=["GROUNDED_ZONE"],
        )

    def _match(
        self,
        rule_id: str,
        violation: bool,
        severity: str,
        confidence: float,
        uncertainty: str = "",
        reason_codes: list[str] | None = None,
        evidence_refs: list[str] | None = None,
        missing_evidence: list[str] | None = None,
    ) -> RuleMatch:
        configured = self.rules.get(rule_id, {})
        if not configured:
            raise RuleConfigurationError(f"RULE_NOT_CONFIGURED:{rule_id}")
        description = configured.get("description")
        configured_action = configured.get("recommended_action")
        if not isinstance(description, str) or not description.strip():
            raise RuleConfigurationError(f"RULE_DESCRIPTION_NOT_CONFIGURED:{rule_id}")
        if not isinstance(configured_action, str) or not configured_action.strip():
            raise RuleConfigurationError(f"RULE_ACTION_NOT_CONFIGURED:{rule_id}")
        return RuleMatch(
            rule_id=rule_id,
            violation=violation,
            severity=severity,
            confidence=round(confidence, 4),
            recommended_action=configured_action,
            reason=description,
            uncertainty=uncertainty,
            reason_codes=reason_codes or [rule_id],
            evidence_refs=list(dict.fromkeys(evidence_refs or [])),
            missing_evidence=missing_evidence or [],
        )


class RuleSeverityAgent:
    """Maps only gate-ready normalized evidence into a candidate rule match."""

    def __init__(self, rules: list[dict[str, Any]]) -> None:
        self._policy = _HelmetZoneRulePolicy(rules)
        self.rules = dict(self._policy.rules)

    def apply(self, candidate: CandidateEvent, gate_result: EvidenceGateResult) -> RuleMatch:
        if gate_result.route is not EvidenceRoute.READY_FOR_RULE:
            raise RuleInputNotReadyError(
                f"RULE_INPUT_NOT_READY:{gate_result.route.value}"
            )
        if candidate.ppe_status is None or candidate.region_grounding is None:
            raise RuleInputNotReadyError("RULE_INPUT_NOT_READY:MISSING_NORMALIZED_EVIDENCE")

        detection_ids = {item.object_id for item in candidate.detections}
        if (
            candidate.ppe_status.target_id != candidate.worker_id
            or candidate.region_grounding.target_id != candidate.worker_id
            or any(
                ref not in detection_ids
                for ref in candidate.ppe_status.evidence_detection_ids
            )
        ):
            raise RuleInputNotReadyError("RULE_INPUT_NOT_READY:INVALID_NORMALIZED_EVIDENCE")
        if candidate.region_grounding.zone_type not in SUPPORTED_ZONE_TYPES:
            raise RuleInputNotReadyError(
                "RULE_INPUT_NOT_READY:UNSUPPORTED_ZONE_TYPE:"
                f"{candidate.region_grounding.zone_type}"
            )
        if candidate.ppe_status.helmet not in {"visible", "missing"}:
            raise RuleInputNotReadyError("RULE_INPUT_NOT_READY:INDETERMINATE_PPE_STATE")

        confirmed = validated_confirmed_context_evidence(candidate)
        detection_by_id = {item.object_id: item for item in candidate.detections}
        near_equipment = any(
            (
                item.kind == ContextEvidenceKind.LOCAL_RELATION.value
                or item.kind is ContextEvidenceKind.LOCAL_RELATION
            )
            and item.label == "NEAR"
            and item.object_detection_id in detection_by_id
            and detection_by_id[item.object_detection_id].class_label.lower() in EQUIPMENT_CLASSES
            for item in confirmed
        )
        evidence_refs = [
            *candidate.ppe_status.evidence_detection_ids,
            candidate.region_grounding.zone_id,
            *(item.evidence_id for item in confirmed),
        ]
        return self._policy.evaluate(
            candidate.ppe_status,
            candidate.region_grounding,
            True if near_equipment else None,
            evidence_refs,
        )
