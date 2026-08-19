from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path

from src.safety_agents.contracts import (
    CandidateEvent,
    ContextEvidence,
    ContextEvidenceKind,
    ContextEvidenceStatus,
    Detection,
    EvidenceGateResult,
    EvidenceReasonCode,
    EvidenceRoute,
    PPEStatus,
    RegionGrounding,
)
from src.safety_agents.rule_severity_agent import (
    RuleConfigurationError,
    RuleInputNotReadyError,
    RuleSeverityAgent,
)


def load_rules() -> list[dict[str, object]]:
    path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "safety_agents"
        / "data"
        / "rules.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))["rules"]


def ready_gate_result() -> EvidenceGateResult:
    return EvidenceGateResult(
        route=EvidenceRoute.READY_FOR_RULE,
        reason_codes=[EvidenceReasonCode.EVIDENCE_SUFFICIENT],
    )


def build_candidate() -> CandidateEvent:
    return CandidateEvent(
        event_id="EVT-001",
        worker_id="W01",
        frame_ref="FRAME-001",
        crop_ref="FRAME-001::W01::100,80,220,420",
        source_ref="frame-001.jpg",
        detections=[
            Detection("W01", "person", [100, 80, 220, 420], 0.95),
            Detection("EQ01", "excavator", [250, 100, 480, 430], 0.91),
        ],
        ppe_status=PPEStatus(
            helmet="missing",
            confidence=0.92,
            target_id="W01",
            evidence_detection_ids=["W01"],
        ),
        region_grounding=RegionGrounding(
            zone_id="ZONE-ACTIVE-01",
            zone_type="active_work_area",
            spatial_relation="inside",
            confidence=0.98,
            target_id="W01",
        ),
    )


def confirmed_near_evidence(**overrides: object) -> ContextEvidence:
    evidence = ContextEvidence(
        evidence_id="CTX-NEAR-001",
        kind=ContextEvidenceKind.LOCAL_RELATION,
        label="NEAR",
        subject_detection_id="W01",
        object_detection_id="EQ01",
        frame_ref="FRAME-001",
        crop_ref="FRAME-001::W01::100,80,220,420",
        zone_ref="ZONE-ACTIVE-01",
        confidence=0.88,
        status=ContextEvidenceStatus.CONFIRMED,
        reason_code="LOCAL_RELATION_CONFIRMED",
    )
    return replace(evidence, **overrides)


class TestRuleSeverityAgent(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = RuleSeverityAgent(load_rules())

    def test_rule_requires_ready_gate(self) -> None:
        gate = EvidenceGateResult(
            route=EvidenceRoute.NEEDS_CONTEXT,
            reason_codes=[EvidenceReasonCode.RELATION_AMBIGUOUS],
            recoverable=True,
        )

        with self.assertRaisesRegex(
            RuleInputNotReadyError,
            "RULE_INPUT_NOT_READY:NEEDS_CONTEXT",
        ):
            self.agent.apply(build_candidate(), gate)

    def test_active_work_area_without_confirmed_near_stays_medium(self) -> None:
        result = self.agent.apply(build_candidate(), ready_gate_result())

        self.assertEqual(result.rule_id, "PPE_ACTIVE_ZONE_001")
        self.assertTrue(result.violation)
        self.assertEqual(result.severity, "medium")
        self.assertIn("NEAR_EQUIPMENT_RELATION_NOT_PROVIDED", result.missing_evidence)

    def test_valid_confirmed_near_equipment_can_upgrade_to_critical(self) -> None:
        candidate = replace(
            build_candidate(),
            context_evidence=[confirmed_near_evidence()],
        )

        result = self.agent.apply(candidate, ready_gate_result())

        self.assertEqual(result.severity, "critical")
        self.assertIn("CTX-NEAR-001", result.evidence_refs)
        self.assertIn("NEAR_HEAVY_EQUIPMENT_CONFIRMED", result.reason_codes)

    def test_invented_context_object_ref_is_ignored(self) -> None:
        candidate = replace(
            build_candidate(),
            context_evidence=[
                confirmed_near_evidence(object_detection_id="INVENTED-EQUIPMENT")
            ],
        )

        result = self.agent.apply(candidate, ready_gate_result())

        self.assertEqual(result.severity, "medium")
        self.assertNotIn("CTX-NEAR-001", result.evidence_refs)

    def test_unsupported_zone_fails_closed(self) -> None:
        candidate = build_candidate()
        candidate = replace(
            candidate,
            region_grounding=replace(
                candidate.region_grounding,
                zone_type="material_handling",
            ),
        )

        with self.assertRaisesRegex(
            RuleInputNotReadyError,
            "RULE_INPUT_NOT_READY:UNSUPPORTED_ZONE_TYPE:material_handling",
        ):
            self.agent.apply(candidate, ready_gate_result())

    def test_missing_catalog_entry_does_not_fabricate_rule(self) -> None:
        rules = [
            rule
            for rule in load_rules()
            if rule["rule_id"] != "PPE_ACTIVE_ZONE_001"
        ]
        agent = RuleSeverityAgent(rules)

        with self.assertRaisesRegex(
            RuleConfigurationError,
            "RULE_NOT_CONFIGURED:PPE_ACTIVE_ZONE_001",
        ):
            agent.apply(build_candidate(), ready_gate_result())


if __name__ == "__main__":
    unittest.main()
