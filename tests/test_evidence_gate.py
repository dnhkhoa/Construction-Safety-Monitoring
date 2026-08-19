from __future__ import annotations

import unittest
from dataclasses import replace

from src.safety_agents.evidence_gate import EvidenceSufficiencyGate
from src.safety_agents.contracts import (
    CandidateEvent,
    ContextAction,
    ContextEvidence,
    ContextEvidenceKind,
    ContextEvidenceStatus,
    Detection,
    EvidenceIssue,
    EvidenceReasonCode,
    EvidenceRoute,
    PPEStatus,
    RegionGrounding,
)


def build_candidate(**overrides: object) -> CandidateEvent:
    candidate = CandidateEvent(
        event_id="EVT-001",
        worker_id="W01",
        frame_ref="F001",
        crop_ref="F001::W01::100,100,180,320",
        source_ref="frame-F001.jpg",
        higher_resolution_source_ref=None,
        detections=[
            Detection("W01", "person", [100, 100, 180, 320], 0.94),
            Detection("EQ01", "excavator", [220, 100, 420, 340], 0.88),
        ],
        ppe_status=PPEStatus(
            helmet="missing",
            confidence=0.9,
            target_id="W01",
            evidence_detection_ids=["W01"],
        ),
        region_grounding=RegionGrounding(
            zone_id="Z-ACTIVE-01",
            zone_type="active_work_area",
            spatial_relation="inside",
            confidence=0.98,
            target_id="W01",
        ),
    )
    return replace(candidate, **overrides)


def confirmed_near_evidence() -> ContextEvidence:
    return ContextEvidence(
        evidence_id="CTXE-NEAR-001",
        kind=ContextEvidenceKind.LOCAL_RELATION,
        label="NEAR",
        subject_detection_id="W01",
        object_detection_id="EQ01",
        frame_ref="F001",
        crop_ref="F001::W01::100,100,180,320",
        zone_ref="Z-ACTIVE-01",
        confidence=0.9,
        status=ContextEvidenceStatus.CONFIRMED,
        reason_code="NEAR_CONFIRMED",
    )


class TestEvidenceSufficiencyGate(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = EvidenceSufficiencyGate(max_context_attempts=1)

    def test_clear_complete_candidate_is_ready_for_rule(self) -> None:
        result = self.gate.evaluate(build_candidate())

        self.assertEqual(result.route, EvidenceRoute.READY_FOR_RULE)
        self.assertEqual(result.reason_codes, [EvidenceReasonCode.EVIDENCE_SUFFICIENT])
        self.assertEqual(result.missing_fields, [])
        self.assertFalse(result.recoverable)

    def test_relation_ambiguity_is_recoverable_with_existing_object(self) -> None:
        result = self.gate.evaluate(
            build_candidate(evidence_issues=[EvidenceIssue.RELATION_UNCLEAR])
        )

        self.assertEqual(result.route, EvidenceRoute.NEEDS_CONTEXT)
        self.assertIn(EvidenceReasonCode.RELATION_AMBIGUOUS, result.reason_codes)
        self.assertTrue(result.recoverable)
        self.assertEqual(
            result.allowed_context_actions,
            [ContextAction.EMIT_CONTEXT_EVIDENCE, ContextAction.ABSTAIN],
        )

    def test_confirmed_relation_resolves_relation_ambiguity(self) -> None:
        result = self.gate.evaluate(
            build_candidate(
                evidence_issues=[EvidenceIssue.RELATION_UNCLEAR],
                context_evidence=[confirmed_near_evidence()],
            )
        )

        self.assertEqual(result.route, EvidenceRoute.READY_FOR_RULE)

    def test_relation_with_mismatched_candidate_refs_does_not_resolve_ambiguity(self) -> None:
        invalid = replace(confirmed_near_evidence(), crop_ref="INVENTED-CROP")

        result = self.gate.evaluate(
            build_candidate(
                evidence_issues=[EvidenceIssue.RELATION_UNCLEAR],
                context_evidence=[invalid],
            )
        )

        self.assertEqual(result.route, EvidenceRoute.NEEDS_CONTEXT)
        self.assertIn(EvidenceReasonCode.RELATION_AMBIGUOUS, result.reason_codes)

    def test_unknown_target_is_unresolvable(self) -> None:
        result = self.gate.evaluate(build_candidate(worker_id="UNKNOWN"))

        self.assertEqual(result.route, EvidenceRoute.UNRESOLVABLE)
        self.assertIn(EvidenceReasonCode.TARGET_NOT_IN_DETECTIONS, result.reason_codes)
        self.assertFalse(result.recoverable)

    def test_missing_ppe_reference_is_unresolvable(self) -> None:
        bad_status = replace(
            build_candidate().ppe_status,
            evidence_detection_ids=["MISSING-DETECTION"],
        )

        result = self.gate.evaluate(build_candidate(ppe_status=bad_status))

        self.assertEqual(result.route, EvidenceRoute.UNRESOLVABLE)
        self.assertIn(EvidenceReasonCode.PPE_EVIDENCE_REF_NOT_FOUND, result.reason_codes)

    def test_target_mismatch_is_unresolvable(self) -> None:
        bad_grounding = replace(build_candidate().region_grounding, target_id="W02")

        result = self.gate.evaluate(build_candidate(region_grounding=bad_grounding))

        self.assertEqual(result.route, EvidenceRoute.UNRESOLVABLE)
        self.assertIn(EvidenceReasonCode.TARGET_ID_MISMATCH, result.reason_codes)

    def test_low_resolution_needs_explicit_higher_resolution_source(self) -> None:
        without_source = self.gate.evaluate(
            build_candidate(evidence_issues=[EvidenceIssue.LOW_RESOLUTION])
        )
        with_source = self.gate.evaluate(
            build_candidate(
                evidence_issues=[EvidenceIssue.LOW_RESOLUTION],
                higher_resolution_source_ref="frame-F001-original.jpg",
            )
        )

        self.assertEqual(without_source.route, EvidenceRoute.UNRESOLVABLE)
        self.assertIn(EvidenceReasonCode.MEDIA_NOT_RECOVERABLE, without_source.reason_codes)
        self.assertEqual(with_source.route, EvidenceRoute.NEEDS_CONTEXT)
        self.assertEqual(
            with_source.allowed_context_actions,
            [ContextAction.REQUEST_HIGHER_RESOLUTION_CROP, ContextAction.ABSTAIN],
        )

    def test_low_confidence_without_acquisition_path_is_unresolvable(self) -> None:
        low_confidence_target = replace(
            build_candidate().detections[0],
            confidence=0.49,
        )

        result = self.gate.evaluate(
            build_candidate(
                detections=[low_confidence_target, build_candidate().detections[1]],
            )
        )

        self.assertEqual(result.route, EvidenceRoute.UNRESOLVABLE)
        self.assertIn(
            EvidenceReasonCode.TARGET_CONFIDENCE_BELOW_THRESHOLD,
            result.reason_codes,
        )
        self.assertEqual(result.allowed_context_actions, [])

    def test_ppe_conflict_without_acquisition_path_is_unresolvable(self) -> None:
        conflicted = replace(
            build_candidate().ppe_status,
            helmet="uncertain",
            conflicts=["helmet", "no_helmet"],
        )

        result = self.gate.evaluate(build_candidate(ppe_status=conflicted))

        self.assertEqual(result.route, EvidenceRoute.UNRESOLVABLE)
        self.assertIn(EvidenceReasonCode.PPE_CONFLICT, result.reason_codes)
        self.assertIn(EvidenceReasonCode.PPE_STATE_UNCERTAIN, result.reason_codes)
        self.assertEqual(result.allowed_context_actions, [])

    def test_low_confidence_with_recoverable_crop_can_only_request_crop(self) -> None:
        low_confidence_target = replace(
            build_candidate().detections[0],
            confidence=0.49,
        )

        result = self.gate.evaluate(
            build_candidate(
                detections=[low_confidence_target, build_candidate().detections[1]],
                evidence_issues=[EvidenceIssue.LOW_RESOLUTION],
                higher_resolution_source_ref="frame-F001-original.jpg",
            )
        )

        self.assertEqual(result.route, EvidenceRoute.NEEDS_CONTEXT)
        self.assertIn(
            EvidenceReasonCode.TARGET_CONFIDENCE_BELOW_THRESHOLD,
            result.reason_codes,
        )
        self.assertEqual(
            result.allowed_context_actions,
            [ContextAction.REQUEST_HIGHER_RESOLUTION_CROP, ContextAction.ABSTAIN],
        )

    def test_occlusion_needs_neighbor_frames(self) -> None:
        without_frames = self.gate.evaluate(
            build_candidate(evidence_issues=[EvidenceIssue.OCCLUDED])
        )
        with_frames = self.gate.evaluate(
            build_candidate(
                evidence_issues=[EvidenceIssue.OCCLUDED],
                neighbor_frame_refs=["F000", "F002"],
            )
        )

        self.assertEqual(without_frames.route, EvidenceRoute.UNRESOLVABLE)
        self.assertEqual(with_frames.route, EvidenceRoute.NEEDS_CONTEXT)

    def test_attempt_budget_exhaustion_is_unresolvable(self) -> None:
        result = self.gate.evaluate(
            build_candidate(
                evidence_issues=[EvidenceIssue.RELATION_UNCLEAR],
                context_attempt_count=1,
            )
        )

        self.assertEqual(result.route, EvidenceRoute.UNRESOLVABLE)
        self.assertIn(
            EvidenceReasonCode.CONTEXT_ATTEMPT_BUDGET_EXHAUSTED,
            result.reason_codes,
        )


if __name__ == "__main__":
    unittest.main()
