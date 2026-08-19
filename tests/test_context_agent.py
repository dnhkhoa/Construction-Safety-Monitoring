from __future__ import annotations

import unittest
from dataclasses import replace

from src.safety_agents.context_agent import ContextAgent
from src.safety_agents.contracts import (
    AbstainParameters,
    ContextAction,
    ContextEvidence,
    ContextEvidenceKind,
    ContextEvidenceStatus,
    ContextProposal,
    ContextRequest,
    Detection,
    EmitContextEvidenceParameters,
    EvidenceIssue,
    PPEStatus,
    RegionGrounding,
    RequestHigherResolutionCropParameters,
    RequestMoreFramesParameters,
)


class FakeContextModelAdapter:
    def __init__(self, proposal: ContextProposal) -> None:
        self.proposal = proposal
        self.call_count = 0

    def analyze(self, request: ContextRequest) -> ContextProposal:
        self.call_count += 1
        return self.proposal


class FailingContextModelAdapter:
    def analyze(self, request: ContextRequest) -> ContextProposal:
        raise RuntimeError("model unavailable")


class InvalidContextModelAdapter:
    def analyze(self, request: ContextRequest) -> object:
        return {"selected_action": "EMIT_CONTEXT_EVIDENCE"}


def build_request(
    allowed_context_actions: list[ContextAction] | None = None,
    evidence_issues: list[EvidenceIssue] | None = None,
) -> ContextRequest:
    return ContextRequest(
        request_id="CTXREQ-001",
        event_id="EVT-001",
        worker_id="W01",
        frame_ref="F001",
        crop_ref="CROP-W01-F001",
        higher_resolution_source_ref="F001-original.jpg",
        neighbor_frame_refs=["F000", "F002"],
        detections=[
            Detection("W01", "person", [100, 100, 180, 320], 0.94),
            Detection("EQ01", "excavator", [220, 100, 420, 340], 0.88),
        ],
        ppe_finding=PPEStatus(
            helmet="missing",
            confidence=0.9,
            target_id="W01",
            evidence_detection_ids=["W01"],
        ),
        zone_grounding=RegionGrounding(
            zone_id="Z-ACTIVE-01",
            zone_type="active_work_area",
            spatial_relation="inside",
            confidence=0.98,
            target_id="W01",
            rule_source="manual",
        ),
        evidence_issues=list(evidence_issues or [EvidenceIssue.RELATION_UNCLEAR]),
        allowed_context_actions=list(
            allowed_context_actions
            if allowed_context_actions is not None
            else [
                ContextAction.EMIT_CONTEXT_EVIDENCE,
                ContextAction.REQUEST_HIGHER_RESOLUTION_CROP,
                ContextAction.REQUEST_MORE_FRAMES,
                ContextAction.ABSTAIN,
            ]
        ),
    )


def build_evidence(
    *,
    evidence_id: str = "CTXE-001",
    kind: str = "VISUAL_QUALITY",
    label: str = "CLEAR",
    status: str = "CONFIRMED",
    subject_detection_id: str = "W01",
    object_detection_id: str | None = None,
    zone_ref: str = "Z-ACTIVE-01",
    confidence: float = 0.9,
) -> ContextEvidence:
    return ContextEvidence(
        evidence_id=evidence_id,
        kind=kind,
        label=label,
        subject_detection_id=subject_detection_id,
        object_detection_id=object_detection_id,
        frame_ref="F001",
        crop_ref="CROP-W01-F001",
        zone_ref=zone_ref,
        confidence=confidence,
        status=status,
        reason_code=f"{label}_OBSERVED",
    )


class TestContextAgent(unittest.TestCase):
    def test_confirmed_local_relation_evidence_can_be_emitted(self) -> None:
        proposal = ContextProposal(
            evidence=[
                build_evidence(
                    kind=ContextEvidenceKind.LOCAL_RELATION.value,
                    label="NEAR",
                    object_detection_id="EQ01",
                )
            ],
            selected_action=ContextAction.EMIT_CONTEXT_EVIDENCE,
            action_parameters={"evidence_ids": ["CTXE-001"]},
            context_confidence=0.9,
        )
        result = ContextAgent(FakeContextModelAdapter(proposal)).analyze(build_request())

        self.assertEqual(result.selected_action, ContextAction.EMIT_CONTEXT_EVIDENCE)
        self.assertIsInstance(result.action_parameters, EmitContextEvidenceParameters)
        self.assertEqual(result.action_parameters.evidence_ids, ["CTXE-001"])
        self.assertEqual(result.validation_errors, [])

    def test_emit_is_rejected_when_gate_only_allows_crop_request(self) -> None:
        proposal = ContextProposal(
            evidence=[
                build_evidence(
                    kind=ContextEvidenceKind.LOCAL_RELATION.value,
                    label="NEAR",
                    object_detection_id="EQ01",
                )
            ],
            selected_action=ContextAction.EMIT_CONTEXT_EVIDENCE,
            action_parameters={"evidence_ids": ["CTXE-001"]},
            context_confidence=0.9,
        )
        request = build_request(
            allowed_context_actions=[
                ContextAction.REQUEST_HIGHER_RESOLUTION_CROP,
                ContextAction.ABSTAIN,
            ],
            evidence_issues=[EvidenceIssue.LOW_RESOLUTION],
        )

        result = ContextAgent(FakeContextModelAdapter(proposal)).analyze(request)

        self.assertEqual(result.selected_action, ContextAction.ABSTAIN)
        self.assertIn(
            "ACTION_NOT_ALLOWED_FOR_ROUTE:EMIT_CONTEXT_EVIDENCE",
            result.validation_errors,
        )

    def test_emit_selection_must_include_route_resolving_relation(self) -> None:
        proposal = ContextProposal(
            evidence=[
                build_evidence(
                    evidence_id="CTXE-RELATION",
                    kind=ContextEvidenceKind.LOCAL_RELATION.value,
                    label="NEAR",
                    object_detection_id="EQ01",
                ),
                build_evidence(evidence_id="CTXE-CLEAR"),
            ],
            selected_action=ContextAction.EMIT_CONTEXT_EVIDENCE,
            action_parameters={"evidence_ids": ["CTXE-CLEAR"]},
            context_confidence=0.9,
        )

        result = ContextAgent(FakeContextModelAdapter(proposal)).analyze(build_request())

        self.assertEqual(result.selected_action, ContextAction.ABSTAIN)
        self.assertIn(
            "EMIT_SELECTION_DOES_NOT_RESOLVE_ROUTE",
            result.validation_errors,
        )

    def test_low_resolution_requests_crop_and_marks_evidence_provisional(self) -> None:
        proposal = ContextProposal(
            evidence=[build_evidence(label="LOW_RESOLUTION")],
            selected_action=ContextAction.REQUEST_HIGHER_RESOLUTION_CROP,
            action_parameters={"reason_code": "TARGET_TOO_SMALL"},
            context_confidence=0.82,
        )
        result = ContextAgent(FakeContextModelAdapter(proposal)).analyze(build_request())

        self.assertEqual(
            result.selected_action,
            ContextAction.REQUEST_HIGHER_RESOLUTION_CROP,
        )
        self.assertIsInstance(result.action_parameters, RequestHigherResolutionCropParameters)
        self.assertEqual(result.action_parameters.target_detection_id, "W01")
        self.assertEqual(result.evidence[0].status, ContextEvidenceStatus.PROVISIONAL.value)

    def test_occlusion_and_low_light_do_not_request_higher_resolution_crop(self) -> None:
        for label in ("OCCLUDED", "LOW_LIGHT"):
            with self.subTest(label=label):
                proposal = ContextProposal(
                    evidence=[build_evidence(label=label)],
                    selected_action=ContextAction.REQUEST_HIGHER_RESOLUTION_CROP,
                    action_parameters={"reason_code": label},
                    context_confidence=0.8,
                )
                result = ContextAgent(FakeContextModelAdapter(proposal)).analyze(build_request())

                self.assertEqual(result.selected_action, ContextAction.ABSTAIN)
                self.assertIn("CROP_REQUEST_REQUIRES_RECOVERABLE_QUALITY", result.validation_errors)

    def test_crop_request_without_higher_resolution_source_abstains(self) -> None:
        proposal = ContextProposal(
            evidence=[build_evidence(label="LOW_RESOLUTION")],
            selected_action=ContextAction.REQUEST_HIGHER_RESOLUTION_CROP,
            action_parameters={"reason_code": "TARGET_TOO_SMALL"},
            context_confidence=0.8,
        )

        result = ContextAgent(FakeContextModelAdapter(proposal)).analyze(
            replace(build_request(), higher_resolution_source_ref=None)
        )

        self.assertEqual(result.selected_action, ContextAction.ABSTAIN)
        self.assertIn("HIGHER_RESOLUTION_SOURCE_UNAVAILABLE", result.validation_errors)

    def test_temporal_context_request_defaults_to_two_frames_each_side(self) -> None:
        proposal = ContextProposal(
            evidence=[build_evidence(label="TEMPORAL_CONTEXT_INSUFFICIENT")],
            selected_action=ContextAction.REQUEST_MORE_FRAMES,
            action_parameters={"reason_code": "BOUNDARY_STATE_UNCLEAR"},
            context_confidence=0.7,
        )
        result = ContextAgent(FakeContextModelAdapter(proposal)).analyze(build_request())

        self.assertEqual(result.selected_action, ContextAction.REQUEST_MORE_FRAMES)
        self.assertIsInstance(result.action_parameters, RequestMoreFramesParameters)
        self.assertEqual(result.action_parameters.frames_before, 2)
        self.assertEqual(result.action_parameters.frames_after, 2)
        self.assertEqual(result.evidence[0].status, ContextEvidenceStatus.PROVISIONAL.value)

    def test_frame_request_without_neighbor_frames_abstains(self) -> None:
        proposal = ContextProposal(
            evidence=[build_evidence(label="TEMPORAL_CONTEXT_INSUFFICIENT")],
            selected_action=ContextAction.REQUEST_MORE_FRAMES,
            action_parameters={"reason_code": "BOUNDARY_STATE_UNCLEAR"},
            context_confidence=0.7,
        )

        result = ContextAgent(FakeContextModelAdapter(proposal)).analyze(
            replace(build_request(), neighbor_frame_refs=[])
        )

        self.assertEqual(result.selected_action, ContextAction.ABSTAIN)
        self.assertIn("NEIGHBOR_FRAMES_UNAVAILABLE", result.validation_errors)

    def test_occlusion_with_neighbor_frames_can_request_more_frames(self) -> None:
        proposal = ContextProposal(
            evidence=[build_evidence(label="OCCLUDED")],
            selected_action=ContextAction.REQUEST_MORE_FRAMES,
            action_parameters={"reason_code": "OCCLUSION_RECOVERY"},
            context_confidence=0.7,
        )

        result = ContextAgent(FakeContextModelAdapter(proposal)).analyze(build_request())

        self.assertEqual(result.selected_action, ContextAction.REQUEST_MORE_FRAMES)

    def test_invalid_request_abstains_without_calling_adapter(self) -> None:
        safe_proposal = ContextProposal(
            evidence=[build_evidence()],
            selected_action=ContextAction.EMIT_CONTEXT_EVIDENCE,
            context_confidence=0.9,
        )
        invalid_requests = [
            replace(build_request(), worker_id="UNKNOWN"),
            replace(build_request(), crop_ref=""),
            replace(build_request(), frame_ref=""),
            replace(
                build_request(),
                ppe_finding=replace(
                    build_request().ppe_finding,
                    evidence_detection_ids=["MISSING-DETECTION"],
                ),
            ),
        ]

        for request in invalid_requests:
            with self.subTest(request=request):
                adapter = FakeContextModelAdapter(safe_proposal)
                result = ContextAgent(adapter).analyze(request)
                self.assertEqual(result.selected_action, ContextAction.ABSTAIN)
                self.assertIsInstance(result.action_parameters, AbstainParameters)
                self.assertGreater(len(result.validation_errors), 0)
                self.assertEqual(adapter.call_count, 0)

    def test_exhausted_attempt_budget_abstains_without_calling_adapter(self) -> None:
        safe_proposal = ContextProposal(
            evidence=[build_evidence()],
            selected_action=ContextAction.EMIT_CONTEXT_EVIDENCE,
            context_confidence=0.9,
        )
        adapter = FakeContextModelAdapter(safe_proposal)

        result = ContextAgent(adapter).analyze(
            replace(build_request(), context_attempt_count=1, max_context_attempts=1)
        )

        self.assertEqual(result.selected_action, ContextAction.ABSTAIN)
        self.assertEqual(result.action_parameters.reason_code, "CONTEXT_ATTEMPT_BUDGET_EXHAUSTED")
        self.assertEqual(adapter.call_count, 0)

    def test_relation_to_unknown_detection_is_rejected(self) -> None:
        proposal = ContextProposal(
            evidence=[
                build_evidence(
                    kind=ContextEvidenceKind.LOCAL_RELATION.value,
                    label="NEAR",
                    object_detection_id="INVENTED-OBJECT",
                )
            ],
            selected_action=ContextAction.EMIT_CONTEXT_EVIDENCE,
            context_confidence=0.85,
        )
        result = ContextAgent(FakeContextModelAdapter(proposal)).analyze(build_request())

        self.assertEqual(result.selected_action, ContextAction.ABSTAIN)
        self.assertIn("UNKNOWN_OBJECT_DETECTION_REF:INVENTED-OBJECT", result.validation_errors)

    def test_invented_subject_frame_crop_and_zone_refs_are_rejected(self) -> None:
        invalid_cases = {
            "subject": (
                build_evidence(subject_detection_id="INVENTED-SUBJECT"),
                "SUBJECT_MUST_MATCH_WORKER:INVENTED-SUBJECT",
            ),
            "frame": (
                replace(build_evidence(), frame_ref="INVENTED-FRAME"),
                "FRAME_REF_MISMATCH:INVENTED-FRAME",
            ),
            "crop": (
                replace(build_evidence(), crop_ref="INVENTED-CROP"),
                "CROP_REF_MISMATCH:INVENTED-CROP",
            ),
            "zone": (
                build_evidence(zone_ref="INVENTED-ZONE"),
                "ZONE_REF_MISMATCH:INVENTED-ZONE",
            ),
        }

        for name, (evidence, expected_error) in invalid_cases.items():
            with self.subTest(name=name):
                proposal = ContextProposal(
                    evidence=[evidence],
                    selected_action=ContextAction.EMIT_CONTEXT_EVIDENCE,
                    context_confidence=0.9,
                )
                result = ContextAgent(FakeContextModelAdapter(proposal)).analyze(
                    build_request()
                )

                self.assertEqual(result.selected_action, ContextAction.ABSTAIN)
                self.assertIn(expected_error, result.validation_errors)

    def test_inconsistent_crop_action_parameters_are_rejected(self) -> None:
        proposal = ContextProposal(
            evidence=[build_evidence(label="LOW_RESOLUTION")],
            selected_action=ContextAction.REQUEST_HIGHER_RESOLUTION_CROP,
            action_parameters={
                "target_detection_id": "W01",
                "frame_ref": "INVENTED-FRAME",
                "crop_ref": "CROP-W01-F001",
            },
            context_confidence=0.8,
        )
        request = build_request(
            allowed_context_actions=[
                ContextAction.REQUEST_HIGHER_RESOLUTION_CROP,
                ContextAction.ABSTAIN,
            ],
            evidence_issues=[EvidenceIssue.LOW_RESOLUTION],
        )

        result = ContextAgent(FakeContextModelAdapter(proposal)).analyze(request)

        self.assertEqual(result.selected_action, ContextAction.ABSTAIN)
        self.assertIn("ACTION_FRAME_REF_MISMATCH:INVENTED-FRAME", result.validation_errors)

    def test_free_action_is_rejected(self) -> None:
        proposal = ContextProposal(
            evidence=[build_evidence()],
            selected_action="SEND_CRITICAL_ALERT",
            context_confidence=0.9,
        )
        result = ContextAgent(FakeContextModelAdapter(proposal)).analyze(build_request())

        self.assertEqual(result.selected_action, ContextAction.ABSTAIN)
        self.assertIn("UNSUPPORTED_ACTION:SEND_CRITICAL_ALERT", result.validation_errors)

    def test_changed_zone_reference_is_rejected(self) -> None:
        proposal = ContextProposal(
            evidence=[build_evidence(zone_ref="Z-INVENTED")],
            selected_action=ContextAction.EMIT_CONTEXT_EVIDENCE,
            context_confidence=0.9,
        )
        result = ContextAgent(FakeContextModelAdapter(proposal)).analyze(build_request())

        self.assertEqual(result.selected_action, ContextAction.ABSTAIN)
        self.assertIn("ZONE_REF_MISMATCH:Z-INVENTED", result.validation_errors)

    def test_provisional_evidence_cannot_be_emitted(self) -> None:
        proposal = ContextProposal(
            evidence=[build_evidence(status="PROVISIONAL")],
            selected_action=ContextAction.EMIT_CONTEXT_EVIDENCE,
            context_confidence=0.75,
        )
        result = ContextAgent(FakeContextModelAdapter(proposal)).analyze(build_request())

        self.assertEqual(result.selected_action, ContextAction.ABSTAIN)
        self.assertIn("EMIT_REQUIRES_CONFIRMED_EVIDENCE", result.validation_errors)

    def test_blocking_visual_issue_cannot_be_emitted(self) -> None:
        proposal = ContextProposal(
            evidence=[build_evidence(label="FRAME_ERROR")],
            selected_action=ContextAction.EMIT_CONTEXT_EVIDENCE,
            context_confidence=0.8,
        )
        result = ContextAgent(FakeContextModelAdapter(proposal)).analyze(build_request())

        self.assertEqual(result.selected_action, ContextAction.ABSTAIN)
        self.assertIn("BLOCKING_VISUAL_QUALITY:FRAME_ERROR", result.validation_errors)

    def test_unconfigured_adapter_abstains_safely(self) -> None:
        result = ContextAgent().analyze(build_request())

        self.assertEqual(result.selected_action, ContextAction.ABSTAIN)
        self.assertIsInstance(result.action_parameters, AbstainParameters)
        self.assertEqual(result.action_parameters.reason_code, "MODEL_NOT_CONFIGURED")

    def test_abstain_evidence_is_provisional(self) -> None:
        proposal = ContextProposal(
            evidence=[build_evidence()],
            selected_action=ContextAction.ABSTAIN,
            action_parameters={"reason_code": "INSUFFICIENT_CONTEXT"},
            context_confidence=0.4,
        )

        result = ContextAgent(FakeContextModelAdapter(proposal)).analyze(build_request())

        self.assertEqual(result.selected_action, ContextAction.ABSTAIN)
        self.assertEqual(result.evidence[0].status, ContextEvidenceStatus.PROVISIONAL.value)

    def test_adapter_exception_abstains_safely(self) -> None:
        result = ContextAgent(FailingContextModelAdapter()).analyze(build_request())

        self.assertEqual(result.selected_action, ContextAction.ABSTAIN)
        self.assertEqual(result.action_parameters.reason_code, "MODEL_ADAPTER_ERROR")
        self.assertIn("MODEL_ADAPTER_ERROR:RuntimeError", result.validation_errors)

    def test_invalid_adapter_return_type_abstains_safely(self) -> None:
        result = ContextAgent(InvalidContextModelAdapter()).analyze(build_request())

        self.assertEqual(result.selected_action, ContextAction.ABSTAIN)
        self.assertEqual(result.action_parameters.reason_code, "INVALID_MODEL_PROPOSAL")
        self.assertEqual(result.validation_errors, ["INVALID_MODEL_PROPOSAL:dict"])


if __name__ == "__main__":
    unittest.main()
