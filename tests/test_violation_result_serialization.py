"""
tests/test_violation_result_serialization.py
==============================================

Regression coverage for the backend-integration fixes:

  1. Role mapping — outbound `role` must only ever be one of
     "LP" | "ALP" | "BOTH" | None. The Spring Boot role enum has no
     "AMBIGUOUS" member; sending it breaks Jackson enum deserialization.
  2. Data types — `status` must serialize as a real JSON boolean (not the
     string "TRUE"/"FALSE"), and `timestamp` / `originalVideoTimestamp`
     must serialize as plain numeric seconds (not "H:MM:SS" display
     strings), matching the other numeric fields (`riskScore`,
     `confidence`, `durationSeconds`).
  3. `role` must be None whenever `status` is False — a role is never
     assigned to a rejected/unverified candidate.

Run with:  python3 -m unittest discover -s tests -v
(No pytest dependency required; plain stdlib unittest.)
"""

from __future__ import annotations

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import ViolationResult  # noqa: E402
import llm_verifier  # noqa: E402


def _make_violation(**overrides) -> ViolationResult:
    """Build a ViolationResult with sane defaults, overridable per test."""
    defaults = dict(
        violation_type="HAND_RAISING",
        severity="LOW",
        confidence=0.865,          # fraction, as analyzer.py passes pre-*100
        risk_score=0.0,
        timestamp_seconds=43.0,
        original_video_timestamp=43.0,
        duration_seconds=1.8,
        trigger_duration_seconds=3.6,
        frame_paths=["development/journeys/212/frames/hand_raise.jpg"],
        status="TRUE",
        role="LP",
    )
    defaults.update(overrides)
    return ViolationResult(**defaults)


class TestViolationResultToDict(unittest.TestCase):

    # ── Data types ────────────────────────────────────────────────────────

    def test_status_serializes_as_real_boolean_not_string(self):
        payload = _make_violation(status="TRUE").to_dict()
        self.assertIs(payload["status"], True)
        self.assertIsInstance(payload["status"], bool)

        payload = _make_violation(status="FALSE", role=None).to_dict()
        self.assertIs(payload["status"], False)
        self.assertIsInstance(payload["status"], bool)

    def test_timestamp_fields_are_numeric_not_hms_strings(self):
        payload = _make_violation(
            timestamp_seconds=141.0, original_video_timestamp=141.0
        ).to_dict()

        # These must be numbers the backend can bind to a numeric DTO
        # field — not "0:02:21"-style display strings.
        self.assertIsInstance(payload["timestamp"], float)
        self.assertIsInstance(payload["originalVideoTimestamp"], float)
        self.assertEqual(payload["timestamp"], 141.0)
        self.assertEqual(payload["originalVideoTimestamp"], 141.0)

    def test_timestamp_fields_are_rounded(self):
        payload = _make_violation(
            timestamp_seconds=16.34567, original_video_timestamp=16.34567
        ).to_dict()
        self.assertEqual(payload["timestamp"], 16.346)
        self.assertEqual(payload["originalVideoTimestamp"], 16.346)

    def test_numeric_fields_are_floats(self):
        payload = _make_violation().to_dict()
        for field in ("confidence", "riskScore", "durationSeconds"):
            self.assertIsInstance(payload[field], float, field)

    def test_trigger_duration_none_stays_none(self):
        payload = _make_violation(trigger_duration_seconds=None).to_dict()
        self.assertIsNone(payload["triggerDurationSeconds"])

    def test_frame_paths_is_a_list(self):
        payload = _make_violation().to_dict()
        self.assertIsInstance(payload["framePaths"], list)

    # ── Role contract ────────────────────────────────────────────────────

    def test_role_is_never_ambiguous_string(self):
        """AMBIGUOUS is not a member of the backend's role enum."""
        for role in ("LP", "ALP", "BOTH", None):
            payload = _make_violation(role=role).to_dict()
            self.assertNotEqual(payload["role"], "AMBIGUOUS")
            self.assertIn(payload["role"], ("LP", "ALP", "BOTH", None))

    def test_role_is_none_when_status_false(self):
        payload = _make_violation(status="FALSE", role=None).to_dict()
        self.assertIs(payload["status"], False)
        self.assertIsNone(payload["role"])

    def test_role_can_be_none_when_status_true(self):
        """Confirmed-but-unattributable violations: status True, role None
        is a legitimate value, distinct from a rejected candidate."""
        payload = _make_violation(status="TRUE", role=None).to_dict()
        self.assertIs(payload["status"], True)
        self.assertIsNone(payload["role"])

    def test_each_canonical_role_round_trips(self):
        for role in ("LP", "ALP", "BOTH"):
            payload = _make_violation(status="TRUE", role=role).to_dict()
            self.assertEqual(payload["role"], role)
            self.assertIs(payload["status"], True)


class TestLlmVerifierRoleMapping(unittest.TestCase):
    """The prompt.py → outbound-payload role mapping (task 1)."""

    def test_descriptive_roles_map_to_canonical_codes(self):
        self.assertEqual(llm_verifier.ROLE_TO_CODE["Loco Pilot"], "LP")
        self.assertEqual(llm_verifier.ROLE_TO_CODE["Assistant Loco Pilot"], "ALP")
        self.assertEqual(llm_verifier.ROLE_TO_CODE["Both"], "BOTH")

    def test_unknown_role_maps_to_none_not_ambiguous(self):
        self.assertIsNone(llm_verifier.ROLE_TO_CODE["Unknown"])

    def test_status_role_verified_true_known_role(self):
        status, role = llm_verifier._status_role(True, "Loco Pilot")
        self.assertEqual(status, "TRUE")
        self.assertEqual(role, "LP")

    def test_status_role_verified_true_unknown_role(self):
        status, role = llm_verifier._status_role(True, "Unknown")
        self.assertEqual(status, "TRUE")
        self.assertIsNone(role)

    def test_status_role_verified_true_unmapped_role_defaults_to_none(self):
        status, role = llm_verifier._status_role(True, "totally-unmapped-value")
        self.assertEqual(status, "TRUE")
        self.assertIsNone(role)

    def test_status_role_not_verified_always_false_role_none(self):
        status, role = llm_verifier._status_role(False, "Loco Pilot")
        self.assertEqual(status, "FALSE")
        self.assertIsNone(role)

    def test_no_role_output_is_ever_ambiguous(self):
        """Exhaustive check: no combination of verify_frame's internal
        inputs can produce the literal string 'AMBIGUOUS' on the wire."""
        for verified in (True, False):
            for raw_role in (
                "Loco Pilot", "Assistant Loco Pilot", "Both", "Unknown",
                "garbage", "",
            ):
                _, role = llm_verifier._status_role(verified, raw_role)
                self.assertNotEqual(role, "AMBIGUOUS")


class TestVideoResultFromDictRoundTrip(unittest.TestCase):
    """
    Regression test for consumer.py's _video_result_from_dict(): journeys run
    in a supervised child subprocess (see _run_journey_supervised), and its
    ViolationResult objects cross the process boundary as a dict (built by
    ViolationResult.to_dict() in the child) before being reconstructed on the
    parent side. status/role were previously never read back out of that
    dict, so every violation silently reverted to the dataclass defaults
    (status="TRUE", role=None) in the parent process — even when the child
    had already computed a real LLM verdict (confirmed/rejected + a role).
    """

    @classmethod
    def setUpClass(cls):
        import sys as _sys
        from unittest.mock import MagicMock

        # consumer.py pulls in the full detector/analyzer import chain
        # (cv2, mediapipe, boto3, pika, ollama). Stub those out — this test
        # only exercises the pure-python dict reconstruction logic.
        for name in ("cv2", "mediapipe", "boto3", "dotenv", "ollama", "pika"):
            _sys.modules.setdefault(name, MagicMock())
        _sys.modules["dotenv"].load_dotenv = lambda *a, **k: None

        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import consumer  # noqa: E402
        cls.consumer = consumer

    def _video_dict(self, violations):
        return {
            "videoId": "2159", "videoName": "001_handgestures.mp4",
            "sequenceNo": 1, "durationSeconds": 254.333,
            "originalS3Key": "development/journeys/212/original/x.mp4",
            "violations": violations,
        }

    def test_confirmed_violation_with_role_survives_round_trip(self):
        vr = self._video_dict([{
            "violationType": "Hand Raise on signaling", "severity": "LOW",
            "confidence": 65.1, "riskScore": 0.0,
            "timestamp": 16.0, "originalVideoTimestamp": 16.0,
            "durationSeconds": 1.2, "triggerDurationSeconds": 2.4,
            "framePaths": ["f1.jpg"], "status": True, "role": "LP",
        }])
        result = self.consumer._video_result_from_dict(vr)
        payload = result.violations[0].to_dict()
        self.assertIs(payload["status"], True)
        self.assertEqual(payload["role"], "LP")

    def test_rejected_violation_survives_round_trip(self):
        """The exact case that was broken: an LLM-rejected candidate
        (status=False in the child) must NOT come back as status=true."""
        vr = self._video_dict([{
            "violationType": "Hand Raise on signaling", "severity": "LOW",
            "confidence": 0, "riskScore": 0.0,
            "timestamp": 43.0, "originalVideoTimestamp": 43.0,
            "durationSeconds": 1.8, "triggerDurationSeconds": 3.6,
            "framePaths": ["f2.jpg"], "status": False, "role": None,
        }])
        result = self.consumer._video_result_from_dict(vr)
        payload = result.violations[0].to_dict()
        self.assertIs(payload["status"], False)
        self.assertIsNone(payload["role"])

    def test_confirmed_ambiguous_role_survives_round_trip(self):
        vr = self._video_dict([{
            "violationType": "Hand Raise on signaling", "severity": "LOW",
            "confidence": 90.0, "riskScore": 0.0,
            "timestamp": 100.0, "originalVideoTimestamp": 100.0,
            "durationSeconds": 1.0, "triggerDurationSeconds": None,
            "framePaths": ["f3.jpg"], "status": True, "role": None,
        }])
        result = self.consumer._video_result_from_dict(vr)
        payload = result.violations[0].to_dict()
        self.assertIs(payload["status"], True)
        self.assertIsNone(payload["role"])


if __name__ == "__main__":
    unittest.main()