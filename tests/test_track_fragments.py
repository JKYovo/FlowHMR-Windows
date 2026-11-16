"""Joining is explicit, refuses unsafe gaps, and keeps normal selection unchanged."""
import copy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch

import numpy as np

from flowhmr.utils.track_fragments import TrackJoinError, join_candidates
from flowhmr.utils.runtime_video_processing import VideoProcessor, TrackInfo, TrackingResult

ROOT = Path(__file__).resolve().parents[1]
spec = spec_from_file_location("track_join_web", ROOT / "tools/web/server.py")
web = module_from_spec(spec)
spec.loader.exec_module(web)


def candidate(index, start, end, box=(10, 10, 40, 90)):
    frames = list(range(start, end + 1))
    return dict(index=index, track_index=index, track_id=index + 1,
                start=start, end=end, num_frames=len(frames), frames=frames,
                boxes=[list(box) for _ in frames])


class TrackFragmentsTest(unittest.TestCase):
    def test_join_sorts_time_preserves_inputs_and_interpolates_short_gap(self):
        parts = [candidate(0, 0, 8), candidate(1, 10, 19, (12, 10, 42, 90))]
        original = copy.deepcopy(parts)
        joined = join_candidates(parts, [1, 0], (0, 19))
        self.assertEqual(parts, original)
        self.assertEqual(joined["index"], 1)
        self.assertEqual(joined["frames"], list(range(9)) + list(range(10, 20)))
        track = TrackInfo(1, np.array(joined["frames"]), np.array(joined["boxes"]))
        boxes, span = VideoProcessor.track_bboxes(TrackingResult([track], 20, 100, 100), 0)
        np.testing.assert_array_equal(span, [0, 19])
        np.testing.assert_allclose(boxes[9], [11, 10, 41, 90])
        self.assertEqual(boxes.shape, (20, 4))

    def test_refuses_overlap_long_gap_jump_and_missing_prefix(self):
        for parts, span, code in [
            ([candidate(0, 0, 10), candidate(1, 9, 20)], (0, 20), "track_join_overlap"),
            ([candidate(0, 0, 8), candidate(1, 25, 30)], (0, 30), "track_join_gap"),
            ([candidate(0, 0, 8), candidate(1, 9, 19, (500, 10, 530, 90))], (0, 19), "track_join_jump"),
            ([candidate(0, 40, 49), candidate(1, 50, 99)], (0, 99), "track_join_coverage"),
        ]:
            with self.subTest(code=code), self.assertRaises(TrackJoinError) as caught:
                join_candidates(parts, [0, 1], span)
            self.assertEqual(caught.exception.code, code)

    def test_checks_gaps_inside_fragments_and_bad_boxes(self):
        parts = [candidate(0, 0, 20), candidate(1, 21, 30)]
        parts[0].update(frames=[0, 20], boxes=[[10, 10, 40, 90]] * 2)
        with self.assertRaises(TrackJoinError) as caught:
            join_candidates(parts, [0, 1], (0, 30))
        self.assertEqual(caught.exception.code, "track_join_gap")
        parts[0]["boxes"][0] = [float("nan"), 10, 40, 90]
        with self.assertRaises(TrackJoinError) as caught:
            join_candidates(parts, [0, 1], (0, 30))
        self.assertEqual(caught.exception.code, "track_join_invalid")

    def test_partial_body_endpoint_does_not_veto_otherwise_continuous_track(self):
        parts = [candidate(0, 0, 8), candidate(1, 10, 19)]
        parts[0]["boxes"][-2:] = [[10, 10, 40, 35], [10, 10, 40, 35]]
        joined = join_candidates(parts, [0, 1], (0, 19))
        self.assertEqual((joined["start"], joined["end"]), (0, 19))

    def test_only_small_edge_holds_are_allowed(self):
        parts = [candidate(0, 2, 10), candidate(1, 11, 27)]
        joined = join_candidates(parts, [0, 1], (0, 29))
        self.assertEqual((joined["frames"][0], joined["frames"][-1]), (0, 29))

    def test_join_needs_two_distinct_fragments(self):
        parts = [candidate(0, 0, 9)]
        for people in ([0], [0, 0]):
            with self.assertRaises(TrackJoinError) as caught:
                join_candidates(parts, people, (0, 9))
            self.assertEqual(caught.exception.code, "track_join_count")


class TrackJoinRouteTest(unittest.TestCase):
    def test_rerun_preserves_join_only_when_detection_is_reused(self):
        from flowhmr.utils.runtime_capabilities import detection_request_key

        with TemporaryDirectory() as directory:
            args = SimpleNamespace(work_dir=directory, device="cuda:0")
            with (patch.object(web, "ARGS", args), patch.object(web, "JOBS", {}),
                  patch.object(web, "MEMBER_DIRS", {}), patch.object(web, "enqueue"),
                  web.app.test_client() as client):
                job = web.new_job("clip.mp4", web.default_params(), ".mp4")
                job.update(status="done", selected_person=0, track_fragments=[{"track_id": 1}, {"track_id": 2}],
                           person_track=dict(track_id=1, start=0, end=19, num_frames=20))
                folder = Path(web.job_dir(job["id"]))
                (folder / "input.mp4").write_bytes(b"original-video")
                sub = folder / "input"; sub.mkdir()
                np.savez(sub / "input_bbox.npz", bbox=np.zeros((20, 4)), start_end=[0, 19],
                         detection_request=detection_request_key((0, None)))
                feature = "input_sam3d_feat_static.pt"
                (sub / feature).write_bytes(b"cached-tokens")
                for body, reused in [({}, True), ({"reselect_person": True}, False), ({"start_frame": 1}, False)]:
                    response = client.post(f"/api/jobs/{job['id']}/rerun", json=body)
                    self.assertEqual(response.status_code, 200)
                    new = web.JOBS[response.json["job_id"]]
                    destination = Path(web.job_dir(new["id"])) / "input"
                    self.assertEqual((destination / feature).exists(), reused)
                    self.assertEqual(bool(new.get("track_fragments")), reused)

    def test_invalid_join_keeps_selection_open_and_normal_multi_select_is_unchanged(self):
        parts = [candidate(0, 0, 10), candidate(1, 9, 19)]
        job = dict(id="job", status="waiting_user", candidates=parts, source_frames=20,
                   params=web.default_params())
        slot = {"event": threading.Event(), "people": None}
        with (patch.object(web, "JOBS", {"job": job}), patch.object(web, "SELECTIONS", {"job": slot}),
              web.app.test_client() as client):
            invalid = client.post("/api/jobs/job/select", json={"people": [0, 1], "join_tracks": True})
            self.assertEqual(invalid.status_code, 400)
            self.assertEqual(invalid.json["error_code"], "track_join_overlap")
            self.assertFalse(slot["event"].is_set())
            self.assertIsNone(slot["people"])
            normal = client.post("/api/jobs/job/select", json={"people": [0, 1]})
            self.assertEqual(normal.status_code, 200)
            self.assertTrue(slot["event"].is_set())
            self.assertNotIn("joined_candidate", slot)

    def test_join_replaces_one_worker_track_and_saves_full_range_without_children(self):
        with TemporaryDirectory() as directory:
            args = SimpleNamespace(work_dir=directory)
            with (patch.object(web, "ARGS", args), patch.object(web, "JOBS", {}),
                  patch.object(web, "SELECTIONS", {}), patch.object(web, "MEMBER_DIRS", {})):
                job = web.new_job("clip.mp4", web.default_params(), ".mp4")
                parts = [candidate(0, 0, 8), candidate(1, 10, 19)]
                job.update(status="waiting_user", candidates=parts, source_frames=20)
                slot = {"event": threading.Event(), "people": None}
                web.SELECTIONS[job["id"]] = slot
                with web.app.test_client() as client:
                    response = client.post(f"/api/jobs/{job['id']}/select",
                                           json={"people": [1, 0], "join_tracks": True})
                self.assertEqual(response.status_code, 200)
                result = TrackingResult([TrackInfo(i + 1, np.array(p["frames"]), np.array(p["boxes"]))
                                         for i, p in enumerate(parts)], 20, 100, 100)
                original_first = result.tracks[0]
                web._confirm_joined_track(job["id"], result, slot["joined_candidate"])
                self.assertIs(result.tracks[0], original_first)
                self.assertEqual(len(web.JOBS), 1)
                self.assertNotIn("person_batch", job)
                self.assertEqual(job["person_track"]["start"], 0)
                self.assertEqual(job["person_track"]["end"], 19)
                self.assertEqual(len(job["track_fragments"]), 2)
                output = Path(directory) / "bbox.npz"
                VideoProcessor.save_detected_track(output, result, 1, (0, None))
                with np.load(output) as cache:
                    np.testing.assert_array_equal(cache["start_end"], [0, 19])
                    self.assertEqual(cache["bbox"].shape, (20, 4))
