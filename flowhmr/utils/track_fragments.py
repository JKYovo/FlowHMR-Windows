"""Explicit, conservative joining of user-confirmed single-person track fragments."""
import numpy as np

MAX_GAP = 15  # Missing frames at normalized 30 FPS; never bridge a long absence.
MAX_EDGE = 5


class TrackJoinError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def join_candidates(candidates, people, frame_range):
    """Return one candidate covering the requested range, or refuse ambiguity.

    This validates temporal/box continuity, not identity. The user must confirm
    that the selected, non-overlapping fragments show the same person.
    """
    if len(people) < 2 or len(set(people)) != len(people):
        raise TrackJoinError("track_join_count")
    parts = []
    for index in people:
        if type(index) is not int or not 0 <= index < len(candidates):
            raise TrackJoinError("track_join_invalid")
        candidate = candidates[index]
        frames = np.asarray(candidate.get("frames", []))
        boxes = np.asarray(candidate.get("boxes", []), dtype=np.float64)
        if (frames.ndim != 1 or frames.size == 0 or frames.dtype.kind not in "iu"
                or boxes.shape != (len(frames), 4) or not np.isfinite(boxes).all()
                or np.any(boxes[:, 2:] <= boxes[:, :2])):
            raise TrackJoinError("track_join_invalid")
        order = np.argsort(frames)
        frames, boxes = frames[order], boxes[order]
        if np.any(np.diff(frames) <= 0):
            raise TrackJoinError("track_join_invalid")
        parts.append((index, frames, boxes))
    parts.sort(key=lambda part: int(part[1][0]))
    for (_, before, left), (_, after, right) in zip(parts, parts[1:]):
        if after[0] <= before[-1]:
            raise TrackJoinError("track_join_overlap")
        # A close hand-off is necessary even when the user confirms identity.
        # ID switches often coincide with a few partial-body detections. Use
        # five observations so one unstable endpoint cannot veto a valid join.
        a, b = np.median(left[-5:], axis=0), np.median(right[:5], axis=0)
        sizes = np.stack([a[2:] - a[:2], b[2:] - b[:2]])
        centers = np.stack([(a[:2] + a[2:]) / 2, (b[:2] + b[2:]) / 2])
        ratio = np.prod(sizes[1]) / np.prod(sizes[0])
        if not .5 <= ratio <= 2 or np.linalg.norm(centers[1] - centers[0]) > .5 * np.linalg.norm(sizes.mean(axis=0)):
            raise TrackJoinError("track_join_jump")
    frames = np.concatenate([part[1] for part in parts])
    boxes = np.concatenate([part[2] for part in parts])
    if np.any(np.diff(frames) - 1 > MAX_GAP):
        raise TrackJoinError("track_join_gap")
    start, end = frame_range
    if (start < 0 or end < start or frames[0] < start or frames[-1] > end
            or frames[0] - start > MAX_EDGE or end - frames[-1] > MAX_EDGE):
        raise TrackJoinError("track_join_coverage")
    # Hold endpoints for at most five frames. Internal holes are interpolated
    # by VideoProcessor.track_bboxes, exactly like ordinary tracking gaps.
    if frames[0] > start:
        frames = np.concatenate([[start], frames]); boxes = np.concatenate([boxes[:1], boxes])
    if frames[-1] < end:
        frames = np.concatenate([frames, [end]]); boxes = np.concatenate([boxes, boxes[-1:]])
    first = candidates[people[0]]
    return {**first, "start": int(start), "end": int(end), "num_frames": len(frames),
            "frames": frames.tolist(), "boxes": boxes.tolist(),
            "fragment_indices": list(people),
            "source_tracks": [{key: candidates[index][key] for key in ("track_id", "start", "end")}
                              for index in people]}
