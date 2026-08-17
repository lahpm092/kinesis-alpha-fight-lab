"""SAM 3.1 video masks for the two fighters.

Each chunk consumes the hong/chung tracks from stage 02 as spatial proposals.
SAM 3.1 finds concept-conditioned person masks on every analysis frame, then
mask-level red/blue hogu evidence is required before a temporal box may name an
instance. Re-prompting the current model avoids the invalid background masks
produced by its MLX Object Multiplex memory path; single-instance cleanup and
conservative shape/on-mat gates reject the referee, spectators, and blue-floor
fragments.

Output ``work/masks/video_chunk_<master-frame>.npz`` contains RLE masks and
per-mask area/centroid statistics at the analysis cadence.  Schema 3 is
reserved for SAM 3.1 video-frame output; older SAM 3/image-only caches are never
accepted silently.

Run this stage with the MLX environment documented in README.md:
    work/sam31-env/bin/python pipeline/03_masks.py [--limit N] [--chunk 30]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

FIGHTERS = ("red", "blue")
SCHEMA = 3
QUALITY_VERSION = "mask-hogu-cc-v5"
UPGRADABLE_QUALITY_VERSION = "mask-hogu-v4"
_sam = None


def sam31_dir():
    """Resolve the MLX SAM 3.1 checkpoint without downloading at runtime."""
    explicit = os.environ.get("FIGHTLAB_SAM31")
    candidates = ([Path(explicit)] if explicit else []) + [
        common.WORK / "models" / "sam3.1-bf16",
        common.models_dir() / "sam3.1-bf16",
    ]
    for path in candidates:
        if (path / "model.safetensors").exists() and (path / "config.json").exists():
            return path
    raise SystemExit(
        "SAM 3.1 MLX weights not found. Run:\n"
        "  hf download mlx-community/sam3.1-bf16 "
        "--local-dir work/models/sam3.1-bf16\n"
        "or set FIGHTLAB_SAM31 to that model directory."
    )


def sam31_video(resolution):
    """Load the native Apple-silicon SAM 3.1 model and inference helpers."""
    global _sam
    if _sam is None:
        try:
            import mlx.core as mx
            from mlx_vlm.generate import wired_limit
            from mlx_vlm.models.sam3.generate import Sam3Predictor
            from mlx_vlm.models.sam3_1.generate import (
                _detect_with_backbone,
                _get_backbone_features,
            )
            from mlx_vlm.models.sam3_1.processing_sam3_1 import Sam31Processor
            from mlx_vlm.utils import load_model
        except ImportError as exc:
            raise SystemExit(
                "stage 03 requires mlx-vlm==0.4.3 in an Apple-silicon Python "
                "environment; see README.md"
            ) from exc

        mdir = sam31_dir()
        model = load_model(mdir)
        config = json.load(open(mdir / "config.json"))
        quant = config.get("quantization") or config.get("quantization_config")
        precision = (f"{quant['bits']}-bit {quant.get('mode', 'affine')}"
                     if quant else "fp32")
        processor = Sam31Processor.from_pretrained(str(mdir))
        processor.image_size = int(resolution)
        predictor = Sam3Predictor(model, processor, score_threshold=.12)
        _sam = {
            "mx": mx,
            "model": model,
            "processor": processor,
            "predictor": predictor,
            "backbone": _get_backbone_features,
            "detect": _detect_with_backbone,
            "wired_limit": wired_limit,
            "model_dir": mdir,
            "precision": precision,
        }
        print(f"loaded SAM 3.1 from {mdir} at {resolution}px", flush=True)
    return _sam


def box_iou(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    wh = np.maximum(0.0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))
    inter = float(wh.prod())
    union = float(np.prod(np.maximum(0.0, a[2:] - a[:2])) +
                  np.prod(np.maximum(0.0, b[2:] - b[:2])) - inter)
    return inter / union if union > 0 else 0.0


def mask_box(mask):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return np.zeros(4, np.float32)
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float32)


def clip_to_detection(mask, box, pad=.08):
    """Remove disconnected decoder spill outside its own person detection.

    SAM concept masks can contain a correct occluded fighter plus distant mat
    or advertising-board islands.  The decoder's instance box remains local
    to the person, so a modest padded crop keeps limbs while dropping those
    unrelated pixels before colour scoring, pose crops, or rendering.
    """
    h, w = mask.shape
    x0, y0, x1, y1 = [float(v) for v in box]
    margin = pad * max(x1 - x0, y1 - y0)
    xa = max(0, int(np.floor(x0 - margin)))
    ya = max(0, int(np.floor(y0 - margin)))
    xb = min(w, int(np.ceil(x1 + margin)))
    yb = min(h, int(np.ceil(y1 + margin)))
    clean = np.zeros_like(mask, dtype=bool)
    if xb > xa and yb > ya:
        clean[ya:yb, xa:xb] = mask[ya:yb, xa:xb]
    return clean


def largest_component(mask):
    """Keep one connected SAM instance and discard detached false fragments.

    A concept mask can occasionally combine several tiny parts of the referee
    into one candidate.  Their union may span a person-sized bounding box even
    though no connected component is a fighter.  Keeping the largest component
    before all geometry and colour checks prevents that union-of-fragments
    failure while retaining the main fighter silhouette.
    """
    import cv2
    mask = np.asarray(mask, bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return np.zeros_like(mask, dtype=bool)
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == component


def torso_colour_score(bgr, box, role):
    """Expected hogu-colour fraction in the middle of a tracked person box."""
    import cv2
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    h, w = bgr.shape[:2]
    bw, bh = x1 - x0, y1 - y0
    x0 = max(0, x0 + int(.18 * bw)); x1 = min(w, x1 - int(.18 * bw))
    y0 = max(0, y0 + int(.18 * bh)); y1 = min(h, y1 - int(.62 * bh))
    if x1 - x0 < 3 or y1 - y0 < 3:
        return 0.0
    hsv = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    if role == "red":
        colour = ((hue <= 9) | (hue >= 168)) & (sat > 70) & (val > 40)
    else:
        colour = (hue >= 92) & (hue <= 138) & (sat > 45) & (val > 25)
    return float(colour.mean())


def choose_anchor(master_frames, bgr_frames, boxes):
    """Choose the best separated, colour-consistent two-fighter prompt frame."""
    best = None
    for q, (f, image) in enumerate(zip(master_frames, bgr_frames)):
        roles = [role for role in FIGHTERS if f in boxes[role]]
        if not roles:
            continue
        colour = sum(torso_colour_score(image, boxes[role][f], role) for role in roles)
        separation = 1.0
        if len(roles) == 2:
            separation = 1.0 - box_iou(boxes["red"][f], boxes["blue"][f])
        central = 1.0 - abs(q - (len(master_frames) - 1) / 2) / max(len(master_frames), 1)
        score = 4.0 * len(roles) + 2.0 * separation + 5.0 * colour + .15 * central
        if best is None or score > best[0]:
            best = score, q, roles
    return None if best is None else (best[1], best[2])


def read_chunk(cap, master_frames):
    import cv2
    cap.set(cv2.CAP_PROP_POS_FRAMES, master_frames[0])
    wanted = set(master_frames)
    frames = {}
    i = master_frames[0]
    while i <= master_frames[-1]:
        ok, frame = cap.read()
        if not ok:
            break
        if i in wanted:
            frames[i] = frame
        i += 1
    return [frames[i] for i in master_frames if i in frames]


def cache_ok(path, master_frames, resolution):
    if not path.exists():
        return False
    try:
        with np.load(path) as chunk:
            return (int(chunk["schema"]) == SCHEMA and
                    str(chunk["quality_version"]) == QUALITY_VERSION and
                    int(chunk["resolution"]) == resolution and
                    np.array_equal(chunk["chunk_F"], np.asarray(master_frames, np.int32)))
    except Exception:
        return False


def upgradeable_cache(path, master_frames, resolution):
    """Return whether a v4 chunk can be deterministically cleaned to v5."""
    if not path.exists():
        return False
    try:
        with np.load(path) as chunk:
            return (int(chunk["schema"]) == SCHEMA and
                    str(chunk["quality_version"]) == UPGRADABLE_QUALITY_VERSION and
                    int(chunk["resolution"]) == resolution and
                    np.array_equal(chunk["chunk_F"],
                                   np.asarray(master_frames, np.int32)))
    except Exception:
        return False


def frame_features(sam, rgb):
    """Compute the SAM 3.1 TriViT backbone once for one source frame."""
    from PIL import Image
    image = Image.fromarray(rgb)
    inputs = sam["processor"].preprocess_image(image)
    return sam["backbone"](sam["model"], sam["mx"].array(inputs["pixel_values"]))


def sane_prompt_mask(mask, box):
    """Reject empty/background visual prompts before they poison video memory."""
    box = np.asarray(box, float)
    box_area = max(1.0, float(np.prod(np.maximum(0.0, box[2:] - box[:2]))))
    area_ratio = float(mask.sum()) / box_area
    return .08 <= area_ratio <= 3.2 and box_iou(mask_box(mask), box) >= .12


def sane_fighter_mask(mask, box, height):
    """A bout fighter must have meaningful mask support down on the mat."""
    mb = mask_box(mask)
    width = mask.shape[1]
    # Text-conditioned masks can occasionally include a very wide, sparse
    # piece of the blue mat along with an otherwise plausible detection.  It
    # can overlap a fighter box enough to pass an IoU-only gate, but no single
    # fighter spans most of this broadcast frame.
    return (sane_prompt_mask(mask, box) and mb[3] >= .58 * height and
            mb[3] - mb[1] >= .20 * height and
            mb[2] - mb[0] <= .55 * width)


def sane_colour_fallback(mask, height, role):
    """Conservative shape gate for a role recovered without a temporal box.

    The competition mat is blue and can therefore beat a weak blue-hogu
    colour score.  A recovery candidate must look like an upright or airborne
    person, not a low, wide floor fragment.  Tracked masks are not subject to
    this extra gate, so a genuinely fallen fighter can still be retained.
    """
    mb = mask_box(mask)
    bw, bh = mb[2] - mb[0], mb[3] - mb[1]
    centre_x = .5 * (mb[0] + mb[2])
    compactness = float(mask.sum()) / max(float(bw * bh), 1.0)
    # The extra vertical gate is blue-specific: a genuinely fallen red
    # fighter remains distinguishable from the blue mat, while a low blue
    # fragment is intrinsically ambiguous and is safer to withhold.
    blue_vertical_ok = role != "blue" or mb[1] <= .62 * height
    return (bh >= .25 * height and mb[3] >= .60 * height and
            .14 * mask.shape[1] <= centre_x <= .86 * mask.shape[1] and
            compactness >= .18 and
            blue_vertical_ok and bw <= 1.8 * bh and
            bw <= .55 * mask.shape[1])


def kit_colour_support(bgr, mask):
    """Return mask-level hogu evidence for both competition roles.

    Evaluating colour only inside the assigned SAM silhouette prevents a
    stale/expanded tracker box from borrowing colour from the other fighter.
    The thresholds deliberately target saturated protector/helmet pixels,
    not skin or the red broadcast graphics.
    """
    import cv2
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    colours = {
        "red": ((hue <= 9) | (hue >= 168)) & (sat > 70) & (val > 40),
        "blue": (hue >= 92) & (hue <= 138) & (sat > 45) & (val > 25),
    }
    area = max(int(mask.sum()), 1)
    return {role: (float((mask & pixels).sum()) / area,
                   int((mask & pixels).sum()))
            for role, pixels in colours.items()}


def role_colour_ok(support, role, mask_area):
    """Require visible kit evidence and reject masks coloured as the rival."""
    other = "blue" if role == "red" else "red"
    own, own_px = support[role]
    rival, _ = support[other]
    return (own >= .045 and own_px >= max(20, int(.01 * mask_area)) and
            own >= 1.2 * rival)


def detect_frame_masks(sam, features, rgb, roles, role_boxes):
    """Assign unique SAM people using kit evidence plus temporal role boxes.

    A detector track may end while a fighter remains plainly visible (most
    notably at the final bell).  Temporal boxes are spatial proposals, never
    identity authority: every selected mask must contain the expected red/blue
    hogu evidence.  A failed or absent box may recover only a full-sized,
    on-mat SAM person instance.
    """
    height, width = rgb.shape[:2]
    result = sam["detect"](
        sam["predictor"], features, ["person"], (width, height), .08
    )
    if not len(result.scores):
        return [], []

    import cv2
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    candidates = []
    evidence = {}
    for q, mask in enumerate(result.masks):
        mask = np.asarray(mask, bool)
        det_box = np.asarray(result.boxes[q], np.float32)
        if mask.shape != (height, width):
            mask = cv2.resize(mask.astype(np.uint8), (width, height),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
        mask = largest_component(clip_to_detection(mask, det_box))
        if not mask.any():
            continue
        candidates.append((q, mask, det_box, float(result.scores[q])))
        evidence[q] = kit_colour_support(bgr, mask)

    # Greedy over all role/candidate pairs is globally correct for two roles
    # once pairs are sorted: one SAM instance cannot be assigned to both hong
    # and chung.  Mask/box overlap prevents the referee or crowd from winning.
    pairs = []
    for role in roles:
        if role not in role_boxes:
            continue
        seed = np.asarray(role_boxes[role], np.float32)
        sx0, sy0, sx1, sy1 = seed.astype(int)
        for q, mask, det_box, confidence in candidates:
            if not role_colour_ok(evidence[q], role, int(mask.sum())):
                continue
            clipped = mask[max(0, sy0):min(height, sy1),
                           max(0, sx0):min(width, sx1)]
            inside = float(clipped.sum()) / max(float(mask.sum()), 1.0)
            score = 2.0 * box_iou(det_box, seed) + inside + .1 * confidence
            pairs.append((score, role, q))
    pairs.sort(reverse=True)

    chosen = {}
    support_boxes = {}
    used = set()
    for score, role, q in pairs:
        if role in chosen or q in used or score < .18:
            continue
        chosen[role] = q
        support_boxes[role] = role_boxes[role]
        used.add(q)

    # A colour-classified detector track can briefly jump to a similarly
    # dressed spectator.  Validate its actual SAM silhouette before it blocks
    # the kit-colour recovery path.
    by_q = {q: mask for q, mask, _, _ in candidates}
    boxes_by_q = {q: box for q, _, box, _ in candidates}
    for role, q in list(chosen.items()):
        if (not sane_fighter_mask(by_q[q], role_boxes[role], height) or
                not role_colour_ok(evidence[q], role, int(by_q[q].sum()))):
            del chosen[role]
            support_boxes.pop(role, None)
            used.remove(q)

    # Recover a role only when its temporal detector box is absent/failed.
    # This does not guess by proximity: the sport's red/blue chest protector
    # is the identity evidence, and the lower-frame gate excludes spectators.
    colour_pairs = []
    for role in roles:
        if role in chosen:
            continue
        other = "blue" if role == "red" else "red"
        for q, mask, det_box, confidence in candidates:
            if q in used:
                continue
            x0, y0, x1, y1 = det_box
            if y1 < .42 * height or y1 - y0 < .11 * height:
                continue
            if not sane_fighter_mask(mask, det_box, height):
                continue
            if not sane_colour_fallback(mask, height, role):
                continue
            colour, _ = evidence[q][role]
            rival, _ = evidence[q][other]
            score = colour - .5 * rival + .01 * confidence
            if role_colour_ok(evidence[q], role, int(mask.sum())):
                colour_pairs.append((score, role, q))
    colour_pairs.sort(reverse=True)
    for score, role, q in colour_pairs:
        if role in chosen or q in used:
            continue
        chosen[role] = q
        support_boxes[role] = boxes_by_q[q]
        used.add(q)

    active_roles, masks = [], []
    for role in roles:
        q = chosen.get(role)
        support_box = support_boxes.get(role)
        if q is None or support_box is None or not sane_fighter_mask(
                by_q[q], support_box, height) or not role_colour_ok(
                evidence[q], role, int(by_q[q].sum())):
            continue
        active_roles.append(role)
        masks.append(by_q[q])
    return active_roles, masks


def upgrade_chunk(path, bgr_frames, master_frames, boxes, resolution):
    """Rewrite a validated v4 chunk with connected-instance v5 cleanup.

    This is loss-only and deterministic: it never invents a mask or reruns the
    model.  Colour evidence is recalculated on the retained component and the
    same tracked/fallback geometry gates are applied again.
    """
    with np.load(path, allow_pickle=False) as old:
        precision = str(old["precision"]) if "precision" in old else "fp32"
        old_masks = {
            (frame, role): np.asarray(old[f"f{frame}_{role}"])
            for frame in master_frames for role in FIGHTERS
            if f"f{frame}_{role}" in old
        }
    data = {
        "schema": np.array(SCHEMA, np.int16),
        "quality_version": np.array(QUALITY_VERSION),
        "resolution": np.array(resolution, np.int16),
        "precision": np.array(precision),
        "chunk_F": np.asarray(master_frames, np.int32),
    }
    found = 0
    height, width = bgr_frames[0].shape[:2]
    for q, frame_no in enumerate(master_frames):
        for role in FIGHTERS:
            runs = old_masks.get((frame_no, role))
            if runs is None:
                continue
            mask = largest_component(common.rle_decode(runs, (height, width)))
            if not mask.any():
                continue
            support = kit_colour_support(bgr_frames[q], mask)
            tracked_ok = (frame_no in boxes[role] and
                          sane_fighter_mask(mask, boxes[role][frame_no], height))
            fallback_ok = (sane_fighter_mask(mask, mask_box(mask), height) and
                           sane_colour_fallback(mask, height, role))
            if (not (tracked_ok or fallback_ok) or
                    not role_colour_ok(support, role, int(mask.sum()))):
                continue
            data[f"f{frame_no}_{role}"] = common.rle_encode(mask)
            ys, xs = np.nonzero(mask)
            data[f"s{frame_no}_{role}"] = np.array(
                [len(xs), xs.mean(), ys.mean()], np.float32
            )
            other = "blue" if role == "red" else "red"
            data[f"c{frame_no}_{role}"] = np.array(
                [support[role][0], support[other][0], support[role][1]],
                np.float32,
            )
            found += 1
    np.savez_compressed(path, **data)
    return found


def segment_chunk(sam, rgb_frames, master_frames, boxes):
    """Refresh SAM 3.1 masks per frame and bind them to temporal role tracks."""
    output = {}
    requested = found = 0
    for q, (rgb, frame_no) in enumerate(zip(rgb_frames, master_frames)):
        roles = list(FIGHTERS)
        role_boxes = {role: boxes[role][frame_no] for role in roles
                      if frame_no in boxes[role]}
        features = frame_features(sam, rgb)
        active_roles, masks = detect_frame_masks(
            sam, features, rgb, roles, role_boxes
        )
        output[q] = dict(zip(active_roles, masks))
        requested += len(roles)
        found += len(active_roles)
        if (q + 1) % 10 == 0:
            print(f"    {q + 1}/{len(rgb_frames)} frames", flush=True)
    return output, requested, found


def main():
    import cv2

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="analysis frames")
    parser.add_argument("--chunk", type=int, default=30,
                        help="analysis frames per restartable SAM 3.1 chunk")
    parser.add_argument("--resolution", type=int, default=840,
                        help="square SAM input; source is only 832x480")
    parser.add_argument("--chunk-start", type=int, default=1,
                        help="first 1-based chunk to process")
    parser.add_argument("--chunk-stop", type=int, default=0,
                        help="last 1-based chunk to process (default: final)")
    parser.add_argument("--reverse", action="store_true",
                        help="process the selected chunk range from the end")
    parser.add_argument("--output-dir", type=Path,
                        help="alternate artifact directory (quality tests)")
    parser.add_argument("--redo", action="store_true")
    args = parser.parse_args()

    source = common.video_path()
    track, boxes = common.load_fighters()
    step = int(track.get("stride", common.STRIDE))
    meta = common.probe(source)
    frames = list(range(0, meta["n"], step))
    if args.limit:
        frames = frames[:args.limit]
    chunks = [frames[i:i + args.chunk] for i in range(0, len(frames), args.chunk)]
    all_chunks = chunks
    first = max(0, args.chunk_start - 1)
    stop = min(len(chunks), args.chunk_stop or len(chunks))
    if first >= stop:
        raise SystemExit(f"empty chunk range {args.chunk_start}..{args.chunk_stop}")
    selected = list(enumerate(chunks))[first:stop]
    if args.reverse:
        selected.reverse()
    out_dir = args.output_dir or (common.WORK / "masks")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(frames)} frames, {len(chunks)} SAM 3.1 video chunks at "
          f"{common.FPS_SRC / step:g} Hz", flush=True)

    sam = None
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {source}")
    written = []
    started = time.time()
    for ci, chunk_frames in selected:
        cache_file = out_dir / f"video_chunk_{chunk_frames[0]:06d}.npz"
        written.append(cache_file.name)
        if not args.redo and cache_ok(cache_file, chunk_frames, args.resolution):
            print(f"  chunk {ci + 1}/{len(chunks)} cached", flush=True)
            continue
        bgr = read_chunk(cap, chunk_frames)
        if len(bgr) != len(chunk_frames):
            chunk_frames = chunk_frames[:len(bgr)]
        if (not args.redo and
                upgradeable_cache(cache_file, chunk_frames, args.resolution)):
            found = upgrade_chunk(
                cache_file, bgr, chunk_frames, boxes, args.resolution
            )
            print(f"  chunk {ci + 1}/{len(chunks)} upgraded: "
                  f"{found}/{2 * len(chunk_frames)} role masks", flush=True)
            continue
        data = {"schema": np.array(SCHEMA, np.int16),
                "quality_version": np.array(QUALITY_VERSION),
                "resolution": np.array(args.resolution, np.int16),
                "precision": np.array(sam["precision"] if sam else "fp32"),
                "chunk_F": np.asarray(chunk_frames, np.int32)}
        if sam is None:
            sam = sam31_video(args.resolution)
            data["precision"] = np.array(sam["precision"])
        rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in bgr]
        t0 = time.time()
        with sam["wired_limit"](sam["model"]):
            outputs, requested, found = segment_chunk(
                sam, rgb, chunk_frames, boxes
            )

        for q, role_masks in outputs.items():
            frame_no = chunk_frames[q]
            for role, mask in role_masks.items():
                if not mask.any():
                    continue
                data[f"f{frame_no}_{role}"] = common.rle_encode(mask)
                ys, xs = np.nonzero(mask)
                data[f"s{frame_no}_{role}"] = np.array(
                    [len(xs), xs.mean(), ys.mean()], np.float32
                )
                other = "blue" if role == "red" else "red"
                support = kit_colour_support(bgr[q], mask)
                data[f"c{frame_no}_{role}"] = np.array(
                    [support[role][0], support[other][0], support[role][1]],
                    np.float32
                )
        np.savez_compressed(cache_file, **data)
        print(f"  chunk {ci + 1}/{len(chunks)}: {found}/{requested} role masks, "
              f"{len(outputs)} frames in {time.time() - t0:.1f}s",
              flush=True)
        sam["mx"].clear_cache()

    cap.release()
    # A bounded worker deliberately does not publish a completion index.  The
    # default full worker validates the shared caches and remains the single
    # writer of that authoritative manifest.
    if len(selected) != len(all_chunks) or args.reverse:
        print(f"SAM 3.1 bounded worker done in "
              f"{(time.time() - started) / 60:.1f} min", flush=True)
        return
    precisions = set()
    for name in written:
        with np.load(out_dir / name) as chunk:
            precisions.add(str(chunk["precision"]) if "precision" in chunk
                           else "fp32")
    index = {
        "schema": SCHEMA,
        "quality_version": QUALITY_VERSION,
        "model": "SAM 3.1 concept segmentation (MLX; "
                 + " + ".join(sorted(precisions)) + ")",
        "checkpoint": "mlx-community/sam3.1-bf16",
        "resolution": args.resolution,
        "prompt": "person concept masks assigned by mask-level red/blue hogu "
                  "evidence plus temporal role boxes, with conservative on-mat recovery",
        "quality_gate": "largest connected SAM instance; tracked-box/person-shape "
                        "validation; conservative size, aspect, and vertical gates "
                        "for colour-only recovery",
        "roles": list(FIGHTERS),
        "wh": track["wh"],
        "stride": step,
        "fps": common.FPS_SRC / step,
        "F": frames,
        "chunks": written,
    }
    with open(out_dir / "masks_index.json", "w") as handle:
        json.dump(index, handle, indent=1)
    print(f"SAM 3.1 video masks done in {(time.time() - started) / 60:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()
