"""Transfers are measured from pixels across surfaces, rather than screen activity."""
import json
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from core.perception.observed_transfer import (
    MAX_EDGE,
    PixelSnapshot,
    ScreenReading,
    TransferReceipt,
    _transfer_receipts_require_evidence,
    bind_transfer,
    finish_delivery,
    make_snapshot,
    revalidate_placement,
    verify_transfer,
)
from core.runtime.skill_contract import PredicateState

WIDTH, HEIGHT = 320, 240
SOURCE = (20, 30, 32, 32)
TARGET = (200, 130, 32, 32)


def pattern():
    rows, cols = np.indices((32, 32))
    return np.stack(((rows * 9 + cols * 3) % 251,
                     (rows * 3 + cols * 11) % 241,
                     ((cols // 8 + rows // 8) % 2) * 160 + 30), axis=2).astype(np.uint8)


def pixels(*places):
    image = np.full((HEIGHT, WIDTH, 3), 224, np.uint8)
    for place, appearance in places:
        x, y, width, height = place
        image[y:y + height, x:x + width] = appearance
    return image


def shape(label, place, *, object_extent=True):
    x, y, width, height = place
    return {"text": label, "x": x / WIDTH, "y": y / HEIGHT,
            "width": width / WIDTH, "height": height / HEIGHT, "shape": object_extent}


def point(place):
    x, y, width, height = place
    return ((x + width / 2) / WIDTH, (y + height / 2) / HEIGHT)


def reading(image, *, at, regions=(), surface="workspace", settled=True, **changes):
    public = {"capture_at": at, "_capture_epoch": f"frame-{at}", "owner": surface,
              "bounds": [30, 70, WIDTH, HEIGHT], "settled": settled,
              "shapes": list(regions), "layout": [], **changes}
    return ScreenReading(public, picture_snapshot=make_snapshot(image))


def intent(before=None, *, target=TARGET, delivered=True):
    if before is None:
        before = reading(pixels((SOURCE, pattern())), at=1, regions=[shape("item", SOURCE)])
    bound = bind_transfer("carry item to destination", before, source="item", destination="destination",
                          source_at=point(SOURCE), destination_at=point(target))
    return finish_delivery(bound, started=1.1, completed=1.2, delivered=delivered)


@pytest.mark.parametrize("surface", ["image editor", "inventory", "diagram canvas", "browser form"])
@pytest.mark.parametrize("copied", [False, True])
def test_new_occupant_is_verified_on_different_surfaces(surface, copied):
    before = reading(pixels((SOURCE, pattern())), at=1, regions=[shape("item", SOURCE)], surface=surface)
    locations = [(TARGET, pattern())] + ([(SOURCE, pattern())] if copied else [])
    # The object need not occur in the bounded post-shape detector list.
    after = reading(pixels(*locations), at=2, surface=surface)
    receipt = verify_transfer(intent(before), after)
    assert receipt.state == PredicateState.SATISFIED
    assert receipt.effect == ("copied" if copied else "moved")
    assert receipt.effect_bbox == pytest.approx((200 / WIDTH, 130 / HEIGHT, 32 / WIDTH, 32 / HEIGHT))
    assert len(receipt.appearance) == 16 * 16 * 3
    assert not hasattr(receipt, "before")


def test_wrong_source_with_same_geometry_does_not_prove_transfer():
    wrong = 255 - pattern()
    after = reading(pixels((TARGET, wrong)), at=2, regions=[shape("item", TARGET)])
    assert verify_transfer(intent(), after).state == PredicateState.UNSATISFIED


def test_animation_or_highlight_does_not_prove_transfer():
    after_pixels = pixels((SOURCE, pattern()))
    after_pixels[130:162, 200:232] = [20, 220, 20]
    after_pixels[0:20, 20:80] = pattern()[:20, :30].repeat(2, axis=1)
    after = reading(after_pixels, at=2, regions=[shape("item", TARGET)],
                    layout=[shape("item", TARGET, object_extent=False)])
    assert verify_transfer(intent(), after).state == PredicateState.UNSATISFIED


def test_moving_only_recognized_writing_has_no_object_extent():
    before = reading(pixels((SOURCE, pattern())), at=1,
                     layout=[shape("item", SOURCE, object_extent=False)])
    after = reading(pixels((TARGET, pattern())), at=2,
                    layout=[shape("item", TARGET, object_extent=False)])
    assert verify_transfer(intent(before), after).state == PredicateState.UNKNOWN


def test_named_writing_can_ground_an_enclosing_measured_object():
    before = reading(pixels((SOURCE, pattern())), at=1,
                     regions=[shape("object", SOURCE)],
                     layout=[shape("item", (25, 35, 10, 10), object_extent=False)])
    bound = bind_transfer("carry", before, source="item", destination="place",
                          source_at=(30 / WIDTH, 40 / HEIGHT), destination_at=(210 / WIDTH, 140 / HEIGHT))
    bound = finish_delivery(bound, started=1.1, completed=1.2, delivered=True)
    assert verify_transfer(bound, reading(pixels((TARGET, pattern())), at=2)).state == PredicateState.SATISFIED


@pytest.mark.parametrize("change", [{"capture_at": 1.15}, {"capture_at": 1.2},
                                   {"_capture_epoch": "frame-1"}, {"settled": False},
                                   {"owner": "other"}, {"bounds": [31, 70, WIDTH, HEIGHT]},
                                   {"_work_epoch": "new task"}, {"occluded": True}])
def test_stale_or_unmatched_observation_is_unknown(change):
    after = reading(pixels((TARGET, pattern())), at=2, **change)
    assert verify_transfer(intent(), after).state == PredicateState.UNKNOWN


def test_unsettled_before_is_unknown_even_when_destination_looks_correct():
    before = reading(pixels((SOURCE, pattern())), at=1, regions=[shape("item", SOURCE)], settled=False)
    assert verify_transfer(intent(before), reading(pixels((TARGET, pattern())), at=2)).state == PredicateState.UNKNOWN


def test_delivered_false_is_not_a_success():
    after = reading(pixels((TARGET, pattern())), at=2)
    assert verify_transfer(intent(delivered=False), after).state == PredicateState.UNSATISFIED


def test_multiple_matching_occupants_near_endpoint_are_unknown():
    left, right = (184, 130, 32, 32), (216, 130, 32, 32)
    after = reading(pixels((left, pattern()), (right, pattern())), at=2,
                    regions=[shape("one", left), shape("two", right)])
    assert verify_transfer(intent(), after).state == PredicateState.UNKNOWN


def test_target_already_occupied_before_delivery_is_not_a_new_build():
    before = reading(pixels((SOURCE, pattern()), (TARGET, pattern())), at=1,
                     regions=[shape("item", SOURCE)])
    after = reading(pixels((SOURCE, pattern()), (TARGET, pattern())), at=2)
    receipt = verify_transfer(intent(before), after)
    assert receipt.state == PredicateState.UNSATISFIED
    assert "already" in receipt.reason


def test_repeated_command_has_new_attempt_identity_and_can_measure_a_new_occupant():
    first = intent()
    first_after = reading(pixels((SOURCE, pattern()), (TARGET, pattern())), at=2)
    first_receipt = verify_transfer(first, first_after)
    next_target = (250, 180, 32, 32)
    next_before = reading(pixels((SOURCE, pattern()), (TARGET, pattern())), at=3,
                          regions=[shape("item", SOURCE)])
    second = bind_transfer(first.move, next_before, source="item", destination="destination",
                           source_at=point(SOURCE), destination_at=point(next_target))
    second = finish_delivery(second, started=3.1, completed=3.2, delivered=True)
    second_after = reading(pixels((SOURCE, pattern()), (TARGET, pattern()), (next_target, pattern())), at=4)
    assert first.attempt_id != second.attempt_id
    assert first_receipt.state == PredicateState.SATISFIED
    assert verify_transfer(second, second_after).state == PredicateState.SATISFIED


def test_source_point_names_alone_and_flat_appearances_remain_unknown():
    before = reading(pixels((SOURCE, pattern())), at=1,
                     regions=[{"text": "item", "center_x": point(SOURCE)[0], "center_y": point(SOURCE)[1], "shape": True}])
    assert verify_transfer(intent(before), reading(pixels((TARGET, pattern())), at=2)).state == PredicateState.UNKNOWN
    flat = np.full((32, 32, 3), [20, 60, 230], np.uint8)
    before = reading(pixels((SOURCE, flat)), at=1, regions=[shape("item", SOURCE)])
    assert verify_transfer(intent(before), reading(pixels((TARGET, flat)), at=2)).state == PredicateState.UNKNOWN


def test_missing_snapshot_cannot_be_replaced_by_asserted_appearance_or_change():
    before = dict(reading(pixels((SOURCE, pattern())), at=1, regions=[shape("item", SOURCE)]))
    before["shapes"][0]["look"] = tuple(float(i % 2) for i in range(48))
    after = reading(pixels((TARGET, pattern())), at=2, changed=True, built=True)
    assert verify_transfer(intent(before), after).state == PredicateState.UNKNOWN


def test_frozen_intent_does_not_follow_mutated_observation_or_image():
    image = pixels((SOURCE, pattern()))
    before = reading(image, at=1, regions=[shape("item", SOURCE)])
    bound = intent(before)
    before["shapes"][0]["x"] = 0.9
    before["owner"] = "other"
    image[:] = 0
    receipt = verify_transfer(bound, reading(pixels((TARGET, pattern())), at=2))
    assert receipt.state == PredicateState.SATISFIED
    with pytest.raises(FrozenInstanceError):
        receipt.state = PredicateState.UNSATISFIED


def test_retained_placement_is_rechecked_from_current_pixels_without_shape_list():
    after = reading(pixels((TARGET, pattern())), at=2)
    receipt = verify_transfer(intent(), after)
    assert revalidate_placement(receipt, after) == PredicateState.UNKNOWN
    current = reading(pixels((TARGET, pattern())), at=3)
    assert revalidate_placement(receipt, current) == PredicateState.SATISFIED
    removed = reading(pixels(), at=4)
    assert revalidate_placement(receipt, removed) == PredicateState.UNSATISFIED
    obscured = reading(pixels(), at=4, occluded=True)
    assert revalidate_placement(receipt, obscured) == PredicateState.UNKNOWN
    edited = reading(pixels((TARGET, 255 - pattern())), at=4)
    assert revalidate_placement(receipt, edited) == PredicateState.UNSATISFIED
    reset = reading(pixels((TARGET, pattern())), at=4, _work_epoch="reset")
    assert revalidate_placement(receipt, reset) == PredicateState.UNKNOWN


def test_retained_placement_does_not_survive_ambiguous_matching_occupants():
    receipt = verify_transfer(intent(), reading(pixels((TARGET, pattern())), at=2))
    left, right = (184, 130, 32, 32), (216, 130, 32, 32)
    current = reading(pixels((left, pattern()), (right, pattern())), at=3,
                      regions=[shape("one", left), shape("two", right)])
    # Neither neighboring object occupies the retained extent.
    assert revalidate_placement(receipt, current) == PredicateState.UNSATISFIED


def test_relocating_one_object_cannot_keep_credit_for_its_previous_placement():
    receipt = verify_transfer(intent(), reading(pixels((TARGET, pattern())), at=2))
    relocated = (220, 130, 32, 32)
    before = reading(pixels((TARGET, pattern())), at=3, regions=[shape("item", TARGET)])
    bound = bind_transfer("carry item elsewhere", before, source="item", destination="new place",
                          source_at=point(TARGET), destination_at=point(relocated))
    bound = finish_delivery(bound, started=3.1, completed=3.2, delivered=True)
    after = reading(pixels((relocated, pattern())), at=4)
    assert verify_transfer(bound, after).state == PredicateState.SATISFIED
    assert revalidate_placement(receipt, after) == PredicateState.UNSATISFIED


def test_retained_placement_requires_matching_capture_dimensions():
    receipt = verify_transfer(intent(), reading(pixels((TARGET, pattern())), at=2))
    current = reading(np.repeat(pixels((TARGET, pattern())), 2, axis=0), at=3)
    assert revalidate_placement(receipt, current) == PredicateState.UNKNOWN


def test_pixel_bytes_are_bounded_private_and_immutable():
    original = np.zeros((1200, 2400, 3), np.uint8)
    snapshot = make_snapshot(original)
    assert (snapshot.height, snapshot.width) == (480, MAX_EDGE)
    original[:] = 255
    assert not any(snapshot.data)
    public = ScreenReading({"owner": "app", "settled": True}, picture_snapshot=snapshot)
    assert json.loads(json.dumps(public)) == {"owner": "app", "settled": True}
    assert "data" not in repr(snapshot)
    assert "picture_snapshot" not in repr(public)
    assert "_picture_snapshot" not in dict(public)
    with pytest.raises(ValueError):
        PixelSnapshot(bytearray(12), 2, 2)
    assert make_snapshot(np.zeros((2, 2), np.uint8)) is None
    assert make_snapshot(np.zeros((2, 2, 3), np.float32)) is None


def test_transfer_evidence_invariant():
    assert _transfer_receipts_require_evidence() == ()


def test_stored_placement_remains_bounded_and_needs_current_revalidation():
    receipt = verify_transfer(intent(), reading(pixels((TARGET, pattern())), at=2, surface_id="page-1"))
    # A different surface identifier requires a fresh matching before capture.
    assert receipt.state == PredicateState.UNKNOWN
    before = reading(pixels((SOURCE, pattern())), at=1, regions=[shape("item", SOURCE)], surface_id="page-1")
    receipt = verify_transfer(intent(before), reading(pixels((TARGET, pattern())), at=2, surface_id="page-1"))
    encoded = json.loads(json.dumps(receipt.as_memory()))
    assert len(encoded["appearance"]) == 1024
    restored = TransferReceipt.from_memory(encoded)
    assert restored == receipt
    assert revalidate_placement(restored, reading(pixels((TARGET, pattern())), at=3, surface_id="page-1")) == PredicateState.SATISFIED
    assert revalidate_placement(restored, reading(pixels((TARGET, pattern())), at=3, surface_id="page-2")) == PredicateState.UNKNOWN


@pytest.mark.parametrize("change", [{"appearance": "x" * 1025}, {"appearance": "%%%%"},
                                   {"appearance": ""}, {"effect_bbox": [-0.1, 0.1, 0.1, 0.1]},
                                   {"effect_bbox": [0, 0, float("nan"), 0.1]}, {"effect": []},
                                   {"surface": [[[], "app"]]}, {"surface": [["owner", "a"], ["owner", "b"]]},
                                   {"bounds": [0, 0, 0, 1]}, {"attempt_id": "carry"},
                                   {"after_epoch": "frame-1"}, {"version": True},
                                   {"pixel_size": [2, 961]}, {"pixel_size": [True, 2]},
                                   {"pixel_size": []}])
def test_malformed_stored_witness_is_rejected(change):
    receipt = verify_transfer(intent(), reading(pixels((TARGET, pattern())), at=2))
    assert TransferReceipt.from_memory({**receipt.as_memory(), **change}) is None


def test_invalid_dispatch_times_and_outside_points_are_unknown():
    after = reading(pixels((TARGET, pattern())), at=2)
    assert verify_transfer(replace(intent(), started=float("nan")), after).state == PredicateState.UNKNOWN
    bound = bind_transfer("carry", intent().before, source="item", destination="place",
                          source_at=(float("nan"), 0.2), destination_at=point(TARGET))
    bound = finish_delivery(bound, started=1.1, completed=1.2, delivered=True)
    assert verify_transfer(bound, after).state == PredicateState.UNKNOWN
