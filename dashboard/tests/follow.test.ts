import { describe, expect, test } from "bun:test";

import { FollowTracker, HANDOVER_DELAY_MS } from "../src/replay/follow";

describe("Voiture suivie", () => {
  test("is linked to the selected car by default", () => {
    expect(new FollowTracker().linked).toBe(true);
  });

  test("stays 2 s on a car that becomes inactive, then moves to the best active car", () => {
    const tracker = new FollowTracker();
    tracker.follow(2, true);
    expect(tracker.update(0, true, () => 1)).toBeNull();
    // The car has a DNF at t = 1000 ms.
    expect(tracker.update(1000, false, () => 1)).toBeNull();
    expect(tracker.update(1000 + HANDOVER_DELAY_MS - 1, false, () => 1)).toBeNull();
    expect(tracker.update(1000 + HANDOVER_DELAY_MS, false, () => 1)).toBe(1);
  });

  test("hands over once, not at every frame after the delay", () => {
    const tracker = new FollowTracker();
    tracker.follow(0, true);
    tracker.update(0, false, () => null);
    // No active car is left: nothing to follow, and no new attempt at each frame.
    expect(tracker.update(HANDOVER_DELAY_MS, false, () => null)).toBeNull();
    expect(tracker.update(HANDOVER_DELAY_MS * 2, false, () => 3)).toBeNull();
  });

  test("cancels the handover when a seek brings the car back in the race", () => {
    const tracker = new FollowTracker();
    tracker.follow(0, true);
    tracker.update(0, false, () => 1);
    expect(tracker.update(500, true, () => 1)).toBeNull();
    expect(tracker.update(HANDOVER_DELAY_MS + 100, true, () => 1)).toBeNull();
    // It becomes inactive again later: the 2 s start again from that moment.
    expect(tracker.update(5000, false, () => 1)).toBeNull();
    expect(tracker.update(5000 + HANDOVER_DELAY_MS, false, () => 1)).toBe(1);
  });

  test("keeps the camera on an inactive car that the user chose", () => {
    const tracker = new FollowTracker();
    tracker.follow(3, false);
    expect(tracker.update(0, false, () => 0)).toBeNull();
    expect(tracker.update(HANDOVER_DELAY_MS * 3, false, () => 0)).toBeNull();
    expect(tracker.followed).toBe(3);
  });
});
