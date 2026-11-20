/**
 * Voiture suivie: the car that the Chase camera follows. It is linked to the Voiture
 * sélectionnée by default. When it becomes inactive, the camera stays for 2 s, then the
 * best-ranked active car takes over.
 */

/** Time the camera stays on a car that just became inactive, in milliseconds. */
export const HANDOVER_DELAY_MS = 2000;

export class FollowTracker {
  /** True while the followed car is the selected car. */
  linked = true;
  private car = 0;
  private wasActive = false;
  /** Wall-clock time when the followed car was seen to become inactive, or null. */
  private inactiveSince: number | null = null;

  get followed(): number {
    return this.car;
  }

  /**
   * Follows `car` from now. `active` is its state at the playhead. A car that is already
   * inactive when it is chosen is not handed over: the user asked to look at it.
   */
  follow(car: number, active: boolean): void {
    this.car = car;
    this.wasActive = active;
    this.inactiveSince = null;
  }

  /**
   * Call at each frame with the wall-clock time and the state of the followed car. Returns the
   * car to follow next when the 2 s are over, or null. `bestActive` gives the best-ranked
   * active car at the playhead, or null when no car races.
   */
  update(now: number, active: boolean, bestActive: () => number | null): number | null {
    if (active) {
      // Also after a seek back to a time when the car still raced.
      this.wasActive = true;
      this.inactiveSince = null;
      return null;
    }
    if (this.wasActive) {
      this.wasActive = false;
      this.inactiveSince = now;
    }
    if (this.inactiveSince === null || now - this.inactiveSince < HANDOVER_DELAY_MS) return null;
    this.inactiveSince = null;
    return bestActive();
  }
}
