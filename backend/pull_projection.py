"""pull_projection.py
Projects how many pulls a player will have by a target date, from the pulls
they have now plus every income source between now and then (Honkai: Star
Rail's sources and amounts by default).

Every schedule and amount is a setting on ProjectionSettings rather than a
hidden constant, since players differ (endgame clear rates, which passes they
buy) and the game's own calendar is only regular until it isn't. The one thing
always needed from the caller is a known patch start date: patch-tied income
(new-patch compensation, character trials, Nameless Honor) is laid out on a
fixed cycle from it. Occasional one-off changes to the patch cycle aren't
modeled; extra_events covers one-off income such as anniversary rewards.

Events are counted strictly after the start date and up to and including the
end date: current_pulls is taken to already include anything collected on the
start date itself.
"""
from dataclasses import dataclass, field
from datetime import date, timedelta
import math


@dataclass
class ProjectionSettings:
    currency_per_pull: int = 160

    # Daily: training (dailies) always; the Express Supply Pass adds its daily
    # amount plus a lump sum at each renewal, only while it's owned.
    daily_training_jade: int = 60
    express_pass: bool = True
    express_pass_daily_jade: int = 90
    express_pass_renewal_jade: int = 300
    express_pass_renewal_days: int = 30
    # First renewal date; None means the projection's start date.
    express_pass_start: date | None = None

    # Weekly, credited on the weekly reset day (0 = Monday). Both are
    # averages so a partial clear is expressible: the endgame default is a
    # full clear of all three modes (3 x 800 jade every 42 days = 2.5 pulls a
    # week), Simulated Universe's is its full 225 jade.
    weekly_reset_weekday: int = 0
    weekly_endgame_pulls: float = 2.5
    weekly_simulated_universe_pulls: float = 225 / 160

    # Monthly.
    hoyolab_checkin_days: tuple = (4, 12, 19)
    hoyolab_checkin_jade: int = 20
    embers_exchange_pulls: int = 5  # on the 1st of each month

    # Patch cycle, laid out from a known patch start date (see generate).
    patch_length_days: int = 42
    banner_phase_days: int = 21
    patch_compensation_jade: int = 600       # maintenance + issue compensation
    patch_free_pulls: int = 0                # free pulls granted with a patch, if any
    trial_characters_per_phase: int = 2      # featured 5-stars with a trial each phase
    trial_jade_each: int = 20

    # Nameless Honor (premium battle pass), one per patch, rewards on days
    # after the patch starts as (day, jade, label).
    nameless_honor: bool = False
    nameless_honor_schedule: tuple = (
        (0, 680, "Nameless Honor Purchase Bonus"),
        (7, 160, "Nameless Honor Reward (1 Pull)"),
        (17, 320, "Nameless Honor Reward (2 Pulls)"),
        (21, 160, "Nameless Honor Reward (1 Pull)"),
    )

    # One-off income on specific dates, e.g. anniversary rewards: a list of
    # (date, label, jade).
    extra_events: list = field(default_factory=list)


def _validate(settings, start, end, patch_anchor):
    if end <= start:
        return "end date must be after the start date"
    if settings.currency_per_pull <= 0:
        return "currency_per_pull must be positive"
    if settings.patch_length_days <= 0 or settings.banner_phase_days <= 0:
        return "patch_length_days and banner_phase_days must be positive"
    if settings.express_pass_renewal_days <= 0:
        return "express_pass_renewal_days must be positive"
    if not 0 <= settings.weekly_reset_weekday <= 6:
        return "weekly_reset_weekday must be 0 (Monday) through 6 (Sunday)"
    if settings.weekly_endgame_pulls < 0 or settings.weekly_simulated_universe_pulls < 0:
        return "weekly averages cannot be negative"
    if patch_anchor is None:
        return "a known patch start date (patch_anchor) is required"
    return None


def _cycle_dates(anchor, period_days, start, end):
    """Every date anchor + k * period_days (k may be negative) that falls in
    (start, end]: the anchor can be any known cycle start, past or future."""
    period = timedelta(days=period_days)
    first_k = math.floor((start - anchor) / period) + 1
    d = anchor + first_k * period
    while d <= end:
        if d > start:
            yield d
        d += period


def _monthly_dates(day_of_month, start, end):
    year, month = start.year, start.month
    while date(year, month, 1) <= end:
        try:
            d = date(year, month, day_of_month)
        except ValueError:
            d = None  # e.g. the 31st in a 30-day month
        if d and start < d <= end:
            yield d
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)


def _weekly_dates(weekday, start, end):
    d = start + timedelta(days=(weekday - start.weekday()) % 7 or 7)
    while d <= end:
        yield d
        d += timedelta(weeks=1)


def projection_events(start, end, patch_anchor, settings=None):
    """Every income event in (start, end] as (date, label, pulls)."""
    s = settings or ProjectionSettings()
    per_pull = s.currency_per_pull
    events = []

    # Daily income, credited as one row per full week from the start date
    # plus one for any remaining days, so each day is counted exactly once.
    daily_jade = s.daily_training_jade + (s.express_pass_daily_jade if s.express_pass else 0)
    label = "Daily Training" + (" + Express Supply Pass" if s.express_pass else "")
    d = start
    while daily_jade and d < end:
        days = min(7, (end - d).days)
        d += timedelta(days=days)
        events.append((d, f"{label} ({days} day{'s' if days != 1 else ''})", days * daily_jade / per_pull))

    if s.express_pass:
        renewal_start = s.express_pass_start or start
        for d in _cycle_dates(renewal_start, s.express_pass_renewal_days, start, end):
            if d >= renewal_start:
                events.append((d, "Express Supply Pass Renewal", s.express_pass_renewal_jade / per_pull))

    for d in _weekly_dates(s.weekly_reset_weekday, start, end):
        if s.weekly_endgame_pulls:
            events.append((d, "Endgame Modes (weekly average)", s.weekly_endgame_pulls))
        if s.weekly_simulated_universe_pulls:
            events.append((d, "Simulated Universe (weekly)", s.weekly_simulated_universe_pulls))

    for day in s.hoyolab_checkin_days if s.hoyolab_checkin_jade else ():
        for d in _monthly_dates(day, start, end):
            events.append((d, "HoYoLAB Check-in", s.hoyolab_checkin_jade / per_pull))
    if s.embers_exchange_pulls:
        for d in _monthly_dates(1, start, end):
            events.append((d, "Embers Exchange Reset", s.embers_exchange_pulls))

    for d in _cycle_dates(patch_anchor, s.patch_length_days, start, end):
        if s.patch_compensation_jade:
            events.append((d, "New Patch Compensation", s.patch_compensation_jade / per_pull))
        if s.patch_free_pulls:
            events.append((d, "New Patch Free Pulls", s.patch_free_pulls))
    trial_jade = s.trial_characters_per_phase * s.trial_jade_each
    if trial_jade:
        for d in _cycle_dates(patch_anchor, s.banner_phase_days, start, end):
            events.append((d, "Character Trials", trial_jade / per_pull))

    if s.nameless_honor:
        # Each patch's pass pays out on days after that patch starts, so a
        # patch that started before the projection can still pay out in it.
        latest_offset = max(day for day, _, _ in s.nameless_honor_schedule)
        for patch_start in _cycle_dates(patch_anchor, s.patch_length_days,
                                        start - timedelta(days=latest_offset + 1), end):
            for day, jade, reward in s.nameless_honor_schedule:
                d = patch_start + timedelta(days=day)
                if start < d <= end:
                    events.append((d, reward, jade / per_pull))

    for d, reward, jade in s.extra_events:
        if start < d <= end:
            events.append((d, reward, jade / per_pull))

    # Stable within a date: the order sources were added above.
    return sorted(events, key=lambda e: e[0])


def generate_pull_projection(current_pulls, start, end, patch_anchor, settings=None):
    """Project pulls from `current_pulls` on `start` to `end`.

    Returns {"error": ...} for invalid input, otherwise {"rows": [...],
    "total_pulls": float, "projected_pulls": int} where each row is
    {"date", "source", "pulls", "cumulative_pulls"} and projected_pulls is
    the whole pulls available by the end date (rounded down)."""
    s = settings or ProjectionSettings()
    error = _validate(s, start, end, patch_anchor)
    if error:
        return {"error": error}

    cumulative = float(current_pulls)
    rows = [{"date": start, "source": "Starting Pulls", "pulls": 0.0, "cumulative_pulls": cumulative}]
    for d, source, pulls in projection_events(start, end, patch_anchor, s):
        cumulative += pulls
        rows.append({"date": d, "source": source, "pulls": round(pulls, 2),
                     "cumulative_pulls": round(cumulative, 2)})
    return {"rows": rows, "total_pulls": round(cumulative, 2), "projected_pulls": math.floor(cumulative + 1e-9)}
