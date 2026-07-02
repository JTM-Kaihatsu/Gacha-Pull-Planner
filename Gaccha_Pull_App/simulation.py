
"""simulation.py
Core gacha‑pull simulation logic for Honkai: Star Rail.

All functions are pure and have no external side‑effects, which
makes them easy to unit‑test and reuse.
"""

from typing import Dict, Tuple
import random
import numpy as np
import pandas as pd

CHAR_PITY_DEFAULTS = {"base_rate": 0.006, "soft_pity_start": 73, "hard_pity": 90}
LC_PITY_DEFAULTS   = {"base_rate": 0.008, "soft_pity_start": 65, "hard_pity": 80}


def banner_probability(pity, base_rate, soft_pity_start, hard_pity):
    if pity < soft_pity_start:
        return base_rate
    if pity < hard_pity:
        increase = (1.0 - base_rate) / (hard_pity - soft_pity_start)
        return min(1.0, base_rate + (pity - soft_pity_start + 1) * increase)
    return 1.0


def simulate_combo_verbose(
    total_pulls: int,
    strategy: list,
    start_char_pity: int = 0,
    start_char_guarantee: bool = False,
    start_lc_pity: int = 0,
    start_lc_guarantee: bool = False,
    full_4star_chars: bool = True,
    simulate_4star: bool = True,
    char_pity_config: dict = None,
    lc_pity_config: dict = None,
) -> Tuple[bool, int, float, float, dict]:
    """
    Simulates pulls with metadata tracking for verbose analysis.

    strategy: ordered list of {"banner": "char"|"lc", "copies": int} phases.
    Pity and guarantee state carry over between phases, so order matters.

    Returns:
        - success (bool)
        - pulls_used (int)
        - refunded (float)
        - leftover (float)
        - metadata (dict)
    """
    char_cfg = char_pity_config or CHAR_PITY_DEFAULTS
    lc_cfg   = lc_pity_config   or LC_PITY_DEFAULTS

    remaining = total_pulls
    used_pulls = 0
    refunded_pulls = 0.0

    pity_char, guarantee_char = start_char_pity, start_char_guarantee
    pity_lc, guarantee_lc = start_lc_pity, start_lc_guarantee
    pity_4star_char = 0
    pity_4star_lc = 0

    char_50_50_wins = 0
    lc_75_25_wins = 0
    char_50_50_encounters = 0
    lc_75_25_encounters = 0
    lost_char_50_50_failure = False
    pity_char_trigger = 0
    pity_lc_trigger = 0

    desired_chars = sum(p["copies"] for p in strategy if p["banner"] == "char")
    desired_lcs   = sum(p["copies"] for p in strategy if p["banner"] == "lc")
    chars_obtained = 0
    lcs_obtained = 0

    phase_detail = []  # per-phase breakdown for visualization

    # Execute each phase in order. Pity/guarantee state is shared, so pulling
    # LC between two char phases genuinely affects the budget for the next phase.
    for phase in strategy:
        banner = phase["banner"]
        copies_needed = phase["copies"]
        copies_got = 0
        phase_pulls = 0
        phase_refunds = 0.0

        while copies_got < copies_needed and remaining > 0:
            if banner == "char":
                remaining -= 1
                used_pulls += 1
                phase_pulls += 1
                pity_char += 1
                pity_4star_char += 1

                hit_5star = np.random.rand() < banner_probability(pity_char, **char_cfg)

                if hit_5star:
                    pity_char_trigger = pity_char
                    char_50_50_encounters += 1

                    if guarantee_char or np.random.rand() < 0.5:
                        copies_got += 1
                        chars_obtained += 1
                        guarantee_char = False
                        char_50_50_wins += 1
                    else:
                        guarantee_char = True
                    pity_char = 0
                    pity_4star_char = 0

                elif simulate_4star and (pity_4star_char >= 10 or np.random.rand() < 0.051):
                    pity_4star_char = 0
                    if full_4star_chars:
                        remaining += 1
                        refunded_pulls += 1.0
                        phase_refunds += 1.0

            else:  # banner == "lc"
                remaining -= 1
                used_pulls += 1
                phase_pulls += 1
                pity_lc += 1
                pity_4star_lc += 1

                hit_5star = np.random.rand() < banner_probability(pity_lc, **lc_cfg)

                if hit_5star:
                    pity_lc_trigger = pity_lc
                    lc_75_25_encounters += 1

                    if guarantee_lc or np.random.rand() < 0.75:
                        copies_got += 1
                        lcs_obtained += 1
                        guarantee_lc = False
                        lc_75_25_wins += 1
                    else:
                        guarantee_lc = True
                    pity_lc = 0
                    pity_4star_lc = 0

                elif simulate_4star and (pity_4star_lc >= 10 or np.random.rand() < 0.06):
                    pity_4star_lc = 0
                    remaining += 0.4
                    refunded_pulls += 0.4
                    phase_refunds += 0.4

        phase_detail.append({
            "banner": banner,
            "copies": copies_needed,
            "pulls_used": phase_pulls,
            "refunds": round(phase_refunds, 2),
        })

    success = (chars_obtained >= desired_chars and lcs_obtained >= desired_lcs)
    pulls_leftover = remaining

    final_pity_char = pity_char
    final_pity_lc = pity_lc

    if not success and guarantee_char:
        lost_char_50_50_failure = True

    meta = {
        "char_50_50_wins": char_50_50_wins,
        "lc_75_25_wins": lc_75_25_wins,
        "pity_char_trigger": pity_char_trigger,
        "pity_lc_trigger": pity_lc_trigger,
        "lost_char_50_50_failure": int(lost_char_50_50_failure),
        "final_pity_char": final_pity_char,
        "final_pity_lc": final_pity_lc,
        "char_50_50_encounters": char_50_50_encounters,
        "lc_75_25_encounters": lc_75_25_encounters,
        "desired_chars": desired_chars,
        "desired_lcs": desired_lcs,
        "chars_obtained": chars_obtained,
        "lcs_obtained": lcs_obtained,
        "phase_detail": phase_detail,
    }

    return success, used_pulls, round(refunded_pulls, 2), round(pulls_leftover, 2), meta


VIZ_SAMPLE_SIZE = 1000


def _build_phase_labels(strategy):
    """Returns a label per strategy phase: E0/E1/E2... for char, S1/S2... for lc."""
    char_idx = 0
    lc_idx = 0
    labels = []
    for phase in strategy:
        n = phase["copies"]
        if phase["banner"] == "char":
            if n == 1:
                labels.append(f"E{char_idx}")
            else:
                labels.append(f"E{char_idx}–E{char_idx + n - 1}")
            char_idx += n
        else:
            if n == 1:
                labels.append(f"S{lc_idx + 1}")
            else:
                labels.append(f"S{lc_idx + 1}–S{lc_idx + n}")
            lc_idx += n
    return labels


def run_simulation_verbose(
    total_pulls,
    strategy,
    start_char_pity=0,
    start_char_guarantee=False,
    start_lc_pity=0,
    start_lc_guarantee=False,
    full_4star_chars=True,
    char_pity_config=None,
    lc_pity_config=None,
    trials=10000
):
    """
    Verbose Monte Carlo simulation with metadata aggregation.

    strategy: ordered list of {"banner": "char"|"lc", "copies": int} phases.
    """
    successes = 0
    pulls_used = []
    pulls_refunded = []
    pulls_leftover = []

    all_used = []
    all_refunded = []
    all_leftover = []

    # Track failure cases for pulls used and refunds
    fail_pulls = []
    fail_refunds = []
    fail_leftover = []

    metadata_logs = []
    phase_labels = _build_phase_labels(strategy)
    viz_reservoir = []  # reservoir sample of per-run phase detail

    for i in range(trials):
        success, used, refunded, leftover, meta = simulate_combo_verbose(
            total_pulls,
            strategy,
            start_char_pity,
            start_char_guarantee,
            start_lc_pity,
            start_lc_guarantee,
            full_4star_chars=full_4star_chars,
            char_pity_config=char_pity_config,
            lc_pity_config=lc_pity_config,
        )

        all_used.append(used)
        all_refunded.append(refunded)
        all_leftover.append(leftover)
        metadata_logs.append({**meta, "success": success})

        # Reservoir sampling: keep a representative sample of VIZ_SAMPLE_SIZE runs
        viz_entry = {
            "trial": i + 1,
            "success": success,
            "total_pulls_used": used,
            "phases": [
                {**pd_item, "label": phase_labels[j]}
                for j, pd_item in enumerate(meta["phase_detail"])
            ],
        }
        if i < VIZ_SAMPLE_SIZE:
            viz_reservoir.append(viz_entry)
        else:
            j = random.randint(0, i)
            if j < VIZ_SAMPLE_SIZE:
                viz_reservoir[j] = viz_entry

        if success:
            successes += 1
            pulls_used.append(used)
            pulls_refunded.append(refunded)
            pulls_leftover.append(leftover)

        else:
            fail_pulls.append(used)
            fail_refunds.append(refunded)

    failures = trials - successes
    success_rate = successes / trials

    df = pd.DataFrame(metadata_logs)

    # Overall stats
    avg_pulls_all = np.mean(all_used)
    avg_left_all = np.mean(all_leftover)

    ### Success stats ###

    # Averages across all successful runs
    avg_pulls_success = np.mean(pulls_used) if pulls_used else 0
    avg_refund_success = np.mean(pulls_refunded) if pulls_refunded else 0
    avg_left_success = np.mean(pulls_leftover) if pulls_leftover else 0

    # Updated: Calculate average pity for char and LC on success using DataFrame
    avg_pity_char_success = round(df[df.success]["pity_char_trigger"].mean(), 2) if not df[df.success].empty else None
    avg_pity_lc_success = round(df[df.success]["pity_lc_trigger"].mean(), 2) if not df[df.success].empty else None

    refund_rate_success = round(np.mean(pulls_refunded) / np.mean(pulls_used) if pulls_used else 0, 4)

    # Updated: Correctly calculate 50/50 and 75/25 win rates for successes

    # Sum the char_50_50_encounters and lc_75_25_encounters for successful runs to get accurate totals
    total_50_50s_encountered_in_successes = df[df.success]["char_50_50_encounters"].sum()
    total_25_75s_encountered_in_successes = df[df.success]["lc_75_25_encounters"].sum()

    won_50_50_runs_successes = df[(df.success) & (df.char_50_50_wins > 0)]["char_50_50_wins"].sum()
    won_25_75_runs_successes = df[(df.success) & (df.lc_75_25_wins > 0)]["lc_75_25_wins"].sum()

    successes_char_win_rate = f"{((won_50_50_runs_successes / total_50_50s_encountered_in_successes) if total_50_50s_encountered_in_successes > 0 else np.nan) * 100:.2f}%"
    successes_lc_win_rate = f"{((won_25_75_runs_successes / total_25_75s_encountered_in_successes) if total_25_75s_encountered_in_successes > 0 else np.nan) * 100:.2f}%"

    # ... (Existing win rate calculations for failures) ...

    ### Failure stats ###
    # Updated: Directly use fail_pulls list for calculating average pulls on failure
    avg_pulls_fail = np.mean(fail_pulls) if fail_pulls else 0
    avg_refund_fail = np.mean(fail_refunds) if fail_refunds else 0
    avg_left_fail = np.mean(fail_leftover) if fail_leftover else 0

    # Updated: Calculate average pity for char and LC on failure using DataFrame
    avg_pity_char_failure = round(df[~df.success]["final_pity_char"].mean(), 2) if not df[~df.success].empty else None
    avg_pity_lc_failure = round(df[~df.success]["final_pity_lc"].mean(), 2) if not df[~df.success].empty else None

    refund_rate_failure = round(np.mean(fail_refunds) / np.mean(fail_pulls) if fail_pulls else 0, 4)

    #Updated: Use .sum() on char_50_50_wins to get total 50/50 encounters in failed runs
    # Sum the char_50_50_encounters and lc_75_25_encounters for failed runs to get accurate totals
    total_50_50s_encountered_in_failures = df[~df.success]["char_50_50_encounters"].sum()
    total_25_75s_encountered_in_failures = df[~df.success]["lc_75_25_encounters"].sum()

    won_50_50_runs_failures = df[(~df.success) & (df.char_50_50_wins > 0)]["char_50_50_wins"].sum()
    won_25_75_runs_failures = df[(~df.success) & (df.lc_75_25_wins > 0)]["lc_75_25_wins"].sum()

    failure_char_win_rate = f"{((won_50_50_runs_failures / total_50_50s_encountered_in_failures) if total_50_50s_encountered_in_failures > 0 else np.nan) * 100:.2f}%"
    failure_lc_win_rate = f"{((won_25_75_runs_failures / total_25_75s_encountered_in_failures) if total_25_75s_encountered_in_failures > 0 else np.nan) * 100:.2f}%"

    desired_chars = sum(p["copies"] for p in strategy if p["banner"] == "char")
    desired_lcs   = sum(p["copies"] for p in strategy if p["banner"] == "lc")

    # Failure state distribution — most common (chars, lcs) combos among failed runs
    fail_df = df[~df.success]
    if not fail_df.empty:
        state_counts = (
            fail_df.groupby(["chars_obtained", "lcs_obtained"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        top_states = [
            {
                "chars": int(row["chars_obtained"]),
                "lcs": int(row["lcs_obtained"]),
                "pct": round(row["count"] / failures * 100, 1),
            }
            for _, row in state_counts.head(3).iterrows()
        ]
        most_common = top_states[0]
    else:
        top_states = []
        most_common = None

    # Correlation stats — for each success factor, count how many successful
    # and failed runs had that factor working in their favour. This lets the
    # AI reason about distributions rather than just averages.
    success_df = df[df.success]
    fail_df_c  = df[~df.success]

    avg_refunds_all = df["char_50_50_wins"].map(lambda _: None)  # placeholder reset
    overall_avg_refunds = np.mean(all_refunded) if all_refunded else 0

    # Refunds: how many runs had above-average refunds, split by outcome
    refund_col = pd.Series(all_refunded, index=df.index)
    df["refunds"] = refund_col
    success_above_avg_refunds = int((df[df.success]["refunds"] > overall_avg_refunds).sum())
    fail_above_avg_refunds    = int((df[~df.success]["refunds"] > overall_avg_refunds).sum())

    # Early pity: char pity trigger < 70 (well before soft pity at 73)
    early_pity_threshold = 70
    success_early_pity = int((df[df.success]["pity_char_trigger"] < early_pity_threshold).sum())
    fail_early_pity    = int((df[~df.success]["final_pity_char"] < early_pity_threshold).sum())

    # 50/50 wins: runs where char_50_50_wins >= desired_chars (won every needed 50/50)
    success_all_5050_won = int((df[df.success]["char_50_50_wins"] >= desired_chars).sum()) if desired_chars > 0 else None
    fail_all_5050_won    = int((df[~df.success]["char_50_50_wins"] >= desired_chars).sum()) if desired_chars > 0 else None

    # LC 75/25 wins: runs where lc_75_25_wins >= desired_lcs
    success_all_lc_won = int((df[df.success]["lc_75_25_wins"] >= desired_lcs).sum()) if desired_lcs > 0 else None
    fail_all_lc_won    = int((df[~df.success]["lc_75_25_wins"] >= desired_lcs).sum()) if desired_lcs > 0 else None

    correlation_stats = {
        "total_successes": successes,
        "total_failures": failures,
        "above_avg_refunds_in_successes": success_above_avg_refunds,
        "above_avg_refunds_in_failures": fail_above_avg_refunds,
        "early_char_pity_in_successes": success_early_pity,
        "early_char_pity_in_failures": fail_early_pity,
        "all_5050s_won_in_successes": success_all_5050_won,
        "all_5050s_won_in_failures": fail_all_5050_won,
        "all_lc_75s_won_in_successes": success_all_lc_won,
        "all_lc_75s_won_in_failures": fail_all_lc_won,
        "early_pity_threshold": early_pity_threshold,
        "overall_avg_refunds": round(overall_avg_refunds, 1),
    }

    stats_summary = {
        # Initial scenario
        "initial_pulls": total_pulls,
        "trials": trials,
        "strategy": strategy,
        "desired_characters": desired_chars,
        "desired_lightcones": desired_lcs,
        "success_rate": f"{success_rate * 100:.2f}%",
        "start_char_pity": start_char_pity,
        "start_char_guarantee": start_char_guarantee,
        "start_lc_pity": start_lc_pity,
        "start_lc_guarantee": start_lc_guarantee,

        # Stats for successful runs
        "avg_pity_char": round(df[df.success]["pity_char_trigger"].mean(), 2) if not df[df.success].empty else None,
        "avg_pity_lc": round(df[df.success]["pity_lc_trigger"].mean(), 2) if not df[df.success].empty else None,
        "successes_char_win_rate": successes_char_win_rate,
        "successes_lc_win_rate": successes_lc_win_rate,
        "avg_leftover_pulls_on_success": round(avg_left_success, 2),
        "avg_refund_success": round(avg_refund_success,0),

        # Stats for failed runs
        "failure_char_win_rate": failure_char_win_rate,
        "failure_lc_win_rate": failure_lc_win_rate,
        "avg_leftover_pulls_on_failure": round(avg_left_fail, 2),
        "avg_refund_fail": round(avg_refund_fail,0),
        "most_common_failure_state": most_common,
        "failure_state_distribution": top_states,
        "correlation_stats": correlation_stats,
        "viz_sample": viz_reservoir,
    }

    return stats_summary
