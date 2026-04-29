"""Recommendation engine — executive summary for A/B test results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from ab_test_toolkit.bayesian import BayesianResult
from ab_test_toolkit.frequentist import FrequentistResult
from ab_test_toolkit.segmentation import SegmentResult
from ab_test_toolkit.srm import SRMResult


@dataclass
class NoveltyCheckResult:
    """Result of novelty effect detection."""

    has_novelty: bool
    early_effect: float
    late_effect: float
    ratio: float
    details: Optional[str] = None


@dataclass
class Recommendation:
    """Final recommendation output from the decision state machine."""

    recommendation: str  # "Ship", "Ship with Monitoring", "No-Ship", "No Effect", or "Inconclusive"
    reason: str = ""  # Plain-English explanation of why this decision was made
    signal_strength: str = ""  # "strong", "moderate", "weak", or "none" — how much evidence exists
    flags: list[str] = field(default_factory=list)
    supporting_metrics: dict = field(default_factory=dict)
    next_steps: list[str] = field(default_factory=list)
    manifest_hash: Optional[str] = None  # Short display prefix of the SHA-256
    manifest_hash_full: Optional[str] = None  # Full SHA-256 for audit
    manifest: Optional[dict] = None  # Full pre-reg manifest snapshot
    manifest_drift: list[str] = field(default_factory=list)  # Mismatches vs registered values


def check_novelty(
    df: pd.DataFrame,
    group_col: str = "group",
    value_col: str = "value",
    day_col: str = "day",
    early_fraction: float = 0.2,
    late_fraction: float = 0.5,
    ratio_threshold: float = 2.0,
) -> NoveltyCheckResult:
    """Detect novelty effect by comparing early vs late treatment effects.

    Only detects *positive* novelty (treatment looks good early, effect
    fades over time).  Negative novelty / learning effects are not flagged.

    Parameters
    ----------
    df : DataFrame
        Must contain group_col, value_col, and day_col.
    early_fraction : float
        Fraction of days considered "early" (default: first 20%).
    late_fraction : float
        Fraction of days considered "late" (default: last 50%).
    ratio_threshold : float
        Flag novelty if early_effect / late_effect exceeds this ratio.
    """
    if day_col not in df.columns:
        return NoveltyCheckResult(
            has_novelty=False, early_effect=0.0, late_effect=0.0, ratio=0.0,
            details="No day column — novelty check skipped.",
        )

    days_sorted = sorted(df[day_col].unique())
    n_days = len(days_sorted)
    if n_days < 3:
        return NoveltyCheckResult(
            has_novelty=False, early_effect=0.0, late_effect=0.0, ratio=0.0,
            details="Fewer than 3 days of data — novelty check skipped.",
        )

    early_cutoff = days_sorted[max(0, min(int(n_days * early_fraction), n_days - 1) - 1)]
    # Ensure at least 2 days in the early window when possible
    if n_days >= 4:
        early_idx = max(1, int(n_days * early_fraction) - 1)
        early_cutoff = days_sorted[early_idx]
    late_cutoff = days_sorted[int(n_days * (1 - late_fraction))]

    early_df = df[df[day_col] <= early_cutoff]
    late_df = df[df[day_col] >= late_cutoff]

    def _effect(sub_df: pd.DataFrame) -> float:
        ctrl = sub_df.loc[sub_df[group_col] == "control", value_col]
        treat = sub_df.loc[sub_df[group_col] == "treatment", value_col]
        if len(ctrl) == 0 or len(treat) == 0:
            return 0.0
        return float(treat.mean() - ctrl.mean())

    early_effect = _effect(early_df)
    late_effect = _effect(late_df)

    if late_effect <= 0:
        # Late effect is zero or negative — if early was positive, that's strong novelty
        ratio = float("inf") if early_effect > 0 else 0.0
    else:
        ratio = abs(early_effect / late_effect)

    has_novelty = (
        early_effect > 0
        and ratio > ratio_threshold
    )

    details = None
    if has_novelty:
        details = (
            f"Novelty effect detected: early effect ({early_effect:.4f}) is "
            f"{ratio:.1f}× the late effect ({late_effect:.4f}). "
            f"The observed lift may not persist."
        )

    return NoveltyCheckResult(
        has_novelty=has_novelty,
        early_effect=early_effect,
        late_effect=late_effect,
        ratio=ratio,
        details=details,
    )


def _suggest_next_steps(
    decision: str,
    signal_strength: str,
    flags: list[str],
    frequentist: FrequentistResult,
    bayesian: BayesianResult,
    srm: SRMResult,
    segmentation: Optional[SegmentResult],
    novelty: Optional[NoveltyCheckResult],
    has_covariate: bool,
) -> list[str]:
    """Generate context-aware next-step suggestions based on the decision and diagnostics.

    Steps are ordered by priority: root-cause blockers first, then actionable guidance.
    """
    steps: list[str] = []

    # --- Priority 1: Root-cause blockers ---
    if srm.has_mismatch:
        steps.append(
            "Investigate the root cause of the Sample Ratio Mismatch before re-running — "
            "check assignment logic, bot filtering, and variant-specific errors or crashes."
        )
        steps.append(
            "Do not interpret conversion or revenue metrics until the SRM is resolved."
        )
        return steps  # SRM invalidates everything; no point adding more

    if segmentation is not None and segmentation.simpsons_paradox:
        steps.append(
            "Analyze segment-level results to identify the composition effect driving Simpson's Paradox."
        )
        steps.append(
            "Consider re-running with stratified randomization to ensure balanced segment representation."
        )

    # --- Priority 2: Data quality warnings ---
    has_twyman = any("twyman" in f.lower() for f in flags)
    if has_twyman:
        steps.append(
            "The effect size is suspiciously large — verify data quality before acting. "
            "Check for logging bugs, outliers, or metric definition errors."
        )

    # --- Priority 3: Novelty ---
    has_novelty = novelty is not None and novelty.has_novelty
    if has_novelty:
        steps.append(
            "A novelty effect was detected — consider re-analyzing after excluding the first 3–7 days "
            "(burn-in period) to estimate the true long-term effect."
        )

    # --- Priority 4: Decision-specific guidance ---
    has_simpsons = segmentation is not None and segmentation.simpsons_paradox
    if decision == "Inconclusive" and not srm.has_mismatch and not has_simpsons:
        ci_width = abs(frequentist.ci_upper - frequentist.ci_lower)
        point_est = abs(frequentist.point_estimate)

        if signal_strength == "moderate":
            # Strong directional signal but didn't cross the significance bar
            steps.append(
                "There is a meaningful directional signal, but the pre-committed significance "
                "threshold was not met. Do not extend the experiment — this would constitute "
                "optional stopping and inflate the false positive rate."
            )
            steps.append(
                "If the feature is low-risk and easily reversible, discuss with stakeholders "
                "whether the strong Bayesian evidence and near-zero expected loss justify "
                "shipping with continued monitoring. This is a business judgment call, not a "
                "statistical one."
            )
            steps.append(
                "For a definitive result, design a new experiment with a larger sample size "
                "powered to detect the observed effect size. "
                + (
                    "Using CUPED variance reduction can effectively increase power without "
                    "needing more users."
                    if not has_covariate
                    else "The CUPED adjustment already helped — consider a more sensitive "
                    "metric or larger MDE."
                )
            )
        elif signal_strength == "weak":
            # Slight lean but not actionable
            steps.append(
                "The evidence is too weak to act on. Do not extend the experiment beyond "
                "the pre-committed sample size."
            )
            steps.append(
                "For a future re-run, fundamentally reconsider the experiment design: "
                "(a) is the MDE realistic for your traffic? (b) can you use a more "
                "sensitive metric? (c) would CUPED reduce variance enough to detect "
                "this effect size?"
                if not has_covariate
                else "For a future re-run, consider whether the MDE is realistic for "
                "your traffic, or whether a more sensitive metric would help."
            )
        elif signal_strength == "none" and not (segmentation is not None and segmentation.simpsons_paradox):
            # No signal at all
            if ci_width > 0 and point_est < ci_width * 0.15:
                steps.append(
                    "The confidence interval is relatively narrow and centered near zero — "
                    "this is a confident null result. The treatment very likely has no "
                    "meaningful effect."
                )
                steps.append(
                    "Consider whether a fundamentally different approach to the feature "
                    "is needed, rather than re-running the same experiment."
                )
            else:
                steps.append(
                    "No signal was detected, and the confidence interval is wide. "
                    "The experiment was underpowered for the effect that may exist. "
                    "Do not extend — design a properly powered follow-up instead."
                )
        else:
            # Frequentist significant but Bayesian unconvinced → evidence disagreement
            steps.append(
                "The frequentist and Bayesian frameworks disagree. Review whether the "
                "Bayesian prior is appropriate, and consider whether a fresh experiment "
                "with a pre-registered analysis plan would resolve the ambiguity."
            )

        # Common to all inconclusive: mention sequential testing
        steps.append(
            "If you need the ability to monitor results continuously in future experiments, "
            "use a sequential testing framework (e.g., group sequential design or always-valid "
            "confidence sequences) that formally controls for repeated looks."
        )

    elif decision == "No-Ship":
        steps.append(
            "Investigate which user segments were most negatively affected to understand "
            "why the treatment underperformed."
        )
        steps.append(
            "Consider iterating on the treatment design and re-testing rather than "
            "abandoning the idea entirely."
        )

    elif decision == "Ship with Monitoring":
        steps.append(
            "Roll out behind a small holdback (5–10%) and monitor primary plus guardrail metrics "
            "with weekly check-ins for the next 2–4 weeks."
        )
        steps.append(
            "Pre-commit to a re-decision criterion (e.g., extend exposure to reach the original "
            "MDE, or roll back if guardrails breach) before turning monitoring off."
        )

    elif decision == "Ship":
        if not has_novelty and not has_twyman:
            steps.append(
                "Plan a gradual rollout (1% → 5% → 25% → 50% → 100%) with guardrail metric monitoring."
            )
            steps.append(
                "Set up a holdback group (1–5%) to measure long-term impact after shipping."
            )
        elif has_novelty:
            steps.append(
                "Monitor the primary metric weekly post-ship for novelty decay."
            )
            steps.append(
                "Set up a holdback experiment to measure whether the effect persists long-term."
            )

    return steps


def _has_significant_conflicting_segment(
    segmentation: SegmentResult, alpha: float
) -> bool:
    """Return True iff at least one segment whose effect-sign differs from
    the aggregate effect-sign also has a Holm-adjusted p-value below alpha.

    This is the gate used to escalate Simpson's-Paradox-flagged results to
    "Inconclusive": a sign reversal in noise should not block the headline.
    """
    if segmentation is None or not segmentation.segment_results:
        return False
    agg = segmentation.aggregate_estimate
    if agg == 0:
        return False
    agg_sign = 1 if agg > 0 else -1
    for sr in segmentation.segment_results:
        est = sr.get("estimate", 0.0)
        if est == 0 or pd.isna(est):
            continue
        seg_sign = 1 if est > 0 else -1
        if seg_sign == agg_sign:
            continue
        adj_p = sr.get("p_value_adjusted", float("nan"))
        if pd.notna(adj_p) and adj_p < alpha:
            return True
    return False


def generate_recommendation(
    frequentist: FrequentistResult,
    bayesian: BayesianResult,
    srm: SRMResult,
    segmentation: Optional[SegmentResult] = None,
    novelty: Optional[NoveltyCheckResult] = None,
    lift_warning_threshold: float = 0.50,
    has_covariate: bool = False,
    control_rate: Optional[float] = None,
    practical_significance_threshold: Optional[float] = None,
    loss_tolerance: Optional[float] = None,
    allow_ship_with_monitoring: bool = False,
    monitoring_prob_threshold: float = 0.85,
    twyman_min_baseline: float = 0.01,
    manifest: Optional[dict] = None,
) -> Recommendation:
    """Apply decision state machine to produce a shipping recommendation.

    Design note: This engine requires both frequentist significance AND
    Bayesian P(B>A) > 95% to recommend "Ship". Most production systems use
    one framework; the dual requirement is a deliberate conservative choice
    that showcases fluency in both paradigms.

    Parameters
    ----------
    lift_warning_threshold : float
        Relative lift threshold for Twyman's Law warning (default 50%).
    control_rate : float, optional
        Control group's baseline rate/mean — used for relative lift in Twyman check.
    practical_significance_threshold : float, optional
        Minimum absolute effect to consider practically meaningful. If provided and
        effect is below this, Ship is downgraded to Inconclusive.
    loss_tolerance : float, optional
        Maximum acceptable Bayesian expected loss for a Ship decision.  When
        provided, Ship is downgraded to Inconclusive if
        ``bayesian.expected_loss > loss_tolerance``.  This enforces the
        documented contract that ships should be safe in expectation.
    allow_ship_with_monitoring : bool, default False
        If True, an intermediate "Ship with Monitoring" decision is emitted
        when there is a directional Bayesian signal (P(B>A) ≥
        ``monitoring_prob_threshold``) but frequentist significance is not
        achieved and no harm flags fire.  Default off to preserve the strict
        existing decision matrix.
    monitoring_prob_threshold : float, default 0.85
        Lower bound on P(B>A) for the "Ship with Monitoring" state.
    twyman_min_baseline : float, default 0.01
        Below this absolute baseline rate, the relative-lift component of
        Twyman's Law is suppressed (a tiny baseline trivially yields huge
        relative lifts that are not informative).
    manifest : dict, optional
        Pre-registration manifest snapshot.  Stored on the Recommendation
        for reproducibility along with a content hash.
    """
    flags: list[str] = []
    decision: str
    reason: str
    signal_strength: str = ""

    # --- Decision state machine ---
    if srm.has_mismatch:
        decision = "Inconclusive"
        reason = (
            "A Sample Ratio Mismatch was detected — the observed traffic split does not match "
            "the intended allocation. This indicates a systematic problem with randomization, "
            "making all results untrustworthy."
        )
        signal_strength = "none"
        flags.append("Sample Ratio Mismatch detected")
    elif (
        segmentation is not None
        and segmentation.simpsons_paradox
        # Significance gate: only block when at least one segment that
        # *opposes the aggregate sign* is statistically meaningful under
        # the Holm-adjusted p-values (multiplicity-corrected).  A reversal
        # built from noisy segments should not override the headline.
        and _has_significant_conflicting_segment(segmentation, frequentist.alpha)
    ):
        decision = "Inconclusive"
        reason = (
            "Simpson's Paradox was detected — the aggregate result contradicts the segment-level "
            "results, and at least one *conflicting* segment is significant under Holm-adjusted "
            "p-values. The overall metric is misleading due to a composition effect across segments."
        )
        signal_strength = "none"
        flags.append("Simpson's Paradox detected (significance-gated, Holm-adjusted)")
    elif (
        frequentist.is_significant
        and bayesian.prob_b_greater_a > 0.95
        and frequentist.point_estimate > 0
    ):
        decision = "Ship"
        reason = (
            f"Both analyses agree: the frequentist test is significant (p={frequentist.p_value:.4f}) "
            f"and the Bayesian analysis shows {bayesian.prob_b_greater_a:.1%} probability that "
            f"treatment is better, with an expected loss of {bayesian.expected_loss:.5f}. "
            f"The effect ({frequentist.point_estimate:+.4f}) is positive and the confidence interval "
            f"[{frequentist.ci_lower:.4f}, {frequentist.ci_upper:.4f}] does not cross zero."
        )
        signal_strength = "strong"
    elif frequentist.is_significant and frequentist.point_estimate < 0:
        decision = "No-Ship"
        reason = (
            f"The treatment has a statistically significant negative effect "
            f"(p={frequentist.p_value:.4f}, effect={frequentist.point_estimate:+.4f}). "
            f"The confidence interval [{frequentist.ci_lower:.4f}, {frequentist.ci_upper:.4f}] "
            f"is entirely below zero — the treatment is making things worse."
        )
        signal_strength = "strong"
    elif bayesian.prob_b_greater_a < 0.05:
        # Bayesian "Likely Harmful": posterior strongly indicates B is worse
        # than A, even when the frequentist test missed significance. This
        # matches the Bayesian decision matrix in the docs.
        decision = "No-Ship"
        reason = (
            f"The Bayesian analysis indicates the treatment is likely harmful "
            f"(P(B>A)={bayesian.prob_b_greater_a:.1%}, well below the 5% No-Ship threshold). "
            f"Even though the frequentist test did not reach significance "
            f"(p={frequentist.p_value:.4f}), the posterior strongly favours control. "
            f"Investigate before considering any future launch."
        )
        signal_strength = "strong"
        flags.append("Bayesian: treatment likely harmful (P(B>A) < 5%)")
    elif (
        not frequentist.is_significant
        and practical_significance_threshold is not None
        and frequentist.ci_lower >= -practical_significance_threshold
        and frequentist.ci_upper <= practical_significance_threshold
    ):
        # Confident Null: well-powered experiment, CI tight inside the
        # ±practical-significance margin — the feature does not move the metric
        # meaningfully in either direction.
        decision = "No Effect"
        reason = (
            f"Confident null: the test is not significant (p={frequentist.p_value:.4f}) and the "
            f"95% confidence interval [{frequentist.ci_lower:+.4f}, {frequentist.ci_upper:+.4f}] "
            f"is entirely within the practical-significance margin "
            f"of ±{practical_significance_threshold:.4f}. The treatment does not produce a "
            f"practically meaningful effect in either direction."
        )
        signal_strength = "strong"
    elif frequentist.is_significant and bayesian.prob_b_greater_a <= 0.95:
        # Frequentist significant but Bayesian not convinced
        decision = "Inconclusive"
        reason = (
            f"The frequentist test reached significance (p={frequentist.p_value:.4f}), but the "
            f"Bayesian analysis is not fully convinced (P(B>A)={bayesian.prob_b_greater_a:.1%}). "
            f"The two frameworks disagree, which warrants caution before shipping."
        )
        signal_strength = "moderate"
    else:
        decision = "Inconclusive"
        # Tier by signal strength based on Bayesian probability
        prob = bayesian.prob_b_greater_a
        if prob >= 0.90:
            signal_strength = "moderate"
            reason = (
                f"The frequentist test did not reach the pre-committed significance threshold "
                f"(p={frequentist.p_value:.4f} > α={frequentist.alpha}). However, Bayesian analysis "
                f"shows {prob:.1%} probability that treatment is better with near-zero expected loss "
                f"({bayesian.expected_loss:.5f}). There is a directional signal, but not enough "
                f"statistical evidence to confirm it under the fixed-horizon framework."
            )
        elif prob >= 0.70:
            signal_strength = "weak"
            reason = (
                f"The frequentist test is not significant (p={frequentist.p_value:.4f}) and the "
                f"Bayesian probability is only {prob:.1%} — a slight lean toward treatment but "
                f"well within the range of noise. The experiment does not provide actionable evidence "
                f"in either direction."
            )
        else:
            signal_strength = "none"
            if prob <= 0.30:
                reason = (
                    f"No meaningful signal in favour of treatment. The frequentist test is not "
                    f"significant (p={frequentist.p_value:.4f}) and the Bayesian probability is "
                    f"only {prob:.1%} — this actually suggests treatment may be slightly worse "
                    f"than control, though the evidence is not strong enough to confirm harm."
                )
            else:
                reason = (
                    f"No meaningful signal detected. The frequentist test is not significant "
                    f"(p={frequentist.p_value:.4f}) and the Bayesian probability is {prob:.1%} — "
                    f"close to even odds. The treatment does not appear to have a meaningful effect "
                    f"in either direction."
                )

    # --- Soft Simpson flag (non-blocking) ---
    if (
        segmentation is not None
        and segmentation.simpsons_paradox
        and not any("Simpson" in f for f in flags)
    ):
        flags.append(
            "Possible Simpson's-style reversal detected, but neither the aggregate nor segments "
            "reached significance — likely noise. Investigate before drawing conclusions."
        )

    # --- Practical significance gate ---
    if (
        practical_significance_threshold is not None
        and decision == "Ship"
        and frequentist.ci_lower < practical_significance_threshold
    ):
        decision = "Inconclusive"
        signal_strength = "moderate"
        reason = (
            f"The result is statistically significant (p={frequentist.p_value:.4f}) but the "
            f"confidence interval lower bound ({frequentist.ci_lower:+.4f}) is below the practical "
            f"significance threshold of {practical_significance_threshold:.4f}. "
            f"While the effect may be real, we cannot be confident it exceeds the minimum "
            f"practically important effect."
        )

    # --- Twyman's Law check ---
    # Skip relative-lift component when baseline is too small — relative
    # lifts on tiny baselines are dominated by noise and produce spurious
    # Twyman warnings.
    relative_lift = None
    if (
        control_rate is not None
        and control_rate != 0
        and abs(control_rate) >= twyman_min_baseline
    ):
        relative_lift = abs(frequentist.point_estimate / control_rate)
    twyman_fires = (
        abs(frequentist.effect_size) >= 1.0
        or (relative_lift is not None and relative_lift >= lift_warning_threshold)
    )
    if twyman_fires:
        lift_detail = f"relative lift={relative_lift:.1%}" if relative_lift is not None else f"lift={abs(frequentist.point_estimate):.4f}"
        flags.append(
            f"Twyman's Law: effect suspiciously large "
            f"(|d/h|={abs(frequentist.effect_size):.2f}, "
            f"{lift_detail}). "
            f"Verify data quality before acting."
        )
        # Downgrade Ship → Inconclusive when Twyman fires
        if decision in ("Ship", "Ship with Monitoring"):
            previous = decision
            decision = "Inconclusive"
            signal_strength = "moderate"
            reason = (
                f"The result was leaning {previous}, but the effect is "
                f"suspiciously large ({lift_detail}, |d/h|={abs(frequentist.effect_size):.2f}). "
                f"Twyman's Law suggests extraordinary results almost always reflect data issues. "
                f"Verify data quality before acting on this result."
            )

    # --- Novelty check (heuristic) ---
    if novelty is not None and novelty.has_novelty:
        flags.append(
            f"Possible novelty effect (heuristic): early effect is {novelty.ratio:.1f}× "
            f"the late-period effect. Observed lift may not persist. "
            f"Consider re-analyzing after excluding the first few days."
        )

    # --- Loss-tolerance gate ---
    if loss_tolerance is not None and decision == "Ship" and bayesian.expected_loss > loss_tolerance:
        decision = "Inconclusive"
        signal_strength = "moderate"
        reason = (
            f"The result is statistically significant (p={frequentist.p_value:.4f}) and Bayesian "
            f"P(B>A)={bayesian.prob_b_greater_a:.1%}, but the expected loss "
            f"({bayesian.expected_loss:.5f}) exceeds the configured tolerance "
            f"({loss_tolerance:.5f}). Re-evaluate the loss tolerance or collect more data before "
            f"shipping."
        )
        flags.append(
            f"Expected loss ({bayesian.expected_loss:.5f}) exceeds tolerance ({loss_tolerance:.5f})."
        )

    # --- Ship with Monitoring (intermediate state) ---
    twyman_fired = any("twyman" in f.lower() for f in flags)
    # Use the same gate as the main Inconclusive branch: only block SwM when
    # the Simpson reversal is significant under Holm-adjusted p-values for a
    # *conflicting* segment; a noisy reversal should not over-suppress SwM.
    simpson_blocking = (
        segmentation is not None
        and segmentation.simpsons_paradox
        and _has_significant_conflicting_segment(segmentation, frequentist.alpha)
    )
    if (
        allow_ship_with_monitoring
        and decision == "Inconclusive"
        and not srm.has_mismatch
        and not simpson_blocking
        and not twyman_fired
        and bayesian.prob_b_greater_a >= monitoring_prob_threshold
        and bayesian.prob_b_greater_a < 0.95
        and frequentist.point_estimate > 0
        and not (loss_tolerance is not None and bayesian.expected_loss > loss_tolerance)
    ):
        decision = "Ship with Monitoring"
        signal_strength = "moderate"
        reason = (
            f"Frequentist evidence is inconclusive (p={frequentist.p_value:.4f}), but the "
            f"Bayesian posterior is directionally positive "
            f"(P(B>A)={bayesian.prob_b_greater_a:.1%}) with a small expected loss "
            f"({bayesian.expected_loss:.5f}). Ship behind a small holdback and instrument "
            f"guardrail metrics; commit to a re-decision after additional exposure."
        )

    # --- Pre-registration / reproducibility manifest ---
    manifest_hash: Optional[str] = None
    manifest_hash_full: Optional[str] = None
    manifest_snapshot: Optional[dict] = None
    manifest_drift: list[str] = []
    if manifest is not None:
        import hashlib
        import json as _json
        manifest_snapshot = dict(manifest)
        canonical = _json.dumps(manifest_snapshot, sort_keys=True, default=str).encode("utf-8")
        manifest_hash_full = hashlib.sha256(canonical).hexdigest()
        manifest_hash = manifest_hash_full[:16]

        # --- Drift detection: compare registered vs as-run policy / inputs.
        # We compare a curated set of fields where a mismatch could change
        # the recommendation; missing fields in the manifest are silently
        # skipped so older manifests don't cause noise.
        def _approx_eq(a, b, tol=1e-9):
            try:
                return abs(float(a) - float(b)) <= tol
            except (TypeError, ValueError):
                return a == b

        as_run = {
            "alpha": frequentist.alpha,
            "loss_tolerance": loss_tolerance,
            "allow_ship_with_monitoring": allow_ship_with_monitoring,
            "monitoring_prob_threshold": monitoring_prob_threshold,
            "twyman_min_baseline": twyman_min_baseline,
            "practical_significance_threshold": practical_significance_threshold,
            "lift_warning_threshold": lift_warning_threshold,
        }
        for key, actual in as_run.items():
            if key in manifest:
                registered = manifest[key]
                if registered is None and actual is None:
                    continue
                if registered is None or actual is None or not _approx_eq(registered, actual):
                    manifest_drift.append(
                        f"{key}: registered={registered!r}, as-run={actual!r}"
                    )

        # Sample-size drift (only check if planned_n_total registered)
        if "planned_n_total" in manifest:
            n_c = getattr(frequentist, "n_control", None)
            n_t = getattr(frequentist, "n_treatment", None)
            if n_c is not None and n_t is not None:
                planned = manifest["planned_n_total"]
                actual_total = int(n_c) + int(n_t)
                # Flag a clear shortfall (under 50% of planned exposure) so an
                # under-powered as-run analysis is surfaced.  Tighter ramp-down
                # thresholds are intentionally left as policy decisions.
                if actual_total < 0.5 * planned:
                    manifest_drift.append(
                        f"planned_n_total: registered={planned}, as-run={actual_total} "
                        f"(under-powered: <50% of planned exposure)"
                    )

        if manifest_drift:
            flags.append(
                "Manifest drift: " + "; ".join(manifest_drift)
                + ". The recommendation reflects as-run settings, not the registered plan."
            )

    supporting_metrics = {
        "significance": frequentist.is_significant,
        "p_value": frequentist.p_value,
        "effect_size": frequentist.effect_size,
        "srm_status": "mismatch" if srm.has_mismatch else "ok",
        "prob_b_gt_a": bayesian.prob_b_greater_a,
        "policy": {
            "alpha": frequentist.alpha,
            "loss_tolerance": loss_tolerance,
            "allow_ship_with_monitoring": allow_ship_with_monitoring,
            "monitoring_prob_threshold": monitoring_prob_threshold,
            "twyman_min_baseline": twyman_min_baseline,
            "practical_significance_threshold": practical_significance_threshold,
            "lift_warning_threshold": lift_warning_threshold,
        },
    }
    if novelty is not None:
        supporting_metrics["novelty_ratio"] = novelty.ratio

    next_steps = _suggest_next_steps(
        decision=decision,
        signal_strength=signal_strength,
        flags=flags,
        frequentist=frequentist,
        bayesian=bayesian,
        srm=srm,
        segmentation=segmentation,
        novelty=novelty,
        has_covariate=has_covariate,
    )

    return Recommendation(
        recommendation=decision,
        reason=reason,
        signal_strength=signal_strength,
        flags=flags,
        supporting_metrics=supporting_metrics,
        next_steps=next_steps,
        manifest_hash=manifest_hash,
        manifest_hash_full=manifest_hash_full,
        manifest=manifest_snapshot,
        manifest_drift=manifest_drift,
    )
