"""Solve Member 3's BOM-based pricing game with configurable customer valuations.

The seller posts a menu. Customers choose their eligible option with the highest
positive surplus. All customer prices are integer cents, with BOMs stored to
four EUR decimals. Contributions in this model deduct only the priced BOM.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import json
from pathlib import Path


PRICE_TICKS = 100  # Number of BOM-cost ticks in one cent.
RATE_SCALE = 10_000  # One basis point is one rate unit.
PRODUCTS = ("RM6_NE_IQI_433520", "RMAirSeT_IIQ_24kV")
HORIZONS = (3, 6, 9)
ELIGIBILITY = ("both", "rm6_only", "airset_only")
SHORT_COSTS = {
    "p10": "empirical_p10_material_cost_eur_per_unit",
    "p50": "central_material_cost_eur_per_unit",
    "p90": "empirical_p90_material_cost_eur_per_unit",
}


@dataclass(frozen=True)
class Customer:
    rm6_wtp: int
    airset_premium: int
    units: int
    eligibility: str = "both"

    @property
    def airset_wtp(self):
        return self.rm6_wtp + self.airset_premium


@dataclass(frozen=True)
class Offer:
    rm6_price: int | None = None
    airset_price: int | None = None


@dataclass(frozen=True)
class Outcome:
    offer: Offer
    profit: int  # BOM-cost ticks, summed across units.
    rm6_units: int
    airset_units: int

    @property
    def rank(self):
        # Ties favor serving more customers, then lower total quoted prices.
        return (self.profit, self.rm6_units + self.airset_units,
                -sum(price or 0 for price in (self.offer.rm6_price, self.offer.airset_price)))


def ceil_ratio(numerator, denominator):
    return -(-numerator // denominator)


def minimum_price(cost, min_margin):
    return max(1, ceil_ratio(cost * RATE_SCALE, PRICE_TICKS * (RATE_SCALE - min_margin)))


def airset_band(rm6_price, costs, min_margin, max_gap):
    """Return the integer-cent price interval satisfying the two margin rules."""
    rm6, airset = costs
    denominator = rm6 * RATE_SCALE
    gap = max_gap * PRICE_TICKS * rm6_price
    numerator = airset * rm6_price * RATE_SCALE
    low = max(minimum_price(airset, min_margin), ceil_ratio(numerator, denominator + gap))
    high = numerator // (denominator - gap) if denominator > gap else None
    return low, high


def choice_at_rm6(customer, rm6_price):
    """Return RM6 units and the highest cent at which AirSeT wins.

    RM6 wins a positive-surplus product tie. The outside option wins at zero.
    """
    rm6_surplus = 0
    if rm6_price is not None and customer.eligibility != "airset_only":
        rm6_surplus = max(0, customer.rm6_wtp - rm6_price)
    airset_limit = None
    if customer.eligibility != "rm6_only":
        airset_limit = customer.airset_wtp - rm6_surplus - 1
    return customer.units if rm6_surplus else 0, airset_limit, customer.units


def evaluate(offer, costs, choices):
    rm6_units = airset_units = 0
    for fallback_units, airset_limit, units in choices:
        if offer.airset_price is not None and airset_limit is not None and offer.airset_price <= airset_limit:
            airset_units += units
        else:
            rm6_units += fallback_units
    rm6_profit = ((offer.rm6_price or 0) * PRICE_TICKS - costs[0]) * rm6_units
    airset_profit = ((offer.airset_price or 0) * PRICE_TICKS - costs[1]) * airset_units
    return Outcome(offer, rm6_profit + airset_profit, rm6_units, airset_units)


def airset_candidates(low, high, choices):
    """Evaluate the last cent of each interval with the same customer choices."""
    if low <= high:
        limits = {limit for _, limit, _ in choices if limit is not None and low <= limit < high}
        yield from sorted(limits)
        yield high


def optimize(costs, customers, min_margin, max_gap):
    """Globally optimize both active prices in cents and compare with single offers."""
    rm6_limits = [c.rm6_wtp - 1 for c in customers if c.eligibility != "airset_only"]
    airset_limits = [c.airset_wtp - 1 for c in customers if c.eligibility != "rm6_only"]
    rm6_floor, airset_floor = (minimum_price(cost, min_margin) for cost in costs)
    rm6_ceiling = max(rm6_limits, default=0)
    airset_ceiling = max(airset_limits, default=0)
    outside_choices = [choice_at_rm6(c, None) for c in customers]
    best = {"no_offer": evaluate(Offer(), costs, outside_choices),
            "rm6_only": None, "airset_only": None, "both": None}

    for price in sorted(set(rm6_limits)):
        if price >= rm6_floor:
            outcome = evaluate(Offer(price), costs, [choice_at_rm6(c, price) for c in customers])
            if best["rm6_only"] is None or outcome.rank > best["rm6_only"].rank:
                best["rm6_only"] = outcome
    for price in airset_candidates(airset_floor, airset_ceiling, outside_choices):
        outcome = evaluate(Offer(airset_price=price), costs, outside_choices)
        if best["airset_only"] is None or outcome.rank > best["airset_only"].rank:
            best["airset_only"] = outcome

    if airset_floor <= airset_ceiling:
        for price in range(rm6_floor, rm6_ceiling + 1):
            low, high = airset_band(price, costs, min_margin, max_gap)
            high = airset_ceiling if high is None else min(high, airset_ceiling)
            if high < low:
                continue
            choices = [choice_at_rm6(c, price) for c in customers]
            for airset_price in airset_candidates(low, high, choices):
                outcome = evaluate(Offer(price, airset_price), costs, choices)
                if best["both"] is None or outcome.rank > best["both"].rank:
                    best["both"] = outcome
    # Single-product menus cover any equivalent two-product menu with an
    # unmarketable price above its reservation ceiling. Prefer simpler menus on ties.
    name = max((name for name, result in best.items() if result is not None),
               key=lambda name: best[name].rank)
    return name, best


def anchored(costs, rm6_price, customer, min_margin, max_gap):
    """Optimize a menu for one buyer while retaining their fixed RM6 fallback."""
    choices = [choice_at_rm6(customer, rm6_price)]
    fallback = evaluate(Offer(rm6_price), costs, choices)
    low, high = airset_band(rm6_price, costs, min_margin, max_gap)
    _, buyer_upper, _ = choices[0]
    allowed = buyer_upper if high is None else min(buyer_upper, high)
    if allowed < low:
        return fallback
    alternative = evaluate(Offer(rm6_price, allowed), costs, choices)
    return alternative if alternative.rank > fallback.rank else fallback


def minimum_premium(costs, rm6_price, rm6_wtp, min_margin, max_gap):
    """Smallest WTP premium for AirSeT to win and strictly improve contribution."""
    low, high = airset_band(rm6_price, costs, min_margin, max_gap)
    fallback_profit = rm6_price * PRICE_TICKS - costs[0] if rm6_wtp > rm6_price else 0
    price = max(low, (costs[1] + fallback_profit) // PRICE_TICKS + 1)
    if high is not None and price > high:
        return None
    return price - min(rm6_wtp, rm6_price) + 1


def parse_number(raw, scale):
    try:
        value = Decimal(str(raw))
        if not value.is_finite():
            raise ValueError("Expected a finite number")
        return int((value * scale).to_integral_value(rounding=ROUND_HALF_UP))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"Invalid number {raw!r}") from error


def parse_customer(raw):
    parts = raw.split(":")
    if len(parts) not in (3, 4):
        raise ValueError("A customer needs B:W:UNITS[:both|rm6_only|airset_only]")
    customer = Customer(parse_number(parts[0], 100), parse_number(parts[1], 100),
                        int(parts[2]), parts[3] if len(parts) == 4 else "both")
    if min(customer.rm6_wtp, customer.airset_wtp) < 0 or customer.units <= 0 or customer.eligibility not in ELIGIBILITY:
        raise ValueError("Customer valuations must be nonnegative and units positive, with valid eligibility")
    return customer


def load_costs(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = {(row["product"], int(row["h"])): row for row in csv.DictReader(file)}
    results = {}
    for horizon in HORIZONS:
        paired = [rows[product, horizon] for product in PRODUCTS]
        results[horizon] = {}
        for case in SHORT_COSTS:
            column = SHORT_COSTS[case] if horizon == 3 else f"planning_{case}_eur_per_unit"
            costs = tuple(parse_number(row[column], 100 * PRICE_TICKS) for row in paired)
            if min(costs) <= 0:
                raise ValueError(f"Both priced BOM costs must be positive at {horizon} months")
            results[horizon][case] = costs
    return results


def describe(result, costs):
    if result is None:
        return None
    r, a = result.offer.rm6_price, result.offer.airset_price
    price_r, price_a = (value / 100 if value is not None else None for value in (r, a))
    margins = [100 * (1 - c / (value * PRICE_TICKS)) if value is not None else None
               for c, value in zip(costs, (r, a), strict=True)]
    return {"rm6_price": price_r, "airset_price": price_a,
            "rm6_units": result.rm6_units, "airset_units": result.airset_units,
            "total_bom_contribution": result.profit / (100 * PRICE_TICKS),
            "rm6_bom_margin_pct": margins[0], "airset_bom_margin_pct": margins[1],
            "bom_margin_gap_pp": abs(margins[0] - margins[1]) if r is not None and a is not None else None}


def run(cost_path, output, reference, anchor_wtp, premiums, customers, min_margin, max_gap):
    all_costs = load_costs(cost_path)
    if not 0 <= min_margin < RATE_SCALE or not 0 <= max_gap <= RATE_SCALE:
        raise ValueError("Margin floor must be in [0%, 100%) and gap in [0, 100] points")
    if anchor_wtp < 0 or anchor_wtp + min(premiums) < 0:
        raise ValueError("The reference scenario must have nonnegative valuations")
    if reference <= 0:
        raise ValueError("The RM6 reference must be positive")

    results = {"reference_rm6_price": reference / 100, "reference_rm6_wtp": anchor_wtp / 100,
               "min_bom_margin_pct": min_margin / 100, "max_bom_margin_gap_pp": max_gap / 100,
               "customers": [{"rm6_wtp": c.rm6_wtp / 100, "airset_premium": c.airset_premium / 100,
                              "units": c.units, "eligibility": c.eligibility} for c in customers],
               "cost_source": str(cost_path), "scenarios": []}
    sensitivity = []
    for horizon, cost_cases in all_costs.items():
        for case, costs in cost_cases.items():
            name, menus = optimize(costs, customers, min_margin, max_gap)
            reference_feasible = reference >= minimum_price(costs[0], min_margin)
            threshold = minimum_premium(costs, reference, anchor_wtp, min_margin, max_gap) if reference_feasible else None
            cutoff = None if threshold is None else threshold / 100
            results["scenarios"].append({
                "months": horizon, "cost_case": case,
                "cost_method": "bottom_up" if horizon == 3 else "flat",
                "rm6_cost": costs[0] / (100 * PRICE_TICKS),
                "airset_cost": costs[1] / (100 * PRICE_TICKS),
                "airset_cost_gap": (costs[1] - costs[0]) / (100 * PRICE_TICKS),
                "fixed_rm6_reference_feasible": reference_feasible,
                "fixed_rm6_switch_premium": cutoff, "optimal_menu": name,
                "menus": {label: describe(result, costs) for label, result in menus.items()},
            })
            for premium in premiums:
                result = (anchored(costs, reference, Customer(anchor_wtp, premium, 1), min_margin, max_gap)
                          if reference_feasible else evaluate(Offer(), costs, []))
                sensitivity.append({"months": horizon, "cost_case": case,
                                    "customer_airset_premium": premium / 100,
                                    "reference_feasible": reference_feasible, **describe(result, costs)})
    output.mkdir(parents=True, exist_ok=True)
    (output / "joint_pricing.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (output / "joint_pricing_sensitivity.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(sensitivity[0]))
        writer.writeheader()
        writer.writerows(sensitivity)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--costs", type=Path, default=Path("outputs/member3/product_costs.csv"))
    parser.add_argument("--out", type=Path, default=Path("outputs/member3"))
    parser.add_argument("--rm6-reference", default="12233", help="Illustrative RM6 offer in EUR")
    parser.add_argument("--rm6-wtp", default="12250", help="RM6 reservation price in the fixed-reference sensitivity")
    parser.add_argument("--premium-grid", nargs=3, default=("0", "250", "5"), metavar=("MIN", "MAX", "STEP"))
    parser.add_argument("--min-bom-margin-pct", default="85")
    parser.add_argument("--max-bom-margin-gap-pp", default="1")
    parser.add_argument("--segment", action="append", help="B:W:UNITS[:both|rm6_only|airset_only]; repeat to add buyers")
    args = parser.parse_args()
    try:
        start, stop, step = (parse_number(value, 100) for value in args.premium_grid)
        if step <= 0 or stop < start:
            raise ValueError("The WTP grid requires a positive step and MAX >= MIN")
        customers = [parse_customer(value) for value in (args.segment or ("12250:40:1", "12250:120:1"))]
        result = run(args.costs, args.out, parse_number(args.rm6_reference, 100),
                     parse_number(args.rm6_wtp, 100), range(start, stop + 1, step), customers,
                     parse_number(args.min_bom_margin_pct, 100), parse_number(args.max_bom_margin_gap_pp, 100))
    except (KeyError, OSError, ValueError) as error:
        parser.error(str(error))
    for case in result["scenarios"]:
        best = case["menus"][case["optimal_menu"]]
        print(f'{case["months"]}m {case["cost_case"]}: {case["optimal_menu"]}, '
              f'RM6 price={best["rm6_price"]}, AirSeT price={best["airset_price"]}, '
              f'BOM contribution EUR {best["total_bom_contribution"]:.2f}')


if __name__ == "__main__":
    main()
