"""Exact joint quote search for the P50/P90 policy on slides 6–10."""

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

import numpy as np


PRICE_TICKS = 100  # BOM-cost ticks in a cent
RATE_SCALE = 10_000  # Basis points in a unit rate
PRODUCTS = ("RM6", "RM AirSeT")

# Slide 8, using product_costs.csv before display rounding. Each row is RM6, AirSeT.
COSTS = {
    3: ((9_243_293, 10_036_894), (10_499_377, 11_552_851)),
    6: ((9_203_837, 9_987_176), (11_488_654, 13_062_582)),
    9: ((9_203_837, 9_987_176), (12_849_397, 14_274_559)),
}


@dataclass(frozen=True)
class Buyer:
    name: str
    base: int
    premium: int
    units: int = 1
    eligibility: str = "Both products"

    @property
    def airset_value(self):
        return self.base + self.premium

    @property
    def can_buy_rm6(self):
        return self.eligibility != "RM AirSeT only"

    @property
    def can_buy_airset(self):
        return self.eligibility != "RM6 only"


@dataclass(frozen=True)
class Offer:
    rm6: int | None = None
    airset: int | None = None

    @property
    def prices(self):
        return self.rm6, self.airset

    @property
    def menu(self):
        if self.rm6 is not None and self.airset is not None:
            return "Both products"
        if self.rm6 is not None:
            return "RM6 only"
        if self.airset is not None:
            return "RM AirSeT only"
        return "No sale"


@dataclass(frozen=True)
class Result:
    offer: Offer
    choices: tuple
    units: tuple
    revenue: int
    contributions: tuple


def cents(value):
    return int((Decimal(str(value)) * 100).to_integral_value(rounding=ROUND_HALF_UP))


def surplus(buyer, offer):
    return (
        buyer.base - offer.rm6 if buyer.can_buy_rm6 and offer.rm6 is not None else None,
        buyer.airset_value - offer.airset
        if buyer.can_buy_airset and offer.airset is not None else None,
    )


def purchase(buyer, offer):
    rm6, airset = surplus(buyer, offer)
    fallback = max(0, rm6 or 0)
    if airset is not None and airset > fallback:
        return "RM AirSeT"
    return "RM6" if fallback > 0 else "No buy"


def evaluate(offer, buyers, states):
    choices = tuple(purchase(buyer, offer) for buyer in buyers)
    units = tuple(sum(buyer.units for buyer, choice in zip(buyers, choices) if choice == product)
                  for product in PRODUCTS)
    revenue = sum((price or 0) * count for price, count in zip(offer.prices, units))
    contributions = tuple(revenue * PRICE_TICKS - sum(cost * count for cost, count in zip(state, units))
                          for state in states)
    return Result(offer, choices, units, revenue, contributions)


def minimum_price(cost, rate):
    return max(1, -(-cost * RATE_SCALE // (PRICE_TICKS * (RATE_SCALE - rate))))


def airset_band(rm6, states, rate, gap):
    """Intersect the AirSeT quote bands across P50/P90 for a fixed RM6 quote."""
    floor = max(minimum_price(costs[1], rate) for costs in states)
    lower = np.full_like(rm6, floor) if isinstance(rm6, np.ndarray) else floor
    ceiling = np.iinfo(np.int64).max
    upper = ceiling
    for cost_r, cost_a in states:
        numerator = cost_a * rm6 * RATE_SCALE
        midpoint = cost_r * RATE_SCALE
        width = gap * PRICE_TICKS * rm6
        lower = np.maximum(lower, -(-numerator // (midpoint + width)))
        denominator = midpoint - width
        upper = np.minimum(upper, np.where(denominator > 0,
                                          numerator // np.maximum(denominator, 1), ceiling))
    if isinstance(rm6, np.ndarray):
        return lower, upper
    return int(lower.item()), None if upper.item() == ceiling else int(upper.item())


def constraints(offer, states, rate, gap):
    """Return individual margin-floor and two-product spread checks per state."""
    checks = []
    for state in states:
        rate_ok = [price >= minimum_price(cost, rate) if price is not None else None
                   for cost, price in zip(state, offer.prices)]
        spread_ok = None
        if offer.rm6 is not None and offer.airset is not None:
            spread_ok = (abs(state[0] * offer.airset - state[1] * offer.rm6) * RATE_SCALE
                         <= gap * offer.rm6 * offer.airset * PRICE_TICKS)
        checks.append((*rate_ok, spread_ok))
    return tuple(checks)


def recommend(buyers, states, rate, gap):
    """Maximise worst-case contribution under the same constraints in both states.

    P90 costs dominate P50 product by product. For a fixed RM6 price, demand
    changes only at each buyer's highest accepted AirSeT cent. Inside each
    segment contribution grows with the AirSeT quote, so check its upper end.
    Search every possible RM6 cent in bounded batches for responsive reruns.
    """
    if any(low > high for low, high in zip(*states)):
        raise ValueError("P90 costs must be at least P50 costs for both products.")
    best = evaluate(Offer(), buyers, states)

    def rank(result):
        return (result.contributions[1], sum(result.units),
                -sum(price or 0 for price in result.offer.prices))

    def consider(offer):
        nonlocal best
        candidate = evaluate(offer, buyers, states)
        if rank(candidate) > rank(best):
            best = candidate

    rm6_targets = {buyer.base - 1 for buyer in buyers if buyer.can_buy_rm6}
    airset_targets = {buyer.airset_value - 1 for buyer in buyers if buyer.can_buy_airset}
    rm6_floor, airset_floor = (max(minimum_price(state[i], rate) for state in states) for i in range(2))
    rm6_ceiling = max(rm6_targets, default=0)
    airset_ceiling = max(airset_targets, default=0)

    for rm6 in sorted(rm6_targets):
        if rm6 >= rm6_floor:
            consider(Offer(rm6=rm6))
    for airset in sorted(airset_targets):
        if airset >= airset_floor:
            consider(Offer(airset=airset))
    if airset_ceiling < airset_floor:
        return best

    stress_r, stress_a = states[1]
    batch = 100_000
    for start in range(rm6_floor, rm6_ceiling + 1, batch):
        rm6 = np.arange(start, min(start + batch, rm6_ceiling + 1), dtype=np.int64)
        low, high = airset_band(rm6, states, rate, gap)
        high = np.minimum(high, airset_ceiling)
        feasible = low <= high
        if not np.any(feasible):
            continue
        rm6, low, high = rm6[feasible], low[feasible], high[feasible]
        fallback = [np.maximum(0, buyer.base - rm6) if buyer.can_buy_rm6 else np.zeros_like(rm6)
                    for buyer in buyers]
        limits = [buyer.airset_value - value - 1 if buyer.can_buy_airset else np.full_like(rm6, -1)
                  for buyer, value in zip(buyers, fallback)]

        for airset in (high, *(np.minimum(limit, high) for limit in limits)):
            sells_airset = [airset <= limit for limit in limits]
            units_a = sum(buyer.units * sold for buyer, sold in zip(buyers, sells_airset))
            units_r = sum(buyer.units * ((value > 0) & ~sold)
                          for buyer, value, sold in zip(buyers, fallback, sells_airset))
            objective = (rm6 * PRICE_TICKS - stress_r) * units_r + (airset * PRICE_TICKS - stress_a) * units_a
            objective = np.where((airset >= low) & (units_r > 0) & (units_a > 0), objective, -1)
            highest = objective.max()
            if highest < best.contributions[1]:
                continue
            tied = np.flatnonzero(objective == highest)
            volume = units_r[tied] + units_a[tied]
            tied = tied[volume == volume.max()]
            winner = tied[np.argmin(rm6[tied] + airset[tied])]
            consider(Offer(int(rm6[winner]), int(airset[winner])))
    return best


def switch_prices(buyer, rm6, states, rate, gap):
    """Return seller floor, seller ceiling, and buyer ceiling for an AirSeT switch."""
    lower, upper = airset_band(rm6, states, rate, gap)
    buys_rm6 = buyer.can_buy_rm6 and buyer.base > rm6
    # At the anchor, require a strict contribution gain in each state.
    recovery = max((cost_a + (rm6 * PRICE_TICKS - cost_r if buys_rm6 else 0)) // PRICE_TICKS + 1
                   for cost_r, cost_a in states)
    fallback_surplus = buyer.base - rm6 if buys_rm6 else 0
    buyer_ceiling = buyer.airset_value - fallback_surplus - 1 if buyer.can_buy_airset else None
    return max(lower, recovery), upper, buyer_ceiling, lower, recovery
