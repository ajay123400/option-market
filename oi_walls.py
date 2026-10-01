"""oi_walls.py -- support / resistance from the option chain, the way a
desk reads it, instead of "the strike with the most OI anywhere".

The naive max-OI strike is usually a far round number (24000 CE with the
market at 23000): cheap to write, lots of contracts, but 1000 points away
with no premium -- it says nothing about today. Here:

  1. Only strikes that can matter: resistance from CALLS ABOVE spot,
     support from PUTS BELOW spot, and only within the expected move --
     window = 2 x ATM straddle, clamped to 300..800 points (it narrows by
     itself into expiry as the straddle shrinks).
  2. Distance-aware OI: weight = OI x (1 - 0.4 x distance / window). A strike
     at the market counts fully, one at the window's edge 60% -- so a clearly
     bigger wall a little further away still wins, but between two similar
     walls the nearer one (the one price meets first) does. (Pure delta-
     weighting was tried first and over-favoured the very next strike, since
     near-the-money deltas are ~0.5 regardless of how much OI sits there.)
  3. The top two walls per side are returned with their OI and distance, so
     "how strong" and "what's next" are visible, not just one number.
"""
MIN_WINDOW, MAX_WINDOW = 300, 800


def walls(ce_oi, pe_oi, spot, fwd=None, T=None, iv_pct=None, straddle=None):
    """ce_oi / pe_oi: {strike: open interest}. Returns
    {"resistance", "support", "window", "resistance_detail", "support_detail"}
    where each detail lists the top-2 walls with OI, distance and weight.
    Falls back to plain OI inside the window if IV/T aren't available."""
    if not spot:
        return {"resistance": None, "support": None, "window": None}
    window = max(MIN_WINDOW, min(MAX_WINDOW, 2 * straddle)) if straddle else 500

    def weight(k, oi, is_call):
        return oi * (1 - 0.4 * abs(k - spot) / window)

    def pick(oi_map, is_call):
        cands = [(k, oi) for k, oi in oi_map.items()
                 if oi and ((spot < k <= spot + window) if is_call else (spot - window <= k < spot))]
        scored = sorted(((weight(k, oi, is_call), k, oi) for k, oi in cands), reverse=True)
        top = [{"strike": k, "oi": int(oi), "distance": round(abs(k - spot)), "weight": int(w)}
               for w, k, oi in scored[:2]]
        return (top[0]["strike"] if top else None), top

    res, res_top = pick(ce_oi, True)
    sup, sup_top = pick(pe_oi, False)
    return {"resistance": res, "support": sup, "window": round(window),
            "method": "OI walls within the expected move (distance-weighted)",
            "resistance_detail": res_top, "support_detail": sup_top}
