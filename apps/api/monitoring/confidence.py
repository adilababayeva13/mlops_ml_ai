def confidence_bucket(p: float) -> str:
    if p >= 0.9:
        return "0.9-1.0"
    if p >= 0.7:
        return "0.7-0.9"
    if p >= 0.5:
        return "0.5-0.7"
    return "<0.5"
