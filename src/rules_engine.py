def check_suspicious_patterns(input_data):
    """
    Rule-based fraud detection logic
    Returns list of triggered warning messages
    """

    warnings = []

    amount = input_data.get("Amount", 0)
    time = input_data.get("Time", 0)

    # Rule 1: Very High Amount
    if amount > 10000:
        warnings.append("⚠️ Unusually High Transaction Amount")

    # Rule 2: Night Time Transaction (Assuming time in seconds)
    # 0–21600 seconds = 12AM–6AM approx
    if time < 21600:
        warnings.append("⚠️ Transaction at Odd Hours")

    return warnings