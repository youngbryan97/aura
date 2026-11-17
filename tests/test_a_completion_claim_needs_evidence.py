

def test_a_sent_notification_and_an_ive_claim_are_effects_to_evidence() -> None:
    """LIVE 16 November 2026, 21:41: "I've sent a desktop notification reminding you to stretch." on a turn where
    nothing was sent. "I've" was never read (no space before its "'ve"), and "sent" was taken for speech."""
    from core.epistemics.unevidenced_action import find_unevidenced_action_claims

    for claim in ("I've sent a desktop notification reminding you to stretch.",
                  "I’ve created the folder on your desktop.", "I sent a notification."):
        assert find_unevidenced_action_claims(claim, effects_observed=False, action_requested=True), claim
    for speech in ("I sent you my thoughts on that above.", "I've thought about it a lot."):
        assert not find_unevidenced_action_claims(speech, effects_observed=False, action_requested=True), speech
