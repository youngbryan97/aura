import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime

from core.runtime.errors import record_degradation
from core.runtime.runtime_settings import get_runtime_setting
from core.runtime.subprocess_gateway import get_subprocess_gateway

logger = logging.getLogger("Senses.Notifications")


@dataclass(frozen=True)
class DeliveryResult:
    """Honest outcome of a notification attempt.

    ``delivered`` is True only when the OS accepted the notification.
    ``status`` is one of: delivered, disabled, suppressed_quiet_hours, failed.
    """

    delivered: bool
    status: str
    detail: str = ""

    def __bool__(self) -> bool:  # truthiness == was it actually delivered
        return self.delivered


def _parse_hhmm(value: object) -> tuple[int, int] | None:
    try:
        hh, mm = str(value).strip().split(":")
        h, m = int(hh), int(mm)
    # not a failure: a reading that cannot be taken, or that is not a
    # number when it is, leaves this at the value below.
    except (ValueError, AttributeError):
        return None
    if 0 <= h < 24 and 0 <= m < 60:
        return h, m
    return None


def _within_quiet_hours(now: datetime, start: object, end: object) -> bool:
    """True if ``now`` falls in the [start, end) quiet window (wraps midnight)."""
    s = _parse_hhmm(start)
    e = _parse_hhmm(end)
    if not s or not e:
        return False
    cur = now.hour * 60 + now.minute
    smin = s[0] * 60 + s[1]
    emin = e[0] * 60 + e[1]
    if smin == emin:
        return False  # zero-length window = quiet hours effectively off
    if smin < emin:
        return smin <= cur < emin  # same-day window
    return cur >= smin or cur < emin  # window wraps past midnight


def _notifications_allowed(now: datetime | None = None) -> bool:
    """Honor the user's notify.enabled + quiet-hours settings (default on / 22:00-08:00).

    Reads the persisted runtime settings the UI writes; defaults to allowed if
    unset/unreadable. See docs/SETTINGS_WIRING_AUDIT.md.
    """
    if not bool(get_runtime_setting("notify.enabled", True)):
        return False
    now = now or datetime.now()
    if _within_quiet_hours(
        now,
        get_runtime_setting("notify.quiet_hours_start", "22:00"),
        get_runtime_setting("notify.quiet_hours_end", "08:00"),
    ):
        return False
    return True


class DesktopNotifier:
    """Handles native OS desktop notifications (macOS focuses)."""

    @staticmethod
    def send(title: str, message: str, subtitle: str | None = None, sound: str = "Tink", *,
             asked_for: bool = False) -> DeliveryResult:
        """Send a native macOS desktop notification.

        Args:
            title: The bold title of the notification (e.g. "Aura")
            message: The body text
            subtitle: Optional subtitle
            sound: System sound to play (e.g. "Glass", "Basso", "Purr", "Tink")

        Returns:
            DeliveryResult stating whether the user was actually reached.
            Callers must not assume delivery — quiet hours, the notify.enabled
            setting, or an osascript failure all yield delivered=False.

        ``asked_for``: the person waiting on this turn asked for this notification. Quiet hours keep her from
        interrupting on her own; they do not refuse what the person has just asked for. LIVE 16 November 2026, 22:16:
        "Could you put up a desktop notification that says time to stretch?" was suppressed by the 21:00 to 08:30
        window, and she said so. Turning notifications off (notify.enabled) still refuses it.
        """
        if not bool(get_runtime_setting("notify.enabled", True)):
            logger.debug("🔕 Notification suppressed: notifications disabled in settings: %s", title)
            return DeliveryResult(
                delivered=False,
                status="disabled",
                detail="Notifications are disabled in user settings (notify.enabled=false).",
            )
        if not asked_for and _within_quiet_hours(
            datetime.now(),
            get_runtime_setting("notify.quiet_hours_start", "22:00"),
            get_runtime_setting("notify.quiet_hours_end", "08:00"),
        ):
            logger.debug("🔕 Notification suppressed by quiet hours: %s", title)
            return DeliveryResult(
                delivered=False,
                status="suppressed_quiet_hours",
                detail="Within the user's configured quiet hours window.",
            )
        words = [str(message or "").strip(), str(title or "").strip() or "Aura"]
        if not words[0]:
            return DeliveryResult(delivered=False, status="failed", detail="A notification with no words in it.")
        try:
            # The words go to AppleScript as arguments, never inside its source. Pasted into a string literal with
            # only its quotation marks escaped, a message holding \" closed the literal, and what followed it ran
            # as AppleScript. Only what the script reads is passed: the gateway refuses an empty argument (LIVE 16
            # November 2026, 23:31, "argv entries must not be empty" on a call with no subtitle, four times).
            clause = "display notification (item 1 of argv) with title (item 2 of argv)"
            for name, value in (("subtitle", subtitle), ("sound name", sound)):
                if value:
                    words.append(str(value))
                    clause += f" {name} (item {len(words)} of argv)"
            script = ["-e", "on run argv", "-e", clause, "-e", "end run"]
            result = get_subprocess_gateway().run(
                ["osascript", *script, *words],
                capture_output=True,
                timeout=5,
                source="senses.notifications.desktop_notification",
                accelerator_capability="none",
            )
            if result.returncode != 0:
                raise subprocess.CalledProcessError(
                    result.returncode,
                    result.args,
                    result.stdout,
                    result.stderr,
                )
            logger.debug("Pushed macOS notification: %s | %s", title, message)
            return DeliveryResult(delivered=True, status="delivered")
        except (subprocess.SubprocessError, OSError) as e:
            record_degradation('notifications', e)
            logger.error("Failed to send desktop notification: %s", e)
            return DeliveryResult(delivered=False, status="failed", detail=str(e)[:300])

    @staticmethod
    def push_insight(message: str) -> DeliveryResult:
        """Helper to push a standard Aura insight notification."""
        return DesktopNotifier.send(
            title="Aura Insight",
            message=message,
            sound="Glass"
        )
