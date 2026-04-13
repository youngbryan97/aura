import logging
import subprocess

logger = logging.getLogger("Senses.Notifications")


class DesktopNotifier:
    """Handles native OS desktop notifications (macOS focuses)."""

    @staticmethod
    def send(title: str, message: str, subtitle: str | None = None, sound: str = "Tink") -> None:
        """Send a native macOS desktop notification.
        
        Args:
            title: The bold title of the notification (e.g. "Aura")
            message: The body text
            subtitle: Optional subtitle
            sound: System sound to play (e.g. "Glass", "Basso", "Purr", "Tink")
        """
        try:
            # Escape strings to prevent shell injection via AppleScript
            safe_title = title.replace('"', '\\"')
            safe_msg = message.replace('"', '\\"')
            
            script = f'display notification "{safe_msg}" with title "{safe_title}"'
            
            if subtitle:
                safe_sub = subtitle.replace('"', '\\"')
                script += f' subtitle "{safe_sub}"'
            
            if sound:
                safe_sound = sound.replace('"', '\\"')
                script += f' sound name "{safe_sound}"'

            # Run AppleScript to trigger the native macOS toast
            subprocess.run(
                ["osascript", "-e", script],
                check=True,
                capture_output=True,
                timeout=5
            )
            logger.debug(f"Pushed macOS notification: {title} | {message}")
        except Exception as e:
            logger.error(f"Failed to send desktop notification: {e}")

    @staticmethod
    def push_insight(message: str) -> None:
        """Helper to push a standard Aura insight notification."""
        DesktopNotifier.send(
            title="Aura Insight",
            message=message,
            sound="Glass"
        )
