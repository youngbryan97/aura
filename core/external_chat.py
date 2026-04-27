"""External Chat Window System
Aura can open chat windows to communicate even when running in background.

CRITICAL CAPABILITIES:
- Open terminal or GUI chat windows
- Aura initiates conversation (not just responds)
- All conversations retained by core model
- Full request execution through external windows
- Multiple simultaneous windows

This allows Aura to "tap you on the shoulder" when she wants to talk.
"""
from core.utils.task_tracker import get_task_tracker
import asyncio
import json
import logging
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger("ExternalChat")


@dataclass
class ChatMessage:
    """A message in external chat"""

    speaker: str  # "aura" or "user"
    text: str
    timestamp: float
    window_id: str
    
    def to_dict(self):
        return {
            "speaker": self.speaker,
            "text": self.text,
            "timestamp": self.timestamp,
            "window_id": self.window_id
        }


class TerminalChatWindow:
    """Terminal-based chat window.
    
    Fast, simple, works on any system.
    Opens in a new terminal window.
    """
    
    def __init__(self, window_id: str, orchestrator):
        self.window_id = window_id
        self.orchestrator = orchestrator
        
        # Communication queues
        self.incoming_queue = queue.Queue()  # User → Aura
        self.outgoing_queue = queue.Queue()  # Aura → User
        
        # State
        self.active = False
        self.process = None
        self.handler_task = None
        
        logger.info("✓ Terminal Chat Window created: %s", window_id)
    
    def open(self, initial_message: Optional[str] = None):
        """Open terminal chat window.
        
        Args:
            initial_message: What Aura says when opening the window

        """
        logger.info("📟 Opening terminal chat window: %s", self.window_id)
        
        # Create chat script
        script_path = self._create_chat_script(initial_message)
        
        # Open in new terminal
        import platform
        system = platform.system()
        
        try:
            if system == "Linux":
                # Try various terminal emulators
                terminals = [
                    ["gnome-terminal", "--", "bash", script_path],
                    ["xterm", "-e", f"bash {script_path}"],
                    ["konsole", "-e", f"bash {script_path}"],
                    ["xfce4-terminal", "-e", f"bash {script_path}"]
                ]
                
                for term_cmd in terminals:
                    try:
                        self.process = subprocess.Popen(term_cmd)
                        break
                    except FileNotFoundError:
                        continue
            
            elif system == "Darwin":  # macOS
                # Use osascript to open Terminal.app
                apple_script = f'''
tell application "Terminal"
    do script "bash {script_path}"
    activate
end tell
'''
                subprocess.Popen(["osascript", "-e", apple_script])
            
            elif system == "Windows":
                # Use cmd.exe
                subprocess.Popen(["cmd", "/c", "start", "cmd", "/k", f"bash {script_path}"])
            
            self.active = True
            
            # Start message handler
            self._start_message_handler()
            
            logger.info("✅ Terminal window opened: %s", self.window_id)
            
        except Exception as e:
            logger.error("Failed to open terminal: %s", e)
    
    def _create_chat_script(self, initial_message: Optional[str]) -> str:
        """Create bash script for chat interface"""
        script = f'''#!/bin/bash

# Chat window for Aura
WINDOW_ID="{self.window_id}"
PIPE_IN="/tmp/aura_chat_{self.window_id}_in"
PIPE_OUT="/tmp/aura_chat_{self.window_id}_out"

# Create named pipes
mkfifo $PIPE_IN 2>/dev/null
mkfifo $PIPE_OUT 2>/dev/null

# Display header
clear
echo "=================================="
echo "    AURA CHAT - {self.window_id}"
echo "=================================="
echo ""

# Initial message from Aura
if [ -n "{initial_message}" ]; then
    echo "AURA: {initial_message}"
    echo ""
fi

# Chat loop
while true; do
    # Check for Aura's messages
    if [ -p $PIPE_OUT ]; then
        if read -t 0.1 line < $PIPE_OUT; then
            echo "AURA: $line"
        fi
    fi
    
    # Prompt for user input
    read -t 0.1 -p "YOU: " user_input
    if [ $? -eq 0 ]; then
        # Send to Aura
        if [ -n "$user_input" ]; then
            echo "$user_input" > $PIPE_IN
        fi
        
        # Check for exit
        if [ "$user_input" = "exit" ] || [ "$user_input" = "quit" ]; then
            echo "Closing chat window..."
            break
        fi
    fi
done

# Cleanup
rm -f $PIPE_IN $PIPE_OUT
'''
        
        # Write script
        script_path = f"/tmp/aura_chat_{self.window_id}.sh"
        with open(script_path, 'w') as f:
            f.write(script)
        
        os.chmod(script_path, 0o755)
        
        return script_path
    
    def _start_message_handler(self):
        """Start background task to handle messages."""
        try:
            loop = asyncio.get_running_loop()
            self.handler_task = loop.create_task(self._message_handler_loop())
        except RuntimeError:
            # No running loop — use ensure_future to schedule for when one starts
            self.handler_task = get_task_tracker().track(self._message_handler_loop())
    
    async def _message_handler_loop(self):
        """Handle bidirectional communication"""
        pipe_in = f"/tmp/aura_chat_{self.window_id}_in"
        pipe_out = f"/tmp/aura_chat_{self.window_id}_out"
        
        while self.active:
            try:
                # Read from user (non-blocking)
                # ... (omitting tricky pipe logic for now as per original code)
                
                # Write Aura's messages
                if not self.outgoing_queue.empty():
                    aura_msg = self.outgoing_queue.get()
                    if os.path.exists(pipe_out):
                        # This opens the pipe, writes, and closes
                        with open(pipe_out, 'w') as f:
                            f.write(aura_msg + '\n')
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error("Message handler error: %s", e)
                await asyncio.sleep(1)
    
    def _process_user_message(self, message: str):
        """Process user message through orchestrator.
        
        This is CRITICAL - allows full request execution through external window.
        """
        try:
            # Add to orchestrator's message queue
            # Add to orchestrator's message queue via threadsafe method
            if hasattr(self.orchestrator, 'enqueue_from_thread'):
                self.orchestrator.enqueue_from_thread(message, origin=f"external_window_{self.window_id}")
            
            # Store in orchestrator's conversation history
            if hasattr(self.orchestrator, 'conversation_history'):
                self.orchestrator.conversation_history.append({
                    "timestamp": time.time(),
                    "source": f"external_window_{self.window_id}",
                    "speaker": "user",
                    "message": message
                })
            
        except Exception as e:
            logger.error("Failed to process message through orchestrator: %s", e)
    
    def send_message(self, text: str):
        """Send message from Aura to user in this window.
        
        Args:
            text: What Aura wants to say

        """
        self.outgoing_queue.put(text)
        
        # Store in conversation history
        msg = ChatMessage(
            speaker="aura",
            text=text,
            timestamp=time.time(),
            window_id=self.window_id
        )
        
        # Add to orchestrator's history
        if hasattr(self.orchestrator, 'conversation_history'):
            self.orchestrator.conversation_history.append(msg.to_dict())
    
    def close(self):
        """Close the chat window"""
        self.active = False
        
        # Clean up pipes
        for pipe in [f"/tmp/aura_chat_{self.window_id}_in", 
                     f"/tmp/aura_chat_{self.window_id}_out"]:
            if os.path.exists(pipe):
                try:
                    os.remove(pipe)
                except Exception as exc:
                    logger.debug("Suppressed: %s", exc)        
        logger.info("✅ Terminal window closed: %s", self.window_id)


class GUIChatWindow:
    """Simple GUI chat window using tkinter.
    
    Better UX than terminal, still simple and fast.
    """
    
    def __init__(self, window_id: str, orchestrator):
        self.window_id = window_id
        self.orchestrator = orchestrator
        
        # Communication
        self.incoming_queue = queue.Queue()
        self.outgoing_queue = queue.Queue()
        
        # State
        self.active = False
        self.window = None
        
        logger.info("✓ GUI Chat Window created: %s", window_id)
    
    def open(self, initial_message: Optional[str] = None):
        """Open GUI chat window"""
        logger.info("🪟 Opening GUI chat window: %s", self.window_id)
        
        # Start GUI in separate thread
        thread = threading.Thread(
            target=self._create_gui,
            args=(initial_message,),
            daemon=True
        )
        thread.start()
        
        self.active = True
    
    def _create_gui(self, initial_message: Optional[str]):
        """Create tkinter GUI"""
        try:
            import tkinter as tk
            from tkinter import scrolledtext
            
            # Create window
            root = tk.Tk()
            root.title(f"Aura Chat - {self.window_id}")
            root.geometry("500x600")
            
            # Chat display
            chat_display = scrolledtext.ScrolledText(
                root,
                wrap=tk.WORD,
                width=60,
                height=30,
                font=("Arial", 10)
            )
            chat_display.pack(padx=10, pady=10)
            chat_display.config(state=tk.DISABLED)
            
            # Show initial message
            if initial_message:
                chat_display.config(state=tk.NORMAL)
                chat_display.insert(tk.END, f"AURA: {initial_message}\n\n")
                chat_display.config(state=tk.DISABLED)
            
            # Input field
            input_frame = tk.Frame(root)
            input_frame.pack(padx=10, pady=5, fill=tk.X)
            
            input_field = tk.Entry(input_frame, font=("Arial", 10))
            input_field.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            def send_message():
                """Send user message"""
                text = input_field.get().strip()
                if text:
                    # Display in chat
                    chat_display.config(state=tk.NORMAL)
                    chat_display.insert(tk.END, f"YOU: {text}\n")
                    chat_display.config(state=tk.DISABLED)
                    chat_display.see(tk.END)
                    
                    # Clear input
                    input_field.delete(0, tk.END)
                    
                    # Process message
                    msg = ChatMessage(
                        speaker="user",
                        text=text,
                        timestamp=time.time(),
                        window_id=self.window_id
                    )
                    
                    self.incoming_queue.put(msg)
                    self._process_user_message(text)
            
            # Send button
            send_btn = tk.Button(
                input_frame,
                text="Send",
                command=send_message,
                font=("Arial", 10)
            )
            send_btn.pack(side=tk.RIGHT, padx=5)
            
            # Bind Enter key
            input_field.bind('<Return>', lambda e: send_message())
            
            # Check for Aura's messages
            def check_outgoing():
                """Check for messages from Aura"""
                try:
                    drained = 0
                    while drained < 32:
                        aura_text = self.outgoing_queue.get_nowait()
                        drained += 1
                        
                        chat_display.config(state=tk.NORMAL)
                        chat_display.insert(tk.END, f"AURA: {aura_text}\n\n")
                        chat_display.config(state=tk.DISABLED)
                        chat_display.see(tk.END)
                except queue.Empty:
                    pass
                except Exception as e:
                    logger.debug("GUI outgoing pump failed: %s", e, exc_info=True)
                
                # Schedule next check
                if self.active:
                    root.after(25 if not self.outgoing_queue.empty() else 100, check_outgoing)
            
            # Start checking for messages
            root.after(100, check_outgoing)
            
            # Store window reference
            self.window = root
            
            # Run GUI
            root.mainloop()
            
            # Cleanup when closed
            self.active = False
            
        except Exception as e:
            logger.error("GUI creation failed: %s", e)
            logger.info("Falling back to terminal window")
            # Could fallback to terminal window here
    
    def _process_user_message(self, message: str):
        """Process user message through orchestrator"""
        try:
            # Same as terminal window - full integration
            # Add to orchestrator's message queue via threadsafe method
            if hasattr(self.orchestrator, 'enqueue_from_thread'):
                self.orchestrator.enqueue_from_thread(message, origin=f"external_window_{self.window_id}")
            
            if hasattr(self.orchestrator, 'conversation_history'):
                self.orchestrator.conversation_history.append({
                    "timestamp": time.time(),
                    "source": f"external_window_{self.window_id}",
                    "speaker": "user",
                    "message": message
                })
        
        except Exception as e:
            logger.error("Failed to process message: %s", e)
    
    def send_message(self, text: str):
        """Send message from Aura to user"""
        self.outgoing_queue.put(text)
        
        # Store in history
        if hasattr(self.orchestrator, 'conversation_history'):
            self.orchestrator.conversation_history.append({
                "timestamp": time.time(),
                "source": f"external_window_{self.window_id}",
                "speaker": "aura",
                "message": text
            })
    
    def close(self):
        """Close GUI window"""
        self.active = False
        if self.window:
            try:
                # Need to run in main thread if possible, or trigger event
                # Tkinter isn't thread safe for destroy from other threads usually
                # But for now basic implementation
                pass 
            except Exception as exc:
                logger.debug("Suppressed: %s", exc)

class ExternalChatManager:
    """Manages all external chat windows.
    
    Aura uses this to initiate conversations with the user.
    """
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        
        # Track windows
        self.windows: Dict[str, Any] = {}
        self.next_window_id = 1
        
        # Preferences
        self.preferred_window_type = "gui"  # "gui" or "terminal"
        
        logger.info("✓ External Chat Manager initialized")
    
    def open_chat_window(self, message: Optional[str] = None, 
                        window_type: Optional[str] = None) -> str:
        """Open a new chat window.
        
        Args:
            message: Initial message from Aura
            window_type: "gui" or "terminal"
            
        Returns:
            Window ID

        """
        window_id = f"chat_{self.next_window_id}"
        self.next_window_id += 1
        
        window_type = window_type or self.preferred_window_type
        
        logger.info("🪟 Opening %s chat window: %s", window_type, window_id)
        
        # Create window
        if window_type == "gui":
            window = GUIChatWindow(window_id, self.orchestrator)
        else:
            window = TerminalChatWindow(window_id, self.orchestrator)
        
        # Open it
        window.open(message)
        
        # Store reference
        self.windows[window_id] = window
        
        return window_id
    
    def send_to_window(self, window_id: str, message: str):
        """Send message to specific window"""
        if window_id in self.windows:
            self.windows[window_id].send_message(message)
        else:
            logger.error("Window %s not found", window_id)
    
    def broadcast(self, message: str):
        """Send message to all open windows"""
        for window in self.windows.values():
            if window.active:
                window.send_message(message)
    
    def close_window(self, window_id: str):
        """Close specific window"""
        if window_id in self.windows:
            self.windows[window_id].close()
            del self.windows[window_id]
    
    def close_all_windows(self):
        """Close all external windows"""
        for window_id in list(self.windows.keys()):
            self.close_window(window_id)
    
    def get_active_windows(self) -> list:
        """Get list of active window IDs"""
        return [wid for wid, w in self.windows.items() if w.active]


def integrate_external_chat(orchestrator):
    """Integrate external chat capability into orchestrator.
    
    After this, Aura can:
    - Open chat windows from background
    - Initiate conversations with user
    - Process requests through external windows
    """
    # Initialize chat manager
    orchestrator.external_chat = ExternalChatManager(orchestrator)
    
    # Add conversation history if not present
    if not hasattr(orchestrator, 'conversation_history'):
        orchestrator.conversation_history = []
    
    # Hook response delivery to also send to external windows
    # Check if there's a method to hook into
    # orchestrator usually just prints or returns.
    # We might need to monkey patch or rely on orchestrator calling this explicitly.
    
    logger.info("✅ External chat integrated")
    logger.info("   Aura can now open chat windows and initiate conversations")
