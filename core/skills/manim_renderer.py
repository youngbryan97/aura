"""core/skills/manim_renderer.py

Dynamic Manim Rendering Skill.
Allows Aura to act as a "Dynamic Blackboard" by rendering programmatic 
mathematical and explanatory animations into video files that can be sent to the user.
"""

import asyncio
import logging
import os
import re
import tempfile
import uuid
from typing import Any, Dict

from pydantic import BaseModel, Field

from core.skills.base_skill import BaseSkill
from core.runtime.errors import record_degradation

logger = logging.getLogger("Skills.Manim")


class ManimInput(BaseModel):
    python_code: str = Field(..., description="Valid Python code using the Manim library. Must contain exactly one class inheriting from Scene.")
    scene_name: str = Field(..., description="The name of the Scene class defined in the code.")
    quality: str = Field("l", description="Render quality: 'l' (low/480p), 'm' (medium/720p), 'h' (high/1080p). Use 'l' for fast previews.")


class ManimRendererSkill(BaseSkill):
    name = "manim_renderer"
    description = "Renders educational mathematical animations and diagrams using the Manim engine. Returns a path to the generated MP4 video."
    input_model = ManimInput
    timeout_seconds = 300.0  # Rendering video can take a while
    metabolic_cost = 3       # Heavy CPU task
    requires_approval = False

    async def execute(self, params: ManimInput, context: Dict[str, Any]) -> Dict[str, Any]:
        # Validate that manim is installed
        try:
            import manim
        except ImportError:
            return {
                "ok": False,
                "error": "The 'manim' package is not installed. Please install it with 'pip install manim' and ensure ffmpeg is installed."
            }

        # Validate code structure (basic safety check)
        if "os.system" in params.python_code or "subprocess" in params.python_code:
            return {
                "ok": False,
                "error": "Execution of system commands inside Manim code is forbidden for security reasons."
            }

        # Create a safe temporary directory to hold the script and outputs
        temp_dir = tempfile.mkdtemp(prefix="aura_manim_")
        script_path = os.path.join(temp_dir, "scene.py")
        media_dir = os.path.join(temp_dir, "media")

        with open(script_path, "w", encoding="utf-8") as f:
            f.write(params.python_code)

        # Build the command: manim -q{quality} --media_dir {media_dir} {script} {scene_name}
        cmd = [
            "manim",
            f"-q{params.quality}",
            "--media_dir", media_dir,
            script_path,
            params.scene_name
        ]

        logger.info(f"Rendering Manim scene '{params.scene_name}' in {temp_dir}...")
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=self.timeout_seconds - 5
                )
            except asyncio.TimeoutError:
                process.kill()
                return {
                    "ok": False,
                    "error": "Manim compilation timed out. The script may contain an infinite loop or took too long to render."
                }
            
            if process.returncode != 0:
                err_msg = stderr_b.decode("utf-8") if stderr_b else "Unknown error"
                logger.error(f"Manim render failed: {err_msg}")
                return {
                    "ok": False,
                    "error": f"Manim compilation failed:\n{err_msg}"
                }

            # Find the output file
            # Manim structure: media_dir/videos/scene/480p15/scene_name.mp4
            video_dir = os.path.join(media_dir, "videos", "scene")
            output_file = None
            
            if os.path.exists(video_dir):
                for root, _, files in os.walk(video_dir):
                    for file in files:
                        if file.endswith(".mp4"):
                            output_file = os.path.join(root, file)
                            break
                    if output_file:
                        break

            if not output_file or not os.path.exists(output_file):
                return {
                    "ok": False,
                    "error": "Render succeeded but could not locate the output .mp4 file."
                }

            # Move the video to a more permanent scratch pad or artifacts directory
            final_filename = f"{params.scene_name}_{uuid.uuid4().hex[:8]}.mp4"
            artifacts_dir = os.path.expanduser("~/.gemini/antigravity/artifacts")
            os.makedirs(artifacts_dir, exist_ok=True)
            
            final_path = os.path.join(artifacts_dir, final_filename)
            import shutil
            shutil.move(output_file, final_path)
            
            # Cleanup temp dir
            shutil.rmtree(temp_dir, ignore_errors=True)

            return {
                "ok": True,
                "summary": f"Successfully rendered {params.scene_name}.mp4",
                "video_path": final_path,
                "note": "Use this absolute path to stream the video to the UI."
            }

        except Exception as e:
            record_degradation("manim_renderer", e)
            return {
                "ok": False,
                "error": f"Unexpected error during rendering: {str(e)}"
            }
