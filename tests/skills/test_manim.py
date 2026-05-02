import asyncio
from core.skills.manim_renderer import ManimRendererSkill, ManimInput

async def test():
    skill = ManimRendererSkill()
    params = ManimInput(
        python_code="""
from manim import *
class SquareToCircle(Scene):
    def construct(self):
        circle = Circle()
        square = Square()
        square.flip(RIGHT)
        square.rotate(-3 * TAU / 8)
        circle.set_fill(PINK, opacity=0.5)
        self.play(Create(square))
        self.play(Transform(square, circle))
        self.play(FadeOut(square))
""",
        scene_name="SquareToCircle",
        quality="l"
    )
    result = await skill.safe_execute(params)
    print(result)

if __name__ == "__main__":
    asyncio.run(test())
