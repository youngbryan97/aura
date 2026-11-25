"""The pictures of a program she makes: decided in its design, painted by her image model, chosen by what is measured of them."""
from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from core.making import maker, pictures
from core.making.design import Design, Look, Picture, Rule, with_an_icon
from core.making.trying import page_of
from core.understanding.workbench import Work
from tests.test_making_programs import GOOD, _in_a_browser


@pytest.fixture(autouse=True)
def written(monkeypatch):
    from core.rebuilding import her_build_writes

    def governed(what, path, *args, **_how):
        path = Path(path)
        if what == "ensure_directory":
            path.mkdir(parents=True, exist_ok=True)
        elif what == "write_bytes":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(args[0])
        elif what == "write_text":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(args[0], encoding="utf-8")
        else:
            path.unlink(missing_ok=True)

    monkeypatch.setattr(her_build_writes, "governed", governed)


def _design(**over) -> Design:
    return Design(**{"name": "Moth Lamp", "kind": "game", "about": "Keep the moths from the lamp.", "rules": [Rule(says="x")], "look": Look(accent="#e8a13a"),
                     "state": {"x": 1}, **over})


def _thing(path: Path, colour=(200, 40, 40), ground=(250, 250, 250), box=(150, 120, 380, 400), noise: int = 0) -> Path:
    image = Image.new("RGB", (512, 512), ground)
    ImageDraw.Draw(image).ellipse(box, fill=colour)
    if noise:
        rng = np.random.default_rng(1)
        image = Image.fromarray(rng.integers(0, 255, (512, 512, 3), dtype="uint8"))
    image.save(path)
    return path


def test_a_program_has_something_to_be_known_by_though_its_design_named_nothing():
    (icon,) = with_an_icon(_design()).pictures
    assert icon.kind == "icon" and icon.shows == "Keep the moths from the lamp."
    several = _design(pictures=[Picture(name="a", kind="icon", shows="a moth"), Picture(name="b", kind="icon", shows="a lamp"), Picture(name="moth", kind="thing", shows="a moth"),
                                Picture(name="moth", kind="thing", shows="another")])
    assert [(p.kind, p.shows) for p in with_an_icon(several).pictures] == [("icon", "a moth"), ("thing", "a moth")]
    assert with_an_icon(_design(kind="tool")).pictures == []


def test_a_colour_is_called_what_it_is():
    assert [pictures.colour_word(c) for c in ("#0a0a0a", "#ffffff", "#3a66e6", "#1f8a8a", "nonsense")] == ["black", "white", "blue", "teal", ""]


def test_a_thing_is_cut_from_its_ground_and_what_has_no_plain_ground_is_not(tmp_path):
    worth, cut = pictures.taken_from_its_ground(Image.open(_thing(tmp_path / "a.png")))
    alpha = np.asarray(cut)[..., 3]
    assert cut.mode == "RGBA" and cut.size[0] == cut.size[1] and alpha[0, 0] == 0 and alpha[alpha.shape[0] // 2, alpha.shape[1] // 2] == 255 and worth > 1
    edge, _cut = pictures.taken_from_its_ground(Image.open(_thing(tmp_path / "b.png", box=(-60, 120, 300, 400))))
    assert edge < worth, "one that runs off the picture is worth less"
    assert pictures.taken_from_its_ground(Image.open(_thing(tmp_path / "c.png", noise=1))) is None
    assert pictures.taken_from_its_ground(Image.new("RGB", (512, 512), "white")) is None


def test_an_icon_that_reads_small_and_is_unlike_hers_is_the_one_that_is_used(tmp_path):
    work, asked = Work("w", "moths"), []
    flat, bold = tmp_path / "flat.png", tmp_path / "bold.png"
    Image.new("RGB", (512, 512), (90, 90, 96)).save(flat)
    _thing(bold, colour=(250, 220, 60), ground=(20, 30, 80))
    given = iter([flat, bold, flat])

    async def paints(seen, width, height, seed):
        asked.append((seen, width, height, seed))
        return next(given)

    design = with_an_icon(_design(pictures=[Picture(name="icon", kind="icon", shows="a moth over a lamp")]))
    assert asyncio.run(pictures.paint(design, tmp_path / "moth", work, paints=paints)) == ["icon"]
    assert len(asked) == 3 and len({seed for *_rest, seed in asked}) == 3 and "a moth over a lamp" in asked[0][0] and "plain gold background" in asked[0][0]
    with Image.open(tmp_path / "moth" / "icon.png") as kept:
        assert kept.getpixel((256, 256)) == (250, 220, 60)
    assert [row["name"] for row in work.seen()["happened"] if row["what"] == "painted"] == ["icon"]
    twin = pictures._stands_apart(Image.open(bold), [bold])
    assert twin < pictures._stands_apart(Image.open(bold), [flat]) <= pictures._stands_apart(Image.open(bold), [])


def test_what_was_painted_is_not_painted_again_and_what_cannot_be_is_left(tmp_path):
    async def nothing(*_a):
        return None

    design = with_an_icon(_design(pictures=[Picture(name="moth", kind="thing", shows="a moth")]))
    assert asyncio.run(pictures.paint(design, tmp_path, Work("w", "x"), paints=nothing)) == []
    assert pictures.as_data(tmp_path, design) == {} and pictures.icon_at(tmp_path) is None
    _thing(tmp_path / "icon.png")

    async def moth(seen, *_a):
        assert "moth" in seen and "plain flat white background" in seen
        return _thing(tmp_path / "painted.png")

    assert asyncio.run(pictures.paint(design, tmp_path, Work("w", "x"), paints=moth)) == ["icon", "moth"]
    data = pictures.as_data(tmp_path, design)
    assert set(data) == {"icon", "moth"} and data["moth"].startswith("data:image/png") and data["icon"].startswith("data:image/jpeg") and len(data["moth"]) < 200_000


def test_a_game_draws_what_was_painted_and_is_drawn_without_it_too(tmp_path):
    design = with_an_icon(_design(pictures=[Picture(name="moth", kind="thing", shows="a moth"), Picture(name="room", kind="backdrop", shows="a dark room")], pieces=["vec"]))
    _thing(tmp_path / "icon.png")
    (tmp_path / "pictures").mkdir()
    pictures.taken_from_its_ground(Image.open(_thing(tmp_path / "m.png")))[1].save(tmp_path / "pictures" / "moth.png")
    _thing(tmp_path / "pictures" / "room.png", colour=(0, 0, 120), ground=(10, 20, 60))
    code = GOOD.replace('g.sky("#11142a", "#1d1233");', 'g.backdrop("room", { scroll: s.x });').replace('g.circle(m.x, m.y, 9, "#cfc7ff")', 'g.picture("moth", m.x, m.y, { size: 24 })')
    page = tmp_path / "index.html"
    page.write_text(page_of(design, code, pictures=pictures.as_data(tmp_path, design)), encoding="utf-8")
    bare = tmp_path / "bare.html"
    bare.write_text(page_of(design, code), encoding="utf-8")

    async def run(browser):
        out = []
        for path in (page, bare):
            view = await browser.new_page(viewport={"width": 960, "height": 700})
            wrong = []
            view.on("pageerror", lambda error, wrong=wrong: wrong.append(str(error)))
            await view.goto(path.as_uri())
            await view.wait_for_timeout(700)
            out.append((wrong, await view.evaluate("() => { const c = document.querySelector('canvas'), d = c.getContext('2d').getImageData(5, Math.round(c.height / 2), 1, 1).data; return [...d]; }"),
                        await view.evaluate("() => !!document.querySelector('link[rel=icon]')")))
        return out

    (errors, pixel, has_icon), (bare_errors, _pixel, bare_icon) = _in_a_browser(run)
    assert errors == [] and bare_errors == [] and has_icon and not bare_icon
    assert pixel[2] > pixel[0], "the painted room is what is behind it"


def test_its_pictures_are_painted_beside_the_writing_and_waited_for_at_the_end(tmp_path):
    made = maker._Maker("x", ask=None, write=None, work=Work("w", "x"), folder=tmp_path, browser=None)
    made.design = with_an_icon(_design())

    async def paints(seen, *_a):
        await asyncio.sleep(0.01)
        return _thing(tmp_path / "p.png")

    async def run():
        made.paints = paints
        made.begin_painting()
        assert made.painting is not None and not made.painting.done()
        await made.painted()
        return made.painting.result()

    assert asyncio.run(run()) == ["icon"] and pictures.icon_at(tmp_path) is not None


def test_its_icon_is_the_picture_rounded_as_mac_icons_are(tmp_path):
    from core.rebuilding import as_a_mac_app

    drawn = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
    assert as_a_mac_app._laid_over(drawn, _thing(tmp_path / "p.png", colour=(0, 0, 250), ground=(10, 200, 10)), 100, 824)
    assert drawn.getpixel((110, 110))[3] == 0 and drawn.getpixel((512, 512))[2] > 200 and drawn.getpixel((512, 160))[1] > 150
    assert not as_a_mac_app._laid_over(drawn, tmp_path / "nothing.png", 100, 824)


def test_half_of_her_painter_is_held_in_half_precision_and_its_decoder_whole():
    """What overflows on this machine's processor is the decoder; the rest of it in half is half the memory."""
    from core.skills.image_gen import ImageGenSkill

    skill = ImageGenSkill.__new__(ImageGenSkill)
    skill._device, skill._model_id = "mps", "stabilityai/sdxl-turbo"
    assert skill._halves("stabilityai/sdxl-turbo") and not skill._halves("runwayml/stable-diffusion-v1-5")

    class Part:
        def __init__(self, dtype):
            self.dtype = dtype

    pipe = type("P", (), {})()
    pipe.vae, pipe.unet = Part("whole"), Part("half")
    assert skill._as_latents(pipe) == {"output_type": "latent"}
    pipe.unet = Part("whole")
    assert skill._as_latents(pipe) == {} and skill._decoded(pipe, ["picture"]) == "picture"

