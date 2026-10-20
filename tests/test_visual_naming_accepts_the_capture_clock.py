"""The ordinary capture clock must reach instance naming without changing ownership."""
import asyncio

import numpy as np
import pytest

from core.perception.frames_as_they_are_drawn import CanvasFrameTime
from core.perception.what_things_look_like import LookingAtThings
from tests.test_visual_names_stay_with_the_boxed_instance import _thing, _scene

pytestmark = pytest.mark.unit


def _clock(at):
    return CanvasFrameTime(at,lower=at,upper=at+.001,arrived=at+.002,source='browser-wall',
                           source_at=at+100.,epoch=1,surface='canvas',sequence=int(at*100))


@pytest.mark.parametrize('clock',[_clock,float,np.float32,np.float64])
def test_current_capture_times_reach_the_provider_with_the_original_box_and_birth(clock):
    async def observe():
        calls=[]
        async def provider(prompt,image):
            calls.append(image)
            return '{"is":"platform"}'
        looking=LookingAtThings(see=provider)
        thing=_thing(at=clock(0.),born=0.)
        scene=_scene(thing)
        picture=np.zeros((200,300,3),np.uint8)
        assert not looking.look(picture,scene,clock(0.))
        thing.seen=clock(4.)
        assert looking.look(picture,scene,clock(4.))
        await looking._asking
        sighting,=looking.by_instance.values()
        assert sighting.query.number==thing.number and sighting.query.born==thing.born
        assert sighting.query.at==4. and sighting.query.box==thing.box()
        assert len(sighting.query.picture_sha256)==64 and len(calls)==1
    asyncio.run(observe())


@pytest.mark.parametrize('invalid',[True,False,None,'4',float('nan'),float('inf'),-1.,10**400,4+0j])
def test_invalid_times_cannot_start_or_age_a_naming_request(invalid):
    looking=LookingAtThings()
    assert not looking.look(np.zeros((200,300,3),np.uint8),_scene(_thing()),invalid)
    assert looking._began is None and looking._asking is None and not looking.asked_instances


def test_the_capture_clock_has_a_registered_measured_claim():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    suite=ValidationSuite()
    install_realtime_control_claims(suite)
    check=next(t for t in suite.tests() if t.name=='visual_naming_accepts_measured_capture_times')
    assert check.predict(None) is True and any(c.test==check.name for c in suite.claims())
