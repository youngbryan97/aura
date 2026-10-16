"""Source coordinates must traverse the actual raster lattice, including rounded axes."""
import numpy as np
import pytest
from PIL import Image

from core.perception.the_pixel_grid_a_mark_uses import SampleGrid

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('factor', [2, 4, 8])
def test_a_native_sample_maps_to_the_centroid_of_its_independent_source_footprint(factor):
    # A block filling exactly one reduced sample has a measurable centroid
    # in the original image. No production coordinate formula draws it.
    image = np.zeros((96, 128), np.uint8)
    image[5*factor:6*factor, 7*factor:8*factor] = 255
    sampled = np.asarray(Image.fromarray(image).resize((128//factor, 96//factor), Image.Resampling.BOX))
    sy, sx = np.unravel_index(sampled.argmax(), sampled.shape)
    yy, xx = np.indices(image.shape)
    centroid = (float((xx*image).sum()/image.sum()), float((yy*image).sum()/image.sum()))
    grid = SampleGrid(image.shape, sampled.shape, (48, 64))
    published = sx*64/sampled.shape[1], sy*48/sampled.shape[0]
    assert grid.to_image(published) == pytest.approx(centroid)
    assert grid.from_image(centroid) == pytest.approx(published)


def test_rounded_vertical_dimensions_keep_their_own_ratio():
    grid = SampleGrid((1522, 2560), (391, 657), (285, 480))
    native = np.array((200., 300.))
    published = native*np.array((480/657, 285/391))
    expected = (native+.5)*np.array((2560/657, 1522/391))-.5
    assert grid.to_image(published) == pytest.approx(expected)
    assert grid.from_image(expected) == pytest.approx(published)
    assert grid.pixel_size == max(480/657, 285/391)


@pytest.mark.parametrize('shape', [None, (0, 20), (10, float('nan')), (4097, 20), (4096, 4096)])
def test_unmeasured_or_oversized_grids_cannot_supply_a_mapping(shape):
    with pytest.raises(ValueError):
        SampleGrid(shape, (20, 20), (20, 20))
