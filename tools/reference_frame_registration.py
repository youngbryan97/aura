"""Measure a static export's source coordinates through a second rasterizer."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from core.perception.coordinates_in_a_visual_reference import (
    MAX_SVG_BYTES,
    calibrated_source_coordinates,
    exported_frame_coordinates,
)


async def rasterized_frame(svg: bytes) -> np.ndarray:
    """Rasterize a static coordinate document in an owned software browser."""
    from core.capabilities.phantom_browser import PhantomBrowser

    size, _linear, _offset = exported_frame_coordinates(svg)
    # No viewBox is admitted by the adapter. Integral viewport padding leaves
    # every source coordinate unchanged, including fractional frame dimensions.
    root = ET.fromstring(svg)
    root.set('width', f'{size[0]}px')
    root.set('height', f'{size[1]}px')
    encoded = base64.b64encode(ET.tostring(root)).decode()
    browser = PhantomBrowser(visible=False, graphics_backend='software', principal='owner')
    try:
        if not await browser.ensure_ready():
            raise ValueError('the independent CPU rasterizer is unavailable')
        data = await browser.page.evaluate('''async ([encoded,width,height])=>{
          const image=new Image();image.src='data:image/svg+xml;base64,'+encoded;await image.decode();
          const canvas=document.createElement('canvas');canvas.width=width;canvas.height=height;
          const ctx=canvas.getContext('2d');ctx.drawImage(image,0,0);
          const pixels=ctx.getImageData(0,0,width,height).data;let text='';
          for(let i=0;i<pixels.length;i+=32768)text+=String.fromCharCode(...pixels.subarray(i,i+32768));
          return btoa(text);
        }''', [encoded, *size])
        raw = base64.b64decode(data, validate=True)
        if len(raw) != size[0]*size[1]*4:
            raise ValueError('the independent raster has the wrong byte length')
        return np.frombuffer(raw, np.uint8).reshape(size[1], size[0], 4).copy()
    except Exception as exc:
        raise ValueError(f'independent rasterization failed: {type(exc).__name__}: {exc}') from exc
    finally:
        await browser.close()


def _svg_inventory(directory: Path, character: int) -> Path:
    files = []
    for count, path in enumerate(directory.rglob('*')):
        if count >= 2048 or path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError('coordinate export leaves its path budget')
        if not path.is_file():
            continue
        relative = path.relative_to(directory)
        if (path.suffix.lower() != '.svg' or path.stat().st_size > MAX_SVG_BYTES
                or not re.match(rf'Define[A-Za-z0-9]+_{character}(?:_|\.|$)', relative.parts[0])):
            raise ValueError('coordinate export has an unexpected file')
        files.append(path)
        if len(files) > 1:
            raise ValueError('coordinate registration requires one frame')
    if len(files) != 1:
        raise ValueError('coordinate export supplied no frame')
    return files[0]


def register_export_frame(java: Path, jar: Path, source: Path, out: Path, asset: object,
                          images: list[dict], source_id: str, *, frame: int = 1) -> dict:
    """Export the same source frame, then measure its registration into the PNG."""
    from core.runtime.file_write_gateway import get_file_write_gateway
    from tools.export_program_visual_reference import _render_command
    from tools.run_proof_step import main as run_step

    if asset.kind not in ('sprite', 'button') or len(images) != 1:
        raise ValueError('source registration requires one sprite or button frame')
    directory, home = out/'source-coordinates', out/'renderer-home'
    get_file_write_gateway().ensure_directory(directory, source='reference_frame_registration')
    command = _render_command(java, jar, source, directory, home, asset, frame=frame)
    command[command.index(f'{asset.kind}:png')] = f'{asset.kind}:svg'
    code = run_step(['--name', 'static_reference_coordinates', '--timeout', '60',
                     '--artifact', str(out/'coordinate-supervisor.json'), '--', *command])
    if code:
        raise ValueError(f'coordinate renderer failed with status {code}')
    svg_file = _svg_inventory(directory, asset.character_id)
    svg = svg_file.read_bytes()
    pixels = asyncio.run(rasterized_frame(svg))
    encoded = io.BytesIO()
    Image.fromarray(pixels).save(encoded, format='PNG')
    raster_file = out/'coordinate-raster.png'
    get_file_write_gateway().write_bytes(raster_file, encoded.getvalue(), source='reference_frame_registration')
    png = out/'rendered'/images[0]['path']
    if png.is_symlink() or hashlib.sha256(png.read_bytes()).hexdigest() != images[0]['file_sha256']:
        raise ValueError('the reference PNG changed before coordinate calibration')
    with Image.open(png) as image:
        reference = np.asarray(image.convert('RGBA'))
    coordinates = calibrated_source_coordinates(source_id, svg, pixels, reference)
    if coordinates is None:
        raise ValueError('the independent raster did not align with the reference PNG')
    return dict(coordinates=asdict(coordinates), svg=str(svg_file.relative_to(out)), command=command,
                raster=str(raster_file.relative_to(out)), raster_file_sha256=hashlib.sha256(encoded.getvalue()).hexdigest(),
                raster_raw_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                rasterizer='owned Chromium software browser',
                evidence_scope='source-coordinate registration into reference bytes; no current scene, identity or input effect')
