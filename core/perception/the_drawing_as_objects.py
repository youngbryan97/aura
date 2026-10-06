"""Visible drawing geometry, when the renderer can describe it directly.

This observes the canvas drawing API, never application variables. Unsupported
paint, clipping and unbounded histories leave the picture as the observation.
Pixels and the display list are captured together; no input or drawing is made
by the observer. Callers continue to receive an RGB array.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from core.verify import invariant

_OBSERVER = r"""
((active) => {
  if (window.__auraDrawingObservation) { if(active)window.__auraDrawingObservation.acquire(); return true; }
  if (typeof CanvasRenderingContext2D === 'undefined') return false;
  const proto = CanvasRenderingContext2D.prototype, replaced = [], preloaded = !active;
  let owners = active ? 1 : 0, states = new WeakMap();
  const state = (c) => {
    let s = states.get(c);
    if (!s || s.width !== c.canvas.width || s.height !== c.canvas.height) {
      s = {width:c.canvas.width,height:c.canvas.height,objects:[],texts:[],complete:false,path:null,clip:false,stack:[],overflow:0};
      states.set(c,s);
    }
    return s;
  };
  const box = (c,x,y,w,h) => {
    const t = c.getTransform();
    if (Math.abs(t.b)>1e-9 || Math.abs(t.c)>1e-9) return null;
    const a=t.a*x+t.e,b=t.d*y+t.f,d=t.a*(x+w)+t.e,e=t.d*(y+h)+t.f;
    return [Math.min(a,d),Math.min(b,e),Math.abs(d-a),Math.abs(e-b)];
  };
  const plain = (c,s) => !s.clip && c.globalAlpha === 1 && c.globalCompositeOperation === 'source-over'
    && !c.shadowBlur && !c.shadowOffsetX && !c.shadowOffsetY && (!c.filter || c.filter === 'none');
  const opaque = (style) => typeof style === 'string' && (/^#[0-9a-f]{6}$/i.test(style) || /^rgb\(/.test(style));
  const full = (s,b) => b && b[0]<=0 && b[1]<=0 && b[0]+b[2]>=s.width && b[1]+b[3]>=s.height;
  const add = (c,s,b,shape,style) => {
    if (!b || !plain(c,s) || !opaque(style)) { s.complete=false; return; }
    s.objects.push({x:b[0],y:b[1],width:b[2],height:b[3],shape,colour:style});
    if (s.objects.length+s.texts.length>1000) { s.objects=[];s.texts=[];s.complete=false; }
  };
  const wrap = (name,observe) => {
    const original=proto[name]; if (typeof original !== 'function') return;
    const wrapped=function(...args) {
      const result=Reflect.apply(original,this,args);
      if(!owners)return result;
      try { observe(this,state(this),args); } catch (_) { const s=state(this);s.complete=false; }
      return result;
    };
    proto[name]=wrapped; replaced.push([name,original,wrapped]);
  };
  wrap('save',(c,s)=>{if(s.stack.length<256)s.stack.push(s.clip);else{s.overflow++;s.clip=true;s.complete=false;}});
  wrap('restore',(c,s)=>{if(s.overflow)s.overflow--;else if(s.stack.length)s.clip=s.stack.pop();});
  wrap('clip',(c,s)=>{s.clip=true;s.complete=false;});
  wrap('clearRect',(c,s,a)=>{
    const b=box(c,...a);
    if(full(s,b)&&!s.clip){s.objects=[];s.texts=[];s.complete=true;}
    else s.complete=false;
  });
  wrap('fillRect',(c,s,a)=>{
    const b=box(c,...a);
    if(full(s,b)&&plain(c,s)&&opaque(c.fillStyle)){s.objects=[];s.texts=[];s.complete=true;return;}
    add(c,s,b,'rectangle',c.fillStyle);
  });
  wrap('beginPath',(c,s)=>{s.path=null;});
  wrap('rect',(c,s,a)=>{s.path=s.path?{unknown:true}:{box:box(c,...a),shape:'rectangle'};});
  wrap('arc',(c,s,a)=>{
    const [x,y,r,start,end]=a;
    s.path=!s.path && Math.abs(end-start)>=2*Math.PI-0.001
      ? {box:box(c,x-r,y-r,2*r,2*r),shape:'circle'} : {unknown:true};
  });
  for(const name of ['moveTo','lineTo','bezierCurveTo','quadraticCurveTo','arcTo','ellipse','roundRect'])
    wrap(name,(c,s)=>{s.path={unknown:true};});
  wrap('fill',(c,s,a)=>{
    if(a[0] instanceof Path2D || !s.path || s.path.unknown) {s.complete=false;return;}
    add(c,s,s.path.box,s.path.shape,c.fillStyle);
  });
  // Stroked paths, image transparency and arbitrary path bounds remain pixels.
  for(const name of ['stroke','strokeRect','drawImage','putImageData','strokeText'])
    wrap(name,(c,s)=>{s.complete=false;});
  wrap('fillText',(c,s,a)=>{
    if(!plain(c,s)||!opaque(c.fillStyle)){s.complete=false;return;}
    const [text,x,y,maxWidth]=a,m=c.measureText(String(text));
    if(maxWidth!==undefined&&maxWidth<m.width){s.complete=false;return;}
    const b=box(c,x-m.actualBoundingBoxLeft,y-m.actualBoundingBoxAscent,
      m.actualBoundingBoxLeft+m.actualBoundingBoxRight,m.actualBoundingBoxAscent+m.actualBoundingBoxDescent);
    if(!b){s.complete=false;return;}
    if(String(text).length>2048){s.complete=false;return;}
    s.texts.push({text:String(text),x:b[0],y:b[1],width:b[2],height:b[3]});
    if(s.objects.length+s.texts.length>1000){s.objects=[];s.texts=[];s.complete=false;}
  });
  window.__auraDrawingObservation={
    acquire(){owners++;},
    snapshot(canvas){
      const c=canvas.getContext('2d');if(!c)return null;
      const s=state(c);
      return {source:'canvas-paint-v1',width:s.width,height:s.height,complete:s.complete,objects:s.objects,texts:s.texts};
    },
    release(){
      if(--owners>0)return;
      if(preloaded){owners=0;states=new WeakMap();return;}
      for(const [name,original,wrapped] of replaced)if(proto[name]===wrapped)proto[name]=original;
      delete window.__auraDrawingObservation;
    }
  };
  return true;
})
"""

# Preload before application code can cache a drawing method. Observing begins
# only when a frame reader acquires it. A preload belongs to the page's realm.
INSTALL = _OBSERVER + "(true)"
BOOTSTRAP = _OBSERVER + "(false)"

RELEASE = "window.__auraDrawingObservation?.release()"


async def acquire(page: Any) -> bool:
    """Acquire the fixed renderer observer; callers cannot supply JavaScript."""
    return bool(await page.evaluate(INSTALL))


async def release(page: Any) -> None:
    """Release this frame owner's observation lease without drawing or input."""
    await page.evaluate(RELEASE)


class ObservedPicture(np.ndarray):
    """An RGB picture with an optional description of the same rendered scene."""

    drawing_scene: dict[str, Any] | None = None

    def __array_finalize__(self, source: Any) -> None:
        self.drawing_scene = getattr(source, "drawing_scene", None)


def described(picture: np.ndarray, scene: Any) -> np.ndarray:
    """Attach only a bounded, complete display list with matching pixel dimensions."""
    if not isinstance(scene, dict) or scene.get("source") != "canvas-paint-v1" or scene.get("complete") is not True:
        return picture
    if (scene.get("height"), scene.get("width")) != picture.shape[:2]:
        return picture
    objects, texts = scene.get("objects"), scene.get("texts")
    if not isinstance(objects, list) or not isinstance(texts, list) or len(objects) + len(texts) > 1000:
        return picture
    for item, is_text in [(o, False) for o in objects] + [(t, True) for t in texts]:
        if not isinstance(item, dict):
            return picture
        bounds = [item.get(key) for key in ("x", "y", "width", "height")]
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or abs(v) > 4 * max(picture.shape[:2])
               for v in bounds) or min(bounds[2:]) < 0:
            return picture
        if is_text and (not isinstance(item.get("text"), str) or len(item["text"]) > 2048):
            return picture
    observed = picture.view(ObservedPicture)
    observed.drawing_scene = scene
    return observed


def words_in(picture: Any) -> list[dict[str, Any]] | None:
    """Renderer text in the same normalized coordinates as the screen reader."""
    scene = getattr(picture, "drawing_scene", None)
    if scene is None:
        return None
    wide, tall = scene["width"], scene["height"]
    return [{"text": t["text"], "x": t["x"] / wide, "y": t["y"] / tall,
             "width": t["width"] / wide, "height": t["height"] / tall,
             "center_x": (t["x"] + t["width"] / 2) / wide,
             "center_y": (t["y"] + t["height"] / 2) / tall}
            for t in scene["texts"]]


def _scene_requires_a_matching_complete_frame() -> bool:
    image = np.zeros((12, 20, 3), dtype=np.uint8)
    scene = {"source": "canvas-paint-v1", "complete": True, "width": 20, "height": 12,
             "objects": [], "texts": []}
    if getattr(described(image, scene), "drawing_scene", None) is None:
        return False
    for change in ({"width": 21}, {"complete": False}, {"source": "unknown"},
                   {"objects": [{"x": float("nan"), "y": 0, "width": 1, "height": 1}]}):
        if described(image, {**scene, **change}) is not image:
            return False
    return True


@invariant("perception.drawing_scene_requires_matching_frame", scope="perception",
           owner="core/perception/the_drawing_as_objects.py", observational=False)
def _drawing_scene_invariant() -> tuple:
    assert _scene_requires_a_matching_complete_frame(), "unverified scene data replaced the picture"
    return ()
