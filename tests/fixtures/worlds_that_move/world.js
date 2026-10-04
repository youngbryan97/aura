// The small engine every test world here shares: a seeded random number, the
// keys being held, a title screen, a game-over screen and a clock that runs on
// real time, so a slow frame does not slow the world down.
//
// Nothing that plays these worlds may read anything in this file. The player
// sees pixels and sends keys and clicks. `window.__world` exists only so a
// measurement can say how a game went.
(function () {
  const params = new URLSearchParams(location.search);
  let seed = (Number(params.get("seed") || 1) * 2654435761) >>> 0;
  function rand() {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 4294967296;
  }
  const held = new Set();
  const named = { w: "ArrowUp", s: "ArrowDown", a: "ArrowLeft", d: "ArrowRight" };
  addEventListener("keydown", (e) => {
    const key = named[e.key] || e.key;
    held.add(key);
    if (key.startsWith("Arrow") || key === " ") e.preventDefault();
    if (world.state !== "play" && (key === "Enter" || key === " ")) world.begin();
  });
  addEventListener("keyup", (e) => held.delete(named[e.key] || e.key));

  const canvas = document.createElement("canvas");
  canvas.width = 480;
  canvas.height = 320;
  canvas.style.display = "block";
  canvas.style.margin = "24px auto";
  document.body.style.margin = "0";
  document.body.style.background = "#222";
  document.body.appendChild(canvas);
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  const button = { x: W / 2 - 60, y: H / 2 + 30, w: 120, h: 36 };

  canvas.addEventListener("mousedown", (e) => {
    const r = canvas.getBoundingClientRect();
    const x = ((e.clientX - r.left) * W) / r.width, y = ((e.clientY - r.top) * H) / r.height;
    if (world.state !== "play") {
      if (x >= button.x && x <= button.x + button.w && y >= button.y && y <= button.y + button.h) world.begin();
      return;
    }
    if (world.game.click) world.game.click(x, y);
  });
  canvas.addEventListener("mousemove", (e) => {
    const r = canvas.getBoundingClientRect();
    world.pointer = { x: ((e.clientX - r.left) * W) / r.width, y: ((e.clientY - r.top) * H) / r.height };
  });

  function text(said, x, y, size, colour, align) {
    ctx.fillStyle = colour || "#fff";
    ctx.font = "bold " + (size || 16) + "px sans-serif";
    ctx.textAlign = align || "center";
    ctx.fillText(said, x, y);
  }

  function screen(title, lines, label) {
    world.game.backdrop(ctx);
    ctx.fillStyle = "rgba(0,0,0,0.55)";
    ctx.fillRect(0, 0, W, H);
    text(title, W / 2, 70, 30, "#ffd84a");
    lines.forEach((line, i) => text(line, W / 2, 110 + i * 22, 15, "#fff"));
    ctx.fillStyle = "#3a7bd5";
    ctx.fillRect(button.x, button.y, button.w, button.h);
    text(label, W / 2, button.y + 24, 18, "#fff");
  }

  const world = {
    W, H, rand, held, text, ctx, state: "title", pointer: { x: W / 2, y: H / 2 },
    score: 0, lives: 3, games: 0, ended: [],
    begin() {
      this.state = "play";
      this.score = 0;
      this.lives = this.game.lives === undefined ? 3 : this.game.lives;
      this.started = performance.now();
      this.game.reset(this);
    },
    over(why) {
      this.state = "over";
      this.games += 1;
      this.ended.push({ score: this.score, why, seconds: (performance.now() - this.started) / 1000 });
    },
    lose() {
      this.lives -= 1;
      this.losses = (this.losses || 0) + 1;
      if (this.lives <= 0) this.over("no lives left");
    },
  };
  window.__world = world;

  window.playWorld = function (game) {
    world.game = game;
    let last = performance.now();
    function frame(now) {
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      if (world.state === "play") {
        game.step(dt, world);
        game.backdrop(ctx);
        game.draw(ctx, world);
        game.readouts(ctx, world);
      } else if (world.state === "title") {
        screen(game.title, game.rules, "PLAY");
      } else {
        screen("GAME OVER", ["Score " + world.score], "PLAY AGAIN");
      }
      requestAnimationFrame(frame);
    }
    if (params.get("start") === "1") world.begin();
    requestAnimationFrame(frame);
  };

  window.hits = function (a, b) {
    return a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;
  };
  window.scoreAndLives = function (ctx, world) {
    text("SCORE " + world.score, 12, 22, 16, "#fff", "left");
    text("LIVES " + world.lives, W - 12, 22, 16, "#fff", "right");
  };
})();
