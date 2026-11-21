//! Search for games of pieces that move by their kind, read from a description.
//!
//! The game comes in as data (core/agency/pieces_by_their_kind.py): a board's size, each kind as steps (leaps and
//! rides, steps that only move or only take, steps only an unmoved piece makes, passing), royal kinds, kinds that
//! become others on the far row, a royal that crosses with a partner, and the draw counts. Nothing here knows a game.
//! What a kind and a square are worth comes in as weights measured by playing; the search is alpha-beta with the
//! general refinements of game-tree search: iterative deepening, a transposition table, quiescence on takes, null
//! moves, late-move reductions, killer and history ordering.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde::Deserialize;
use std::collections::HashMap;
use std::time::Instant;

const EMPTY: u8 = 0;
const OFF: u8 = 255;
const NONE: u16 = u16::MAX;
const MATE: i32 = 1_000_000;
const INF: i32 = 2_000_000;
const MAX_PLY: usize = 128;

#[derive(Deserialize, Clone)]
struct StepD {
    offsets: Vec<(i32, i32)>,
    rides: i32,
    may: String,
    only_unmoved: bool,
    leaves_passable: bool,
    takes_in_passing: bool,
}

#[derive(Deserialize, Clone)]
struct KindD {
    letter: String,
    steps: Vec<StepD>,
    royal: bool,
    becomes: Vec<String>,
    cannot_go_back: bool,
    crosses_with: String,
}

#[derive(Deserialize)]
struct FamilyD {
    rows: i32,
    columns: i32,
    kinds: Vec<KindD>,
    count_to_draw: i32,
    repeats_to_draw: i32,
    no_move_unattacked: String,
}

#[derive(Deserialize, Clone)]
struct PosD {
    cells: Vec<(i32, i32, i32, String)>,
    turn: i32,
    unmoved: Vec<(i32, i32)>,
    passable: Option<((i32, i32), (i32, i32))>,
    count: i32,
    #[serde(default)]
    history: Vec<PosD>,
}

#[derive(Deserialize, Default)]
struct WeightsD {
    #[serde(default)]
    worth: HashMap<String, f64>,
    #[serde(default)]
    mobility: HashMap<String, f64>,
    /// Per kind, rows by columns in the first side's frame; the second side's is mirrored across the middle row.
    #[serde(default)]
    squares: HashMap<String, Vec<Vec<f64>>>,
    #[serde(default)]
    tempo: f64,
    /// Per attack on a square next to a royal piece, against that royal's side.
    #[serde(default)]
    royal_pressed: f64,
}

#[derive(Clone)]
struct Step {
    deltas: [Vec<i32>; 2],
    rides: i32,
    may: u8, // 0 either, 1 only moves, 2 only takes
    only_unmoved: bool,
    leaves_passable: bool,
    takes_in_passing: bool,
}

#[derive(Clone)]
struct Kind {
    letter: String,
    steps: Vec<Step>,
    royal: bool,
    becomes: Vec<u8>,
    cannot_go_back: bool,
    crosses_with: Option<u8>,
    worth: i32,
    mobility: i32,
    squares: [Vec<i32>; 2],
}

struct Game {
    rows: i32,
    cols: i32,
    border: i32,
    stride: i32,
    size: usize,
    kinds: Vec<Kind>,
    count_to_draw: i32,
    repeats_to_draw: i32,
    no_move_draws: bool,
    tempo: i32,
    royal_pressed: i32,
    zob_piece: Vec<u64>,
    zob_unmoved: Vec<u64>,
    zob_passable: Vec<u64>,
    zob_turn: u64,
    on_board: Vec<usize>,
}

fn code(kind: usize, side: usize) -> u8 {
    (1 + kind * 2 + side) as u8
}
fn kind_of(c: u8) -> usize {
    ((c - 1) / 2) as usize
}
fn side_of(c: u8) -> usize {
    ((c - 1) % 2) as usize
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

impl Game {
    fn new(f: &FamilyD, w: &WeightsD) -> Result<Game, String> {
        let mut border = 1;
        for k in &f.kinds {
            for s in &k.steps {
                for (a, b) in &s.offsets {
                    border = border.max(a.abs()).max(b.abs());
                }
            }
        }
        let stride = f.columns + 2 * border;
        let size = ((f.rows + 2 * border) * stride) as usize;
        let letters: Vec<String> = f.kinds.iter().map(|k| k.letter.clone()).collect();
        let index_of = |l: &str| letters.iter().position(|x| x == l);
        let mut kinds = Vec::new();
        for k in &f.kinds {
            let mut steps = Vec::new();
            for s in &k.steps {
                let mut deltas: [Vec<i32>; 2] = [Vec::new(), Vec::new()];
                for side in 0..2 {
                    let ahead = if side == 0 { 1 } else { -1 };
                    for (fw, rt) in &s.offsets {
                        deltas[side].push(fw * ahead * stride + rt * ahead);
                    }
                }
                let may = match s.may.as_str() {
                    "moves" => 1,
                    "takes" => 2,
                    _ => 0,
                };
                steps.push(Step {
                    deltas,
                    rides: s.rides,
                    may,
                    only_unmoved: s.only_unmoved,
                    leaves_passable: s.leaves_passable,
                    takes_in_passing: s.takes_in_passing,
                });
            }
            let mut becomes = Vec::new();
            for b in &k.becomes {
                becomes.push(index_of(b).ok_or(format!("unknown kind {b}"))? as u8);
            }
            let crosses_with = if k.crosses_with.is_empty() {
                None
            } else {
                Some(index_of(&k.crosses_with).ok_or("unknown partner")? as u8)
            };
            let worth = (w.worth.get(&k.letter).copied().unwrap_or(0.0)).round() as i32;
            let mobility = (w.mobility.get(&k.letter).copied().unwrap_or(0.0)).round() as i32;
            let mut squares: [Vec<i32>; 2] = [vec![0; size], vec![0; size]];
            if let Some(table) = w.squares.get(&k.letter) {
                for r in 0..f.rows {
                    for c in 0..f.columns {
                        let v = table
                            .get(r as usize)
                            .and_then(|row| row.get(c as usize))
                            .copied()
                            .unwrap_or(0.0)
                            .round() as i32;
                        let sq0 = ((r + border) * stride + c + border) as usize;
                        let sq1 = ((f.rows - 1 - r + border) * stride + c + border) as usize;
                        squares[0][sq0] = v;
                        squares[1][sq1] = v;
                    }
                }
            }
            kinds.push(Kind {
                letter: k.letter.clone(),
                steps,
                royal: k.royal,
                becomes,
                cannot_go_back: k.cannot_go_back,
                crosses_with,
                worth,
                mobility,
                squares,
            });
        }
        let mut rng = Rng(0x9E3779B97F4A7C15);
        let codes = 1 + kinds.len() * 2;
        let zob_piece = (0..codes * size).map(|_| rng.next()).collect();
        let zob_unmoved = (0..size).map(|_| rng.next()).collect();
        let zob_passable = (0..size).map(|_| rng.next()).collect();
        let zob_turn = rng.next();
        let mut on_board = Vec::new();
        for r in 0..f.rows {
            for c in 0..f.columns {
                on_board.push(((r + border) * stride + c + border) as usize);
            }
        }
        Ok(Game {
            rows: f.rows,
            cols: f.columns,
            border,
            stride,
            size,
            kinds,
            count_to_draw: f.count_to_draw,
            repeats_to_draw: f.repeats_to_draw,
            no_move_draws: f.no_move_unattacked == "draw",
            tempo: w.tempo.round() as i32,
            royal_pressed: w.royal_pressed.round() as i32,
            zob_piece,
            zob_unmoved,
            zob_passable,
            zob_turn,
            on_board,
        })
    }
    fn sq(&self, r: i32, c: i32) -> usize {
        ((r + self.border) * self.stride + c + self.border) as usize
    }
    fn rc(&self, sq: usize) -> (i32, i32) {
        let sq = sq as i32;
        (sq / self.stride - self.border, sq % self.stride - self.border)
    }
    fn row(&self, sq: usize) -> i32 {
        sq as i32 / self.stride - self.border
    }
    fn far_row(&self, side: usize) -> i32 {
        if side == 0 {
            self.rows - 1
        } else {
            0
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Move {
    from: u16,
    to: u16,
    becomes: u8, // 0 none, else kind + 1
    taken_at: u16,
    partner_from: u16,
    partner_to: u16,
}

impl Move {
    fn null() -> Move {
        Move { from: NONE, to: NONE, becomes: 0, taken_at: NONE, partner_from: NONE, partner_to: NONE }
    }
}

struct Undo {
    m: Move,
    moved: u8,
    captured: u8,
    unmoved_cleared: [u16; 4],
    passable: Option<(u16, u16)>,
    count: i32,
    hash: u64,
}

#[derive(Clone)]
struct Board {
    cells: Vec<u8>,
    unmoved: Vec<bool>,
    turn: usize,
    passable: Option<(u16, u16)>,
    count: i32,
    hash: u64,
    history: Vec<u64>,
}

impl Board {
    fn from(g: &Game, p: &PosD) -> Result<Board, String> {
        let mut cells = vec![OFF; g.size];
        for &sq in &g.on_board {
            cells[sq] = EMPTY;
        }
        for (r, c, side, letter) in &p.cells {
            let k = g.kinds.iter().position(|k| &k.letter == letter).ok_or(format!("unknown kind {letter}"))?;
            cells[g.sq(*r, *c)] = code(k, *side as usize);
        }
        let mut unmoved = vec![false; g.size];
        for (r, c) in &p.unmoved {
            unmoved[g.sq(*r, *c)] = true;
        }
        let passable = p.passable.map(|((a, b), (c, d))| (g.sq(a, b) as u16, g.sq(c, d) as u16));
        let mut b = Board { cells, unmoved, turn: p.turn as usize, passable, count: p.count, hash: 0, history: vec![] };
        b.hash = b.full_hash(g);
        for h in &p.history {
            let earlier = Board::from(g, h)?;
            b.history.push(earlier.hash);
        }
        Ok(b)
    }

    fn full_hash(&self, g: &Game) -> u64 {
        let mut h = 0u64;
        for &sq in &g.on_board {
            let c = self.cells[sq];
            if c != EMPTY {
                h ^= g.zob_piece[c as usize * g.size + sq];
            }
            if self.unmoved[sq] && c != EMPTY {
                h ^= g.zob_unmoved[sq];
            }
        }
        if let Some((p, _)) = self.passable {
            h ^= g.zob_passable[p as usize];
        }
        if self.turn == 1 {
            h ^= g.zob_turn;
        }
        h
    }

    fn attacked(&self, g: &Game, sq: usize, by: usize) -> bool {
        for (ki, k) in g.kinds.iter().enumerate() {
            let want = code(ki, by);
            for s in &k.steps {
                if s.may == 1 {
                    continue;
                }
                for &d in &s.deltas[by] {
                    let mut t = sq as i32 - d;
                    let mut n = 1;
                    loop {
                        let c = self.cells[t as usize];
                        if c == OFF {
                            break;
                        }
                        if c != EMPTY {
                            if c == want && (!s.only_unmoved || self.unmoved[t as usize]) {
                                return true;
                            }
                            break;
                        }
                        if s.rides != 0 && n >= s.rides {
                            break;
                        }
                        t -= d;
                        n += 1;
                    }
                }
            }
        }
        false
    }

    fn royals<'b>(&'b self, g: &'b Game, side: usize) -> impl Iterator<Item = usize> + 'b {
        let kinds = &g.kinds;
        g.on_board.iter().copied().filter(move |&sq| {
            let c = self.cells[sq];
            c != EMPTY && c != OFF && side_of(c) == side && kinds[kind_of(c)].royal
        })
    }

    fn in_check(&self, g: &Game, side: usize) -> bool {
        let royals: Vec<usize> = self.royals(g, side).collect();
        royals.iter().any(|&sq| self.attacked(g, sq, 1 - side))
    }

    fn possible(&self, g: &Game, out: &mut Vec<Move>, only_takes: bool) {
        let side = self.turn;
        let far = g.far_row(side);
        for &sq in &g.on_board {
            let c = self.cells[sq];
            if c == EMPTY || side_of(c) != side {
                continue;
            }
            let k = &g.kinds[kind_of(c)];
            let start = out.len();
            for s in &k.steps {
                if s.only_unmoved && !self.unmoved[sq] {
                    continue;
                }
                if only_takes && s.may == 1 && k.becomes.is_empty() {
                    continue;
                }
                for &d in &s.deltas[side] {
                    let mut t = sq as i32 + d;
                    let mut n = 1;
                    loop {
                        let c2 = self.cells[t as usize];
                        if c2 == OFF {
                            break;
                        }
                        let mut taken = NONE;
                        let mut ok = false;
                        if c2 == EMPTY {
                            if s.may != 2 {
                                ok = !only_takes || (!k.becomes.is_empty() && g.row(t as usize) == far);
                            } else if s.takes_in_passing {
                                if let Some((p, victim)) = self.passable {
                                    if p as i32 == t {
                                        ok = true;
                                        taken = victim;
                                    }
                                }
                            }
                        } else if side_of(c2) != side && s.may != 1 {
                            ok = true;
                            taken = t as u16;
                        }
                        if ok {
                            let base = Move {
                                from: sq as u16,
                                to: t as u16,
                                becomes: 0,
                                taken_at: taken,
                                partner_from: NONE,
                                partner_to: NONE,
                            };
                            if !k.becomes.is_empty() && g.row(t as usize) == far {
                                for &b in &k.becomes {
                                    let m = Move { becomes: b + 1, ..base };
                                    if !out[start..].contains(&m) {
                                        out.push(m);
                                    }
                                }
                            } else if !out[start..].contains(&base) {
                                out.push(base);
                            }
                        }
                        if c2 != EMPTY {
                            break;
                        }
                        if s.rides != 0 && n >= s.rides {
                            break;
                        }
                        t += d;
                        n += 1;
                    }
                }
            }
            if !only_takes {
                if let Some(partner) = k.crosses_with {
                    if self.unmoved[sq] {
                        self.crossings(g, sq, partner as usize, out);
                    }
                }
            }
        }
    }

    fn crossings(&self, g: &Game, sq: usize, partner: usize, out: &mut Vec<Move>) {
        let side = self.turn;
        let (row, col) = g.rc(sq);
        let want = code(partner, side);
        for c in 0..g.cols {
            let p = g.sq(row, c);
            if self.cells[p] != want || !self.unmoved[p] || p == sq {
                continue;
            }
            let way = if c > col { 1 } else { -1 };
            let mut clear = true;
            let mut x = col + way;
            while x != c {
                if self.cells[g.sq(row, x)] != EMPTY {
                    clear = false;
                    break;
                }
                x += way;
            }
            if !clear {
                continue;
            }
            let end_c = col + 2 * way;
            if end_c < 0 || end_c >= g.cols {
                continue;
            }
            let crossed = [sq, g.sq(row, col + way), g.sq(row, end_c)];
            if crossed.iter().any(|&q| self.attacked(g, q, 1 - side)) {
                continue;
            }
            out.push(Move {
                from: sq as u16,
                to: g.sq(row, end_c) as u16,
                becomes: 0,
                taken_at: NONE,
                partner_from: p as u16,
                partner_to: g.sq(row, col + way) as u16,
            });
        }
    }

    fn clear_unmoved(&mut self, g: &Game, sq: usize, undo: &mut Undo, slot: &mut usize) {
        if self.unmoved[sq] {
            self.unmoved[sq] = false;
            if self.cells[sq] != EMPTY {
                self.hash ^= g.zob_unmoved[sq];
            }
            if *slot < 4 {
                undo.unmoved_cleared[*slot] = sq as u16;
                *slot += 1;
            }
        }
    }

    fn make(&mut self, g: &Game, m: Move) -> Undo {
        let from = m.from as usize;
        let to = m.to as usize;
        let moved = self.cells[from];
        let mut undo = Undo {
            m,
            moved,
            captured: EMPTY,
            unmoved_cleared: [NONE; 4],
            passable: self.passable,
            count: self.count,
            hash: self.hash,
        };
        self.history.push(self.hash);
        let mut slot = 0;
        // unmoved marks are hashed only on occupied squares: clear them before the pieces go
        self.clear_unmoved(g, from, &mut undo, &mut slot);
        if m.taken_at != NONE {
            let at = m.taken_at as usize;
            self.clear_unmoved(g, at, &mut undo, &mut slot);
            undo.captured = self.cells[at];
            self.hash ^= g.zob_piece[undo.captured as usize * g.size + at];
            self.cells[at] = EMPTY;
        }
        self.clear_unmoved(g, to, &mut undo, &mut slot);
        self.hash ^= g.zob_piece[moved as usize * g.size + from];
        self.cells[from] = EMPTY;
        let placed = if m.becomes != 0 { code((m.becomes - 1) as usize, side_of(moved)) } else { moved };
        self.cells[to] = placed;
        self.hash ^= g.zob_piece[placed as usize * g.size + to];
        if m.partner_from != NONE {
            let pf = m.partner_from as usize;
            let pt = m.partner_to as usize;
            self.clear_unmoved(g, pf, &mut undo, &mut slot);
            let pc = self.cells[pf];
            self.hash ^= g.zob_piece[pc as usize * g.size + pf];
            self.cells[pf] = EMPTY;
            self.cells[pt] = pc;
            self.hash ^= g.zob_piece[pc as usize * g.size + pt];
        }
        if let Some((p, _)) = self.passable {
            self.hash ^= g.zob_passable[p as usize];
        }
        self.passable = None;
        let k = &g.kinds[kind_of(moved)];
        let (fr, fc) = g.rc(from);
        let (tr, tc) = g.rc(to);
        let distance = (tr - fr).abs().max((tc - fc).abs());
        if distance > 1 && m.partner_from == NONE && k.steps.iter().any(|s| s.leaves_passable) {
            let dr = (tr - fr) / distance;
            let dc = (tc - fc) / distance;
            let p = g.sq(fr + dr, fc + dc);
            self.passable = Some((p as u16, to as u16));
            self.hash ^= g.zob_passable[p];
        }
        self.count = if undo.captured != EMPTY || k.cannot_go_back { 0 } else { self.count + 1 };
        self.turn = 1 - self.turn;
        self.hash ^= g.zob_turn;
        undo
    }

    fn unmake(&mut self, u: Undo) {
        let m = u.m;
        let from = m.from as usize;
        let to = m.to as usize;
        if m.partner_from != NONE {
            let pc = self.cells[m.partner_to as usize];
            self.cells[m.partner_to as usize] = EMPTY;
            self.cells[m.partner_from as usize] = pc;
        }
        self.cells[to] = EMPTY;
        self.cells[from] = u.moved;
        if m.taken_at != NONE {
            self.cells[m.taken_at as usize] = u.captured;
        }
        for &sq in &u.unmoved_cleared {
            if sq != NONE {
                self.unmoved[sq as usize] = true;
            }
        }
        self.passable = u.passable;
        self.count = u.count;
        self.hash = u.hash;
        self.turn = 1 - self.turn;
        self.history.pop();
    }

    fn make_null(&mut self, g: &Game) -> (Option<(u16, u16)>, u64) {
        let saved = (self.passable, self.hash);
        self.history.push(self.hash);
        if let Some((p, _)) = self.passable {
            self.hash ^= g.zob_passable[p as usize];
        }
        self.passable = None;
        self.turn = 1 - self.turn;
        self.hash ^= g.zob_turn;
        saved
    }

    fn unmake_null(&mut self, saved: (Option<(u16, u16)>, u64)) {
        self.passable = saved.0;
        self.hash = saved.1;
        self.turn = 1 - self.turn;
        self.history.pop();
    }

    fn legal(&mut self, g: &Game) -> Vec<Move> {
        let mut out = Vec::new();
        self.possible(g, &mut out, false);
        let side = self.turn;
        out.into_iter()
            .filter(|&m| {
                let u = self.make(g, m);
                let ok = !self.in_check(g, side);
                self.unmake(u);
                ok
            })
            .collect()
    }

    fn repeated(&self, times: usize) -> bool {
        // positions since the last irreversible move, same side to move
        let n = self.history.len();
        let back = (self.count as usize).min(n);
        let mut seen = 0;
        let mut i = 2;
        while i <= back {
            if self.history[n - i] == self.hash {
                seen += 1;
                if seen >= times {
                    return true;
                }
            }
            i += 2;
        }
        false
    }
}

#[derive(Clone, Copy, Default)]
struct Entry {
    key: u64,
    depth: i8,
    flag: u8, // 0 exact, 1 lower, 2 upper
    score: i32,
    best: Option<Move>,
}

struct Searcher<'a> {
    g: &'a Game,
    b: Board,
    tt: Vec<Entry>,
    killers: Vec<[Move; 2]>,
    history: Vec<i32>,
    nodes: u64,
    started: Instant,
    seconds: f64,
    stop: bool,
    max_nodes: u64,
    excluded: Vec<Move>,
    /// How far the best score moved between the last two depths completed: the search's own unsteadiness, under
    /// which two moves' scores do not tell them apart.
    unsteady: i32,
}

impl<'a> Searcher<'a> {
    fn evaluate(&mut self) -> i32 {
        let g = self.g;
        let mut score = [0i32; 2];
        let mut moves = Vec::with_capacity(64);
        for &sq in &g.on_board {
            let c = self.b.cells[sq];
            if c == EMPTY {
                continue;
            }
            let k = &g.kinds[kind_of(c)];
            let s = side_of(c);
            score[s] += k.worth + k.squares[s][sq];
        }
        let any_mobility = g.kinds.iter().any(|k| k.mobility != 0);
        if any_mobility {
            for side in 0..2 {
                let saved = self.b.turn;
                self.b.turn = side;
                moves.clear();
                self.b.possible(g, &mut moves, false);
                self.b.turn = saved;
                for m in &moves {
                    let k = &g.kinds[kind_of(self.b.cells[m.from as usize])];
                    score[side] += k.mobility;
                }
            }
        }
        if g.royal_pressed != 0 {
            for side in 0..2 {
                let royals: Vec<usize> = self.b.royals(g, side).collect();
                for r in royals {
                    for dr in -1..=1 {
                        for dc in -1..=1 {
                            let t = (r as i32 + dr * g.stride + dc) as usize;
                            if self.b.cells[t] != OFF && self.b.attacked(g, t, 1 - side) {
                                score[side] -= g.royal_pressed;
                            }
                        }
                    }
                }
            }
        }
        let me = self.b.turn;
        score[me] - score[1 - me] + g.tempo
    }

    fn worth_of(&self, c: u8) -> i32 {
        if c == EMPTY || c == OFF {
            0
        } else {
            self.g.kinds[kind_of(c)].worth.max(1)
        }
    }

    fn order(&self, moves: &mut [Move], best: Option<Move>, ply: usize) {
        let mut keyed: Vec<(i64, Move)> = moves
            .iter()
            .map(|&m| {
                let key: i64 = if Some(m) == best {
                    1 << 40
                } else if m.taken_at != NONE {
                    let victim = self.worth_of(self.b.cells[m.taken_at as usize]) as i64;
                    let attacker = self.worth_of(self.b.cells[m.from as usize]) as i64;
                    (1 << 30) + victim * 64 - attacker / 16
                } else if m.becomes != 0 {
                    (1 << 30) + self.g.kinds[(m.becomes - 1) as usize].worth as i64 * 64
                } else if ply < MAX_PLY && self.killers[ply][0] == m {
                    1 << 29
                } else if ply < MAX_PLY && self.killers[ply][1] == m {
                    (1 << 29) - 1
                } else {
                    self.history[m.from as usize * self.g.size + m.to as usize] as i64
                };
                (key, m)
            })
            .collect();
        keyed.sort_by(|a, b| b.0.cmp(&a.0));
        for (i, (_, m)) in keyed.into_iter().enumerate() {
            moves[i] = m;
        }
    }

    fn time_up(&mut self) -> bool {
        if self.nodes & 1023 == 0
            && (self.started.elapsed().as_secs_f64() > self.seconds || (self.max_nodes > 0 && self.nodes > self.max_nodes))
        {
            self.stop = true;
        }
        self.stop
    }

    fn tt_index(&self, key: u64) -> usize {
        (key as usize) % self.tt.len()
    }

    fn quiesce(&mut self, mut alpha: i32, beta: i32, ply: usize) -> i32 {
        self.nodes += 1;
        if self.time_up() {
            return 0;
        }
        let side = self.b.turn;
        let checked = self.b.in_check(self.g, side);
        if !checked {
            let stand = self.evaluate();
            if stand >= beta {
                return stand;
            }
            if stand > alpha {
                alpha = stand;
            }
        }
        if ply >= MAX_PLY - 1 {
            return alpha;
        }
        let mut moves = Vec::with_capacity(32);
        self.b.possible(self.g, &mut moves, !checked);
        self.order(&mut moves, None, ply);
        let mut any = false;
        let mut best = if checked { -MATE + ply as i32 } else { alpha };
        for m in moves {
            let u = self.b.make(self.g, m);
            if self.b.in_check(self.g, side) {
                self.b.unmake(u);
                continue;
            }
            any = true;
            let score = -self.quiesce(-beta, -alpha, ply + 1);
            self.b.unmake(u);
            if self.stop {
                return 0;
            }
            if score > best {
                best = score;
            }
            if score > alpha {
                alpha = score;
                if alpha >= beta {
                    break;
                }
            }
        }
        if checked && !any {
            return -MATE + ply as i32;
        }
        best.max(if checked { best } else { alpha }).min(INF)
    }

    fn has_pieces_beyond_irreversible(&self, side: usize) -> bool {
        self.g.on_board.iter().any(|&sq| {
            let c = self.b.cells[sq];
            c != EMPTY && side_of(c) == side && {
                let k = &self.g.kinds[kind_of(c)];
                !k.royal && !k.cannot_go_back
            }
        })
    }

    fn search(&mut self, mut depth: i32, mut alpha: i32, beta: i32, ply: usize, null_ok: bool) -> i32 {
        self.nodes += 1;
        if self.time_up() {
            return 0;
        }
        let g = self.g;
        if ply > 0 {
            if self.b.repeated(1) || (g.count_to_draw > 0 && self.b.count >= g.count_to_draw) {
                return 0;
            }
        }
        if ply >= MAX_PLY - 1 {
            return self.evaluate();
        }
        let side = self.b.turn;
        let checked = self.b.in_check(g, side);
        if checked {
            depth += 1;
        }
        if depth <= 0 {
            return self.quiesce(alpha, beta, ply);
        }
        let key = self.b.hash;
        let idx = self.tt_index(key);
        let entry = self.tt[idx];
        let mut tt_move = None;
        if entry.key == key {
            tt_move = entry.best;
            if ply > 0 && entry.depth as i32 >= depth {
                let mut s = entry.score;
                if s > MATE - 1000 {
                    s -= ply as i32;
                } else if s < -MATE + 1000 {
                    s += ply as i32;
                }
                match entry.flag {
                    0 => return s,
                    1 if s >= beta => return s,
                    2 if s <= alpha => return s,
                    _ => {}
                }
            }
        }
        let pv_node = beta - alpha > 1;
        if !checked && null_ok && !pv_node && depth >= 3 && self.has_pieces_beyond_irreversible(side) {
            let stand = self.evaluate();
            if stand >= beta {
                let r = 2 + depth / 6;
                let saved = self.b.make_null(g);
                let score = -self.search(depth - 1 - r, -beta, -beta + 1, ply + 1, false);
                self.b.unmake_null(saved);
                if self.stop {
                    return 0;
                }
                if score >= beta && score < MATE - 1000 {
                    return beta;
                }
            }
        }
        let mut moves = Vec::with_capacity(48);
        self.b.possible(g, &mut moves, false);
        self.order(&mut moves, tt_move, ply);
        let original_alpha = alpha;
        let mut best_score = -INF;
        let mut best_move = None;
        let mut legal = 0;
        for m in moves {
            if ply == 0 && self.excluded.contains(&m) {
                continue;
            }
            let quiet = m.taken_at == NONE && m.becomes == 0;
            let u = self.b.make(g, m);
            if self.b.in_check(g, side) {
                self.b.unmake(u);
                continue;
            }
            legal += 1;
            let gives_check = self.b.in_check(g, 1 - side);
            let mut score;
            if legal == 1 {
                score = -self.search(depth - 1, -beta, -alpha, ply + 1, true);
            } else {
                let mut reduction = 0;
                if depth >= 3 && legal > 3 && quiet && !checked && !gives_check {
                    reduction = 1 + (legal as i32 > 8) as i32 + (depth > 6) as i32;
                }
                score = -self.search(depth - 1 - reduction, -alpha - 1, -alpha, ply + 1, true);
                if score > alpha && reduction > 0 {
                    score = -self.search(depth - 1, -alpha - 1, -alpha, ply + 1, true);
                }
                if score > alpha && score < beta {
                    score = -self.search(depth - 1, -beta, -alpha, ply + 1, true);
                }
            }
            self.b.unmake(u);
            if self.stop {
                return 0;
            }
            if score > best_score {
                best_score = score;
                best_move = Some(m);
            }
            if score > alpha {
                alpha = score;
                if alpha >= beta {
                    if quiet && ply < MAX_PLY {
                        if self.killers[ply][0] != m {
                            self.killers[ply][1] = self.killers[ply][0];
                            self.killers[ply][0] = m;
                        }
                        let h = &mut self.history[m.from as usize * g.size + m.to as usize];
                        *h += depth * depth;
                        if *h > 1 << 20 {
                            for v in self.history.iter_mut() {
                                *v /= 2;
                            }
                        }
                    }
                    break;
                }
            }
        }
        if legal == 0 {
            return if checked || !g.no_move_draws { -MATE + ply as i32 } else { 0 };
        }
        let flag = if best_score <= original_alpha {
            2
        } else if best_score >= beta {
            1
        } else {
            0
        };
        let mut stored = best_score;
        if stored > MATE - 1000 {
            stored += ply as i32;
        } else if stored < -MATE + 1000 {
            stored -= ply as i32;
        }
        let replace = entry.key != key || depth >= entry.depth as i32 || flag == 0;
        if replace {
            self.tt[idx] = Entry { key, depth: depth.min(127) as i8, flag, score: stored, best: best_move };
        }
        best_score
    }

    fn line(&mut self, first: Move, most: usize) -> Vec<Move> {
        let mut line = vec![first];
        let mut undos = vec![self.b.make(self.g, first)];
        let mut seen = vec![self.b.hash];
        while line.len() < most {
            let e = self.tt[self.tt_index(self.b.hash)];
            if e.key != self.b.hash {
                break;
            }
            let Some(m) = e.best else { break };
            let legal = self.b.legal(self.g);
            if !legal.contains(&m) {
                break;
            }
            undos.push(self.b.make(self.g, m));
            line.push(m);
            if seen.contains(&self.b.hash) {
                break;
            }
            seen.push(self.b.hash);
        }
        while let Some(u) = undos.pop() {
            self.b.unmake(u);
        }
        line
    }
}

impl<'a> Searcher<'a> {
    /// Iterative deepening at the root over ``legal``: the best move, its score, and the depth completed.
    fn best(&mut self, deepest: i32, legal: &[Move]) -> (Option<Move>, i32, i32) {
        if legal.is_empty() {
            return (None, 0, 0);
        }
        let mut best = legal[0];
        let mut best_score = 0;
        let mut done = 0;
        let mut window = (-INF, INF);
        for depth in 1..=deepest {
            let mut score = self.search(depth, window.0, window.1, 0, false);
            if !self.stop && (score <= window.0 || score >= window.1) {
                score = self.search(depth, -INF, INF, 0, false);
            }
            if self.stop {
                break;
            }
            let e = self.tt[self.tt_index(self.b.hash)];
            if e.key == self.b.hash {
                if let Some(m) = e.best {
                    if legal.contains(&m) {
                        best = m;
                    }
                }
            }
            if done > 0 {
                self.unsteady = (score - best_score).abs();
            }
            best_score = score;
            done = depth;
            window = if score.abs() < MATE - 1000 { (score - 50, score + 50) } else { (-INF, INF) };
            if score.abs() > MATE - 1000 || legal.len() == 1 {
                break;
            }
        }
        (Some(best), best_score, done)
    }

    /// What a position is made of, from the first side's view, for fitting weights to outcomes: per kind the count
    /// held more than the other side, per kind the moves it has more, the squares next to a royal piece the other
    /// side attacks fewer, whose turn it is, and per kind and square (each side's own frame) the pieces standing there.
    fn features(&mut self) -> Vec<f32> {
        let g = self.g;
        let k = g.kinds.len();
        let squares = (g.rows * g.cols) as usize;
        let mut f = vec![0f32; 2 * k + 2 + k * squares];
        for &sq in &g.on_board {
            let c = self.b.cells[sq];
            if c == EMPTY {
                continue;
            }
            let side = side_of(c);
            let sign = if side == 0 { 1.0 } else { -1.0 };
            let kind = kind_of(c);
            f[kind] += sign;
            let (r, col) = g.rc(sq);
            let own_row = if side == 0 { r } else { g.rows - 1 - r };
            f[2 * k + 2 + kind * squares + (own_row * g.cols + col) as usize] += sign;
        }
        let saved = self.b.turn;
        let mut moves = Vec::with_capacity(64);
        for side in 0..2 {
            self.b.turn = side;
            moves.clear();
            self.b.possible(g, &mut moves, false);
            let sign = if side == 0 { 1.0 } else { -1.0 };
            for m in &moves {
                f[k + kind_of(self.b.cells[m.from as usize])] += sign;
            }
        }
        self.b.turn = saved;
        for side in 0..2 {
            let sign = if side == 0 { 1.0 } else { -1.0 };
            let royals: Vec<usize> = self.b.royals(g, side).collect();
            for r in royals {
                for dr in -1..=1 {
                    for dc in -1..=1 {
                        let t = (r as i32 + dr * g.stride + dc) as usize;
                        if self.b.cells[t] != OFF && self.b.attacked(g, t, 1 - side) {
                            f[2 * k] -= sign;
                        }
                    }
                }
            }
        }
        f[2 * k + 1] = if self.b.turn == 0 { 1.0 } else { -1.0 };
        f
    }
}

fn move_out(g: &Game, m: &Move) -> (i32, i32, i32, i32, String) {
    let (fr, fc) = g.rc(m.from as usize);
    let (tr, tc) = g.rc(m.to as usize);
    let becomes = if m.becomes != 0 { g.kinds[(m.becomes - 1) as usize].letter.clone() } else { String::new() };
    (fr, fc, tr, tc, becomes)
}

fn parse(description: &str, position: &str, weights: &str) -> PyResult<(Game, Board)> {
    let f: FamilyD = serde_json::from_str(description).map_err(|e| PyValueError::new_err(format!("description: {e}")))?;
    let w: WeightsD = if weights.trim().is_empty() {
        WeightsD::default()
    } else {
        serde_json::from_str(weights).map_err(|e| PyValueError::new_err(format!("weights: {e}")))?
    };
    let p: PosD = serde_json::from_str(position).map_err(|e| PyValueError::new_err(format!("position: {e}")))?;
    let g = Game::new(&f, &w).map_err(PyValueError::new_err)?;
    let b = Board::from(&g, &p).map_err(PyValueError::new_err)?;
    Ok((g, b))
}

fn perft_inner(g: &Game, b: &mut Board, depth: u32) -> u64 {
    let moves = b.legal(g);
    if depth == 1 {
        return moves.len() as u64;
    }
    let mut n = 0;
    for m in moves {
        let u = b.make(g, m);
        n += perft_inner(g, b, depth - 1);
        b.unmake(u);
    }
    n
}

/// The number of move sequences of ``depth`` from the position: a check of the description and the move maker against
/// published counts.
#[pyfunction]
fn perft(description: &str, position: &str, depth: u32) -> PyResult<u64> {
    let (g, mut b) = parse(description, position, "")?;
    Ok(if depth == 0 { 1 } else { perft_inner(&g, &mut b, depth) })
}

/// The legal moves of the position, each as (from row, from column, to row, to column, kind it becomes).
#[pyfunction]
fn legal_moves(description: &str, position: &str) -> PyResult<Vec<(i32, i32, i32, i32, String)>> {
    let (g, mut b) = parse(description, position, "")?;
    Ok(b.legal(&g).iter().map(|m| move_out(&g, m)).collect())
}

/// The value of the position for the side to move, by the weights, without search.
#[pyfunction]
fn evaluate(description: &str, position: &str, weights: &str) -> PyResult<i32> {
    let (g, b) = parse(description, position, weights)?;
    let mut s = Searcher {
        g: &g,
        b,
        tt: vec![Entry::default(); 1],
        killers: vec![[Move::null(); 2]; MAX_PLY],
        history: vec![0; g.size * g.size],
        nodes: 0,
        started: Instant::now(),
        seconds: 1e9,
        stop: false,
        max_nodes: 0,
        excluded: vec![],
        unsteady: 0,
    };
    Ok(s.evaluate())
}

/// The best move found in ``seconds`` (or ``nodes``, when given): a dict with the move, its score for the side to move,
/// the depth completed, the nodes looked at, and the line expected after it.
#[pyfunction]
#[pyo3(signature = (description, position, weights, seconds, deepest=64, hash_mb=64, nodes=0, excluded=vec![]))]
#[allow(clippy::too_many_arguments)]
fn search(
    py: Python<'_>,
    description: &str,
    position: &str,
    weights: &str,
    seconds: f64,
    deepest: i32,
    hash_mb: usize,
    nodes: u64,
    excluded: Vec<(i32, i32, i32, i32, String)>,
) -> PyResult<PyObject> {
    let (g, b) = parse(description, position, weights)?;
    let result = py.allow_threads(|| {
        let entries = (hash_mb.max(1) * 1024 * 1024) / std::mem::size_of::<Entry>();
        let mut s = Searcher {
            g: &g,
            b,
            tt: vec![Entry::default(); entries.max(1024)],
            killers: vec![[Move::null(); 2]; MAX_PLY],
            history: vec![0; g.size * g.size],
            nodes: 0,
            started: Instant::now(),
            seconds,
            stop: false,
            max_nodes: nodes,
            excluded: vec![],
            unsteady: 0,
        };
        let all = s.b.legal(&g);
        let legal: Vec<Move> = all.into_iter().filter(|m| !excluded.contains(&move_out(&g, m))).collect();
        s.excluded = s.b.legal(&g).into_iter().filter(|m| !legal.contains(m)).collect();
        if legal.is_empty() {
            return (None, 0, 0, 0u64, vec![], 0, 0);
        }
        let (found, best_score, done) = s.best(deepest, &legal);
        let best = found.unwrap_or(legal[0]);
        let line = s.line(best, 12);
        let n = s.nodes;
        let ms = (s.started.elapsed().as_secs_f64() * 1000.0) as u64;
        (Some(best), best_score, done, n, line, ms, s.unsteady)
    });
    let d = PyDict::new_bound(py);
    let (best, score, depth, n, line, ms, unsteady) = result;
    match best {
        Some(m) => d.set_item("move", move_out(&g, &m))?,
        None => d.set_item("move", py.None())?,
    }
    d.set_item("score", score)?;
    d.set_item("depth", depth)?;
    d.set_item("nodes", n)?;
    d.set_item("ms", ms)?;
    d.set_item("mate", score.abs() > MATE - 1000)?;
    d.set_item("unsteady", unsteady)?;
    let line_out: Vec<(i32, i32, i32, i32, String)> = line.iter().map(|m| move_out(&g, m)).collect();
    d.set_item("line", line_out)?;
    Ok(d.into())
}

/// Games played in the model between two sets of weights, colours swapped each game, the first ``random_plies``
/// of each chosen at random for variety: the points the first weights won, and samples of quiet positions (features,
/// and the game's result for the first side: 1, 0.5, 0) for fitting weights to what happened. With them, which game
/// each sample came from, and each game's count of every kind where it stopped (first side less second): a game that
/// stopped without a result can still be judged by how it stood.
#[pyfunction]
#[pyo3(signature = (description, start, weights_a, weights_b, games, nodes_per_move, random_plies=4, seed=1, max_plies=300, sample_every=2))]
#[allow(clippy::too_many_arguments)]
fn self_play(
    py: Python<'_>,
    description: &str,
    start: &str,
    weights_a: &str,
    weights_b: &str,
    games: u32,
    nodes_per_move: u64,
    random_plies: u32,
    seed: u64,
    max_plies: u32,
    sample_every: u32,
) -> PyResult<PyObject> {
    let (ga, b0) = parse(description, start, weights_a)?;
    let (gb, _) = parse(description, start, weights_b)?;
    let (points, results, samples, lengths, sample_games, final_counts) = py.allow_threads(|| {
        let mut rng = Rng(seed.wrapping_mul(0x2545F4914F6CDD1D) | 1);
        let entries = (4 * 1024 * 1024) / std::mem::size_of::<Entry>();
        let mut points = 0.0f64;
        let mut results: Vec<f64> = Vec::new();
        let mut lengths: Vec<u32> = Vec::new();
        let mut samples: Vec<(Vec<f32>, f64)> = Vec::new();
        let mut sample_games: Vec<u32> = Vec::new();
        let mut final_counts: Vec<Vec<f32>> = Vec::new();
        for game in 0..games {
            let a_is_first = game % 2 == 0;
            let mut sa = Searcher { g: &ga, b: b0.clone(), tt: vec![Entry::default(); entries], killers: vec![[Move::null(); 2]; MAX_PLY], history: vec![0; ga.size * ga.size], nodes: 0, started: Instant::now(), seconds: 1e9, stop: false, max_nodes: nodes_per_move, excluded: vec![], unsteady: 0 };
            let mut sb = Searcher { g: &gb, b: b0.clone(), tt: vec![Entry::default(); entries], killers: vec![[Move::null(); 2]; MAX_PLY], history: vec![0; gb.size * gb.size], nodes: 0, started: Instant::now(), seconds: 1e9, stop: false, max_nodes: nodes_per_move, excluded: vec![], unsteady: 0 };
            let mut board = b0.clone();
            let mut taken: Vec<Vec<f32>> = Vec::new();
            let mut result_first = 0.5;
            let mut last_took = false;
            let mut plies = 0u32;
            for ply in 0..max_plies {
                plies = ply;
                let legal = board.legal(&ga);
                if legal.is_empty() {
                    let mover = board.turn;
                    let lost = board.in_check(&ga, mover) || !ga.no_move_draws;
                    result_first = if lost { if mover == 0 { 0.0 } else { 1.0 } } else { 0.5 };
                    break;
                }
                let repeats = if ga.repeats_to_draw > 1 { (ga.repeats_to_draw - 1) as usize } else { 2 };
                if board.repeated(repeats) || (ga.count_to_draw > 0 && board.count >= ga.count_to_draw) {
                    break;
                }
                let first_moves = (board.turn == 0) == a_is_first;
                let chosen = if ply < random_plies {
                    legal[(rng.next() % legal.len() as u64) as usize]
                } else {
                    let s = if first_moves { &mut sa } else { &mut sb };
                    s.b = board.clone();
                    s.nodes = 0;
                    s.stop = false;
                    s.started = Instant::now();
                    s.best(64, &legal).0.unwrap_or(legal[0])
                };
                if ply >= random_plies && sample_every > 0 && ply % sample_every == 0 && !last_took
                    && !board.in_check(&ga, board.turn)
                {
                    sa.b = board.clone();
                    taken.push(sa.features());
                }
                last_took = chosen.taken_at != NONE;
                board.make(&ga, chosen);
            }
            let a_points = if a_is_first { result_first } else { 1.0 - result_first };
            points += a_points;
            results.push(a_points);
            lengths.push(plies);
            for f in taken {
                samples.push((f, result_first));
                sample_games.push(game);
            }
            sa.b = board.clone();
            let k = ga.kinds.len();
            final_counts.push(sa.features()[..k].to_vec());
        }
        (points, results, samples, lengths, sample_games, final_counts)
    });
    let d = PyDict::new_bound(py);
    d.set_item("points", points)?;
    d.set_item("results", results)?;
    d.set_item("plies", lengths)?;
    d.set_item("samples", samples)?;
    d.set_item("sample_games", sample_games)?;
    d.set_item("final_counts", final_counts)?;
    Ok(d.into())
}

#[pymodule]
fn aura_board_search(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(perft, m)?)?;
    m.add_function(wrap_pyfunction!(legal_moves, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate, m)?)?;
    m.add_function(wrap_pyfunction!(search, m)?)?;
    m.add_function(wrap_pyfunction!(self_play, m)?)?;
    Ok(())
}
