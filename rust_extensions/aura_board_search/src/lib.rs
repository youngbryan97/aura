//! Search for games of pieces that move by their kind, read from a description.
//!
//! The game comes in as data (core/agency/pieces_by_their_kind.py): a board's size, each kind as steps (leaps and
//! rides, steps that only move or only take, steps only an unmoved piece makes, passing), royal kinds, kinds that
//! become others on the far row, a royal that crosses with a partner, and the draw counts. Nothing here knows a game.
//! What a kind and a square are worth comes in as weights measured by playing; the search is alpha-beta with the
//! general refinements of game-tree search: iterative deepening, a transposition table, quiescence on takes, null
//! moves, late-move reductions, killer and history ordering, and margins: a line whose verdict cannot change by as
//! much as a quiet move has moved her judgement, for each ply left, is not looked at further.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

const EMPTY: u8 = 0;
/// What lies beyond a search's horizon, counted per side from the rules alone: quantities that change little in a move
/// and decide games long after, so no search's end sees what they come to. In order: how far each piece that only goes
/// forward and becomes another kind has come with no such piece of the other side ahead of it to stop it (each counted
/// as the square of the share of the way it has come); such pieces with no quiet move forward; pieces held by another of
/// their side; squares in the other side's half that the side reaches and the other does not; and pieces of the side
/// beside its royal one.
const BEYOND: usize = 5;
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
    #[serde(default)]
    takes_again: bool,
    #[serde(default)]
    wins_arriving: bool,
}

#[derive(Deserialize)]
struct FamilyD {
    rows: i32,
    columns: i32,
    kinds: Vec<KindD>,
    count_to_draw: i32,
    repeats_to_draw: i32,
    no_move_unattacked: String,
    #[serde(default)]
    takes_first: bool,
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
    /// Per kind, against its side, for each piece of it the other side attacks and its own side does not defend:
    /// what she cannot afford to lose (core/cognition/what_she_cannot_afford_to_lose.py).
    #[serde(default)]
    exposed: HashMap<String, f64>,
    /// A second judgement, learned from what her longer looks found positions worth: added to the first.
    #[serde(default)]
    learned: Option<LearnedD>,
    /// Per quantity of what lies beyond the horizon, in BEYOND's order: its worth to the side that has more of it.
    #[serde(default)]
    beyond: Vec<f64>,
}

/// One hidden layer over which kind stands where, from the side to move: per kind, for the mover's pieces and the
/// other side's, a row of weights into the hidden layer for each square in the mover's own frame.
#[derive(Deserialize, Clone)]
struct LearnedD {
    hidden: usize,
    #[serde(default)]
    own: HashMap<String, Vec<Vec<f32>>>,
    #[serde(default)]
    theirs: HashMap<String, Vec<Vec<f32>>>,
    first_bias: Vec<f32>,
    out: Vec<f32>,
    out_bias: f32,
}

/// The learned judgement as the search reads it: weights by (kind, the mover's or not, square) into the hidden layer.
struct Learned {
    hidden: usize,
    first: Vec<f32>,
    first_bias: Vec<f32>,
    out: Vec<f32>,
    out_bias: f32,
}

/// The most hidden units a learned judgement may have.
const HIDDEN_AT_MOST: usize = 256;

#[derive(Clone)]
struct Step {
    deltas: [Vec<i32>; 2],
    /// Per side, each offset's one square along its line, and how many such squares the offset is.
    units: [Vec<i32>; 2],
    lengths: Vec<i32>,
    rides: i32,
    may: u8, // 0 either, 1 only moves, 2 only takes, 3 takes over: beyond one piece of the other side, which it takes,
    // 4 swaps: onto a square of its own side's piece, which goes to the square it left
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
    /// Having taken over, it takes over again from where it landed while it can, in the same move.
    takes_again: bool,
    /// It has a step that takes over.
    takes_over: bool,
    /// Every way it goes is forward, and it becomes another kind: how far it has come, and what stands ahead of it,
    /// matter long before it gets there.
    runner: bool,
    /// A piece of it on its side's far row has won the game for its side.
    wins_arriving: bool,
    worth: i32,
    mobility: i32,
    squares: [Vec<i32>; 2],
    exposed: i32,
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
    /// A side that can take must.
    takes_first: bool,
    /// Some kind wins by arriving on its far row: each position is asked whether the side that just moved has.
    arrival_wins: bool,
    tempo: i32,
    royal_pressed: i32,
    /// What lies beyond the horizon is worth, per quantity, in BEYOND's order; all zero where it is not judged.
    beyond: [f32; BEYOND],
    /// The most one quiet move moved her judgement at the position a search began from, the turn's own worth left
    /// aside: the unit the search's margins are counted in. Zero where no quiet move moved it, and then there are no
    /// margins. Measured at each search's start, because how far a move moves her judgement is a property of the
    /// weights she judges by: 94 on her offline weights of 22 November 2026, 357 on her live ones, which count each
    /// square a piece reaches nearly half a pawn.
    margin: AtomicI32,
    learned: Option<Learned>,
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
                let mut units: [Vec<i32>; 2] = [Vec::new(), Vec::new()];
                let mut lengths = Vec::new();
                for (fw, rt) in &s.offsets {
                    let mut a = fw.abs();
                    let mut b = rt.abs();
                    while b != 0 {
                        let t = a % b;
                        a = b;
                        b = t;
                    }
                    lengths.push(a.max(1));
                }
                for side in 0..2 {
                    let ahead = if side == 0 { 1 } else { -1 };
                    for (i, (fw, rt)) in s.offsets.iter().enumerate() {
                        deltas[side].push(fw * ahead * stride + rt * ahead);
                        units[side].push((fw / lengths[i]) * ahead * stride + (rt / lengths[i]) * ahead);
                    }
                }
                let may = match s.may.as_str() {
                    "moves" => 1,
                    "takes" => 2,
                    "takes over" => 3,
                    "swaps" => 4,
                    _ => 0,
                };
                steps.push(Step {
                    deltas,
                    units,
                    lengths,
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
            let exposed = (w.exposed.get(&k.letter).copied().unwrap_or(0.0)).round() as i32;
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
            let takes_over = steps.iter().any(|s| s.may == 3);
            let runner = !becomes.is_empty() && !k.steps.is_empty()
                && k.steps.iter().all(|s| s.offsets.iter().all(|(fw, _rt)| *fw > 0));
            kinds.push(Kind {
                letter: k.letter.clone(),
                steps,
                royal: k.royal,
                becomes,
                cannot_go_back: k.cannot_go_back,
                crosses_with,
                takes_again: k.takes_again,
                takes_over,
                runner,
                wins_arriving: k.wins_arriving,
                worth,
                mobility,
                squares,
                exposed,
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
        let squares = (f.rows * f.columns) as usize;
        let learned = match &w.learned {
            None => None,
            Some(l) => {
                if l.hidden == 0 || l.hidden > HIDDEN_AT_MOST || l.first_bias.len() != l.hidden || l.out.len() != l.hidden {
                    return Err("learned: hidden layer sizes disagree".into());
                }
                let mut first = vec![0f32; f.kinds.len() * 2 * squares * l.hidden];
                for (ki, k) in f.kinds.iter().enumerate() {
                    for (rel, rows) in [(0usize, l.own.get(&k.letter)), (1usize, l.theirs.get(&k.letter))] {
                        let Some(rows) = rows else { continue };
                        if rows.len() != squares {
                            return Err(format!("learned: {} has {} squares, the board {}", k.letter, rows.len(), squares));
                        }
                        for (sq, row) in rows.iter().enumerate() {
                            if row.len() != l.hidden {
                                return Err("learned: a square's row is not the hidden layer's size".into());
                            }
                            let at = ((ki * 2 + rel) * squares + sq) * l.hidden;
                            first[at..at + l.hidden].copy_from_slice(row);
                        }
                    }
                }
                Some(Learned { hidden: l.hidden, first, first_bias: l.first_bias.clone(), out: l.out.clone(),
                               out_bias: l.out_bias })
            }
        };
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
            takes_first: f.takes_first,
            arrival_wins: f.kinds.iter().any(|k| k.wins_arriving),
            tempo: w.tempo.round() as i32,
            royal_pressed: w.royal_pressed.round() as i32,
            beyond: {
                let mut beyond = [0f32; BEYOND];
                for (i, v) in w.beyond.iter().take(BEYOND).enumerate() {
                    beyond[i] = *v as f32;
                }
                beyond
            },
            margin: AtomicI32::new(0),
            learned,
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

/// The most pieces one move may take after its first, by taking over again: a move that would take more is not made.
const MORE: usize = 11;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Move {
    from: u16,
    to: u16,
    becomes: u8, // 0 none, else kind + 1
    taken_at: u16,
    partner_from: u16,
    partner_to: u16,
    /// The pieces a move that took again took after ``taken_at``, in order of square, so one set taken is one move.
    more: [u16; MORE],
    n_more: u8,
}

impl Move {
    fn null() -> Move {
        Move { from: NONE, to: NONE, becomes: 0, taken_at: NONE, partner_from: NONE, partner_to: NONE,
               more: [NONE; MORE], n_more: 0 }
    }
}

struct Undo {
    m: Move,
    moved: u8,
    captured: u8,
    captured_more: [u8; MORE],
    unmoved_cleared: [u16; 4 + MORE],
    passable: Option<(u16, u16)>,
    count: i32,
    hash: u64,
}

/// What one walk of each side's reach finds, kept per thread so a position's judgement allocates nothing.
#[derive(Default)]
struct Reach {
    /// Per side and square, how many of the side's pieces could take there: its own pieces' squares included, which
    /// is what defends them.
    attacks: [Vec<u8>; 2],
    /// Per side and kind, the moves its pieces have.
    mobility: [Vec<i32>; 2],
    /// Squares one piece has already counted a move to, marked with that piece's stamp.
    seen: Vec<u32>,
    stamp: u32,
}

thread_local! {
    static REACH: std::cell::RefCell<Reach> = std::cell::RefCell::new(Reach::default());
}

impl Reach {
    fn walked(&mut self, g: &Game, b: &Board) {
        for side in 0..2 {
            self.attacks[side].clear();
            self.attacks[side].resize(g.size, 0);
            self.mobility[side].clear();
            self.mobility[side].resize(g.kinds.len(), 0);
        }
        if self.seen.len() != g.size {
            self.seen = vec![0; g.size];
            self.stamp = 0;
        }
        for side in 0..2 {
            b.reach(g, side, &mut self.attacks[side], &mut self.mobility[side], &mut self.seen, &mut self.stamp);
        }
    }
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
            if k.takes_over && self.taken_over(sq, by, k, want) {
                return true;
            }
            for s in &k.steps {
                if s.may == 1 || s.may == 3 || s.may == 4 {
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

    /// Whether a piece ``want`` of ``by`` could take over the piece on ``sq``: from a square behind it on a line of one
    /// of its take overs, the way clear, onto an empty square beyond.
    fn taken_over(&self, sq: usize, by: usize, k: &Kind, want: u8) -> bool {
        for s in &k.steps {
            if s.may != 3 {
                continue;
            }
            for (i, &d) in s.deltas[by].iter().enumerate() {
                let u = s.units[by][i];
                let n = s.lengths[i];
                if s.rides == 1 {
                    for j in 1..n {
                        let a = sq as i32 - u * j;
                        if a < 0 || a as usize >= self.cells.len() || self.cells[a as usize] != want
                            || (s.only_unmoved && !self.unmoved[a as usize]) {
                            continue;
                        }
                        let end = a + d;
                        if end < 0 || end as usize >= self.cells.len() || self.cells[end as usize] != EMPTY {
                            continue;
                        }
                        if (1..n).all(|x| x == j || self.cells[(a + u * x) as usize] == EMPTY) {
                            return true;
                        }
                    }
                    continue;
                }
                let beyond = sq as i32 + u;
                if beyond < 0 || beyond as usize >= self.cells.len() || self.cells[beyond as usize] != EMPTY {
                    continue;
                }
                let mut t = sq as i32 - u;
                let mut times = 1;
                while t >= 0 && (t as usize) < self.cells.len() {
                    let c = self.cells[t as usize];
                    if c == OFF {
                        break;
                    }
                    if c != EMPTY {
                        if c == want && (s.rides == 0 || times < s.rides) && (!s.only_unmoved || self.unmoved[t as usize]) {
                            return true;
                        }
                        break;
                    }
                    t -= u;
                    times += 1;
                }
            }
        }
        false
    }

    /// One walk of the reach of every piece of ``side``: per kind the moves its pieces have, as ``possible`` counts
    /// them (a promotion once for each kind it may become; crossings left out), and per square how many of the side's
    /// pieces could take there. What ``attacked`` answers for one square, this answers for all of them at once.
    fn reach(&self, g: &Game, side: usize, attacks: &mut [u8], mobility: &mut [i32], seen: &mut [u32], stamp: &mut u32) {
        let far = g.far_row(side);
        for &sq in &g.on_board {
            let c = self.cells[sq];
            if c == EMPTY || side_of(c) != side {
                continue;
            }
            let ki = kind_of(c);
            let k = &g.kinds[ki];
            *stamp = stamp.wrapping_add(1);
            if *stamp == 0 {
                seen.fill(0);
                *stamp = 1;
            }
            if k.takes_over {
                let mut hops = Vec::new();
                self.hops(sq, sq, side, k, &[], &mut hops);
                for (end, victim) in hops {
                    attacks[victim] = attacks[victim].saturating_add(1);
                    if seen[end] != *stamp {
                        seen[end] = *stamp;
                        mobility[ki] += 1;
                    }
                }
            }
            for s in &k.steps {
                if (s.only_unmoved && !self.unmoved[sq]) || s.may == 3 || s.may == 4 {
                    continue;
                }
                for &d in &s.deltas[side] {
                    let mut t = sq as i32 + d;
                    let mut n = 1;
                    loop {
                        let tu = t as usize;
                        let c2 = self.cells[tu];
                        if c2 == OFF {
                            break;
                        }
                        if s.may != 1 {
                            attacks[tu] = attacks[tu].saturating_add(1);
                        }
                        let movable = if c2 == EMPTY {
                            s.may != 2
                                || (s.takes_in_passing && self.passable.is_some_and(|(p, _)| p as usize == tu))
                        } else {
                            side_of(c2) != side && s.may != 1
                        };
                        if movable && seen[tu] != *stamp {
                            seen[tu] = *stamp;
                            mobility[ki] += if !k.becomes.is_empty() && g.row(tu) == far { k.becomes.len() as i32 } else { 1 };
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
        }
    }

    fn royals<'b>(&'b self, g: &'b Game, side: usize) -> impl Iterator<Item = usize> + 'b {
        let kinds = &g.kinds;
        g.on_board.iter().copied().filter(move |&sq| {
            let c = self.cells[sq];
            c != EMPTY && c != OFF && side_of(c) == side && kinds[kind_of(c)].royal
        })
    }

    /// Whether a piece of ``side`` whose kind wins by arriving stands on that side's far row: only that row is looked at.
    fn arrived(&self, g: &Game, side: usize) -> bool {
        let row = g.far_row(side);
        (0..g.cols).any(|c| {
            let cell = self.cells[g.sq(row, c)];
            cell != EMPTY && cell != OFF && side_of(cell) == side && g.kinds[kind_of(cell)].wins_arriving
        })
    }

    fn in_check(&self, g: &Game, side: usize) -> bool {
        let royals: Vec<usize> = self.royals(g, side).collect();
        royals.iter().any(|&sq| self.attacked(g, sq, 1 - side))
    }

    fn possible(&self, g: &Game, out: &mut Vec<Move>, only_takes: bool) {
        let side = self.turn;
        let far = g.far_row(side);
        let begin = out.len();
        for &sq in &g.on_board {
            let c = self.cells[sq];
            if c == EMPTY || side_of(c) != side {
                continue;
            }
            let k = &g.kinds[kind_of(c)];
            let start = out.len();
            for s in &k.steps {
                if (s.only_unmoved && !self.unmoved[sq]) || s.may == 3 {
                    continue;
                }
                if s.may == 4 {
                    // Onto the first piece along the way, where it is one of its own side: the two change places.
                    if !only_takes {
                        for &d in &s.deltas[side] {
                            let mut t = sq as i32 + d;
                            let mut n = 1;
                            loop {
                                let c2 = self.cells[t as usize];
                                if c2 == OFF {
                                    break;
                                }
                                if c2 != EMPTY {
                                    if side_of(c2) == side {
                                        let m = Move { from: sq as u16, to: t as u16, partner_from: t as u16,
                                                       partner_to: sq as u16, ..Move::null() };
                                        if !out[start..].contains(&m) {
                                            out.push(m);
                                        }
                                    }
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
                            let base = Move { from: sq as u16, to: t as u16, taken_at: taken, ..Move::null() };
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
            if k.takes_over {
                let mut taken: Vec<u16> = Vec::with_capacity(MORE + 1);
                let from_here = out.len();
                self.take_over_from(g, sq, sq, side, k, far, &mut taken, out, from_here);
            }
        }
        if g.takes_first && out[begin..].iter().any(|m| m.taken_at != NONE) {
            let takes: Vec<Move> = out.drain(begin..).filter(|m| m.taken_at != NONE).collect();
            out.extend(takes);
        }
    }

    /// Each take over a piece of ``k`` could make from ``at``, having begun its move on ``origin`` (empty now) and taken
    /// ``taken`` (standing still until the move ends): where it lands, and what it takes.
    fn hops(&self, origin: usize, at: usize, side: usize, k: &Kind, taken: &[u16], found: &mut Vec<(usize, usize)>) {
        let empty = |q: usize| q == origin || self.cells[q] == EMPTY;
        for s in &k.steps {
            if s.may != 3 || (s.only_unmoved && !self.unmoved[origin]) {
                continue;
            }
            for (i, &d) in s.deltas[side].iter().enumerate() {
                let u = s.units[side][i];
                let n = s.lengths[i];
                if s.rides == 1 {
                    let end = (at as i32 + d) as usize;
                    if self.cells[end] == OFF || !empty(end) {
                        continue;
                    }
                    let mut victim = None;
                    let mut standing = 0;
                    for j in 1..n {
                        let q = (at as i32 + u * j) as usize;
                        if !empty(q) {
                            standing += 1;
                            victim = Some(q);
                        }
                    }
                    if standing == 1 {
                        let v = victim.unwrap();
                        if side_of(self.cells[v]) != side && !taken.contains(&(v as u16)) {
                            found.push((end, v));
                        }
                    }
                    continue;
                }
                let mut t = at as i32 + u;
                let mut times = 1;
                let mut victim: Option<usize> = None;
                loop {
                    let q = t as usize;
                    if self.cells[q] == OFF || (s.rides != 0 && times > s.rides) {
                        break;
                    }
                    if !empty(q) {
                        if victim.is_some() || side_of(self.cells[q]) == side || taken.contains(&(q as u16)) {
                            break;
                        }
                        victim = Some(q);
                    } else if let Some(v) = victim {
                        found.push((q, v));
                    }
                    t += u;
                    times += 1;
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn take_over_from(&self, g: &Game, origin: usize, at: usize, side: usize, k: &Kind, far: i32, taken: &mut Vec<u16>,
                      out: &mut Vec<Move>, from_here: usize) {
        let mut found = Vec::new();
        self.hops(origin, at, side, k, taken, &mut found);
        if found.is_empty() {
            if !taken.is_empty() {
                Board::push_chain(origin, at, 0, taken, out, from_here);
            }
            return;
        }
        for (end, victim) in found {
            taken.push(victim as u16);
            if !k.becomes.is_empty() && g.row(end) == far {
                for &b in &k.becomes {
                    Board::push_chain(origin, end, b + 1, taken, out, from_here);
                }
            } else if k.takes_again && taken.len() <= MORE {
                self.take_over_from(g, origin, end, side, k, far, taken, out, from_here);
            } else {
                Board::push_chain(origin, end, 0, taken, out, from_here);
            }
            taken.pop();
        }
    }

    fn push_chain(origin: usize, end: usize, becomes: u8, taken: &[u16], out: &mut Vec<Move>, from_here: usize) {
        if taken.is_empty() || taken.len() > MORE + 1 {
            return;
        }
        let mut sorted = taken.to_vec();
        sorted.sort_unstable();
        let mut m = Move { from: origin as u16, to: end as u16, becomes, taken_at: sorted[0], ..Move::null() };
        for (i, &q) in sorted[1..].iter().enumerate() {
            m.more[i] = q;
        }
        m.n_more = (sorted.len() - 1) as u8;
        if !out[from_here..].contains(&m) {
            out.push(m);
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
            if (c - col).abs() <= 2 {
                // The partner must stand beyond where the royal lands.
                continue;
            }
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
                partner_from: p as u16,
                partner_to: g.sq(row, col + way) as u16,
                ..Move::null()
            });
        }
    }

    fn clear_unmoved(&mut self, g: &Game, sq: usize, undo: &mut Undo, slot: &mut usize) {
        if self.unmoved[sq] {
            self.unmoved[sq] = false;
            if self.cells[sq] != EMPTY {
                self.hash ^= g.zob_unmoved[sq];
            }
            if *slot < 4 + MORE {
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
            captured_more: [EMPTY; MORE],
            unmoved_cleared: [NONE; 4 + MORE],
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
        for i in 0..m.n_more as usize {
            let at = m.more[i] as usize;
            self.clear_unmoved(g, at, &mut undo, &mut slot);
            undo.captured_more[i] = self.cells[at];
            self.hash ^= g.zob_piece[undo.captured_more[i] as usize * g.size + at];
            self.cells[at] = EMPTY;
        }
        self.clear_unmoved(g, to, &mut undo, &mut slot);
        // The partner is lifted before the mover lands: where the two change places, it stands on the square the mover
        // takes.
        let mut partner = EMPTY;
        if m.partner_from != NONE {
            let pf = m.partner_from as usize;
            self.clear_unmoved(g, pf, &mut undo, &mut slot);
            partner = self.cells[pf];
            self.hash ^= g.zob_piece[partner as usize * g.size + pf];
            self.cells[pf] = EMPTY;
        }
        self.hash ^= g.zob_piece[moved as usize * g.size + from];
        self.cells[from] = EMPTY;
        let placed = if m.becomes != 0 { code((m.becomes - 1) as usize, side_of(moved)) } else { moved };
        self.cells[to] = placed;
        self.hash ^= g.zob_piece[placed as usize * g.size + to];
        if m.partner_from != NONE {
            let pt = m.partner_to as usize;
            self.cells[pt] = partner;
            self.hash ^= g.zob_piece[partner as usize * g.size + pt];
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
        let partner = if m.partner_from != NONE { self.cells[m.partner_to as usize] } else { EMPTY };
        if m.partner_from != NONE {
            self.cells[m.partner_to as usize] = EMPTY;
        }
        self.cells[to] = EMPTY;
        self.cells[from] = u.moved;
        if m.partner_from != NONE {
            self.cells[m.partner_from as usize] = partner;
        }
        if m.taken_at != NONE {
            self.cells[m.taken_at as usize] = u.captured;
        }
        for i in 0..m.n_more as usize {
            self.cells[m.more[i] as usize] = u.captured_more[i];
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
    /// The best move as ``packed``: its squares and what it becomes, which find it again among the moves made.
    best: Option<u32>,
}

/// A move as a table keeps it: its two squares and what it becomes. None where a board or a family is too big to pack.
fn packed(m: &Move) -> Option<u32> {
    if m.from >= 1024 || m.to >= 1024 || m.becomes >= 16 {
        return None;
    }
    Some(m.from as u32 | (m.to as u32) << 10 | (m.becomes as u32) << 20)
}

fn found_among(moves: &[Move], key: u32) -> Option<Move> {
    moves.iter().copied().find(|m| packed(m) == Some(key))
}

/// The transposition table, in slots each one word of key and one of what is known, so searchers in threads of their
/// own can share it without a lock: the key is kept xor'd with the word it goes with, and a slot read while another
/// thread wrote it reads as some other position's, and is not used.
struct Table {
    slots: Vec<(AtomicU64, AtomicU64)>,
}

/// The table kept between one search and the next of the same game with the same weights: what the last move's search
/// worked out is where the next one starts. Other weights, or another game, are another table.
static KEPT: OnceLock<Mutex<Option<(u64, Arc<Table>)>>> = OnceLock::new();

fn kept_table(description: &str, weights: &str, entries: usize) -> Arc<Table> {
    let mut key: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in description.bytes().chain([0u8]).chain(weights.bytes()) {
        key = (key ^ byte as u64).wrapping_mul(0x0000_0100_0000_01b3);
    }
    key ^= entries as u64;
    let mut held = KEPT.get_or_init(|| Mutex::new(None)).lock().unwrap();
    if let Some((k, table)) = held.as_ref() {
        if *k == key {
            return Arc::clone(table);
        }
    }
    let table = Arc::new(Table::new(entries));
    *held = Some((key, Arc::clone(&table)));
    table
}

impl Table {
    fn new(entries: usize) -> Table {
        Table { slots: (0..entries.max(1)).map(|_| (AtomicU64::new(0), AtomicU64::new(0))).collect() }
    }

    fn len(&self) -> usize {
        self.slots.len()
    }

    fn get(&self, i: usize) -> Entry {
        let data = self.slots[i].1.load(Ordering::Relaxed);
        let key = self.slots[i].0.load(Ordering::Relaxed) ^ data;
        if data == 0 {
            return Entry::default();
        }
        let score = ((data & 0x3F_FFFF) as i32) << 10 >> 10; // 22 bits, sign kept
        let depth = ((data >> 22) & 0xFF) as u8 as i8;
        let flag = ((data >> 30) & 0x3) as u8;
        let best = if (data >> 32) & 1 == 1 { Some(((data >> 33) & 0x1FF_FFFF) as u32) } else { None };
        Entry { key, depth, flag, score, best }
    }

    fn set(&self, i: usize, e: Entry) {
        let score = (e.score.clamp(-(1 << 21), (1 << 21) - 1) as u32 as u64) & 0x3F_FFFF;
        let mut data = score | (e.depth as u8 as u64) << 22 | (e.flag as u64 & 0x3) << 30;
        if let Some(b) = e.best {
            data |= 1 << 32 | (b as u64 & 0x1FF_FFFF) << 33;
        }
        // Never a word of nothing for a position kept: nothing is what an empty slot reads as.
        data |= 1 << 63;
        self.slots[i].0.store(e.key ^ data, Ordering::Relaxed);
        self.slots[i].1.store(data, Ordering::Relaxed);
    }
}

struct Searcher<'a> {
    g: &'a Game,
    b: Board,
    tt: &'a Table,
    /// Set by the searcher whose answer is used, when it is done: the others stop with it.
    halt: Option<&'a AtomicBool>,
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
        let any_exposed = g.kinds.iter().any(|k| k.exposed != 0);
        let any_beyond = g.beyond.iter().any(|w| *w != 0.0);
        if any_mobility || any_exposed || any_beyond || g.royal_pressed != 0 {
            // One walk of each side's reach answers mobility, pressure on a royal piece and what is left exposed: it
            // used to be a whole move generation per side and a ray scan for every square asked about.
            let b = &self.b;
            REACH.with(|cell| {
                let mut reach = cell.borrow_mut();
                reach.walked(g, b);
                for side in 0..2 {
                    for (ki, k) in g.kinds.iter().enumerate() {
                        score[side] += reach.mobility[side][ki] * k.mobility;
                    }
                }
                if g.royal_pressed != 0 {
                    for side in 0..2 {
                        for r in b.royals(g, side) {
                            for dr in -1..=1 {
                                for dc in -1..=1 {
                                    let t = (r as i32 + dr * g.stride + dc) as usize;
                                    if b.cells[t] != OFF && reach.attacks[1 - side][t] > 0 {
                                        score[side] -= g.royal_pressed;
                                    }
                                }
                            }
                        }
                    }
                }
                if any_exposed {
                    for &sq in &g.on_board {
                        let c = b.cells[sq];
                        if c == EMPTY {
                            continue;
                        }
                        let s = side_of(c);
                        let k = &g.kinds[kind_of(c)];
                        if k.exposed != 0 && reach.attacks[1 - s][sq] > 0 && reach.attacks[s][sq] == 0 {
                            score[s] -= k.exposed;
                        }
                    }
                }
                if any_beyond {
                    let counted = beyond_the_horizon(g, b, &reach);
                    for side in 0..2 {
                        let worth: f32 = (0..BEYOND).map(|i| g.beyond[i] * counted[side][i]).sum();
                        score[side] += worth.round() as i32;
                    }
                }
            });
        }
        let me = self.b.turn;
        score[me] - score[1 - me] + g.tempo + self.learned_judgement()
    }

    /// Which (kind, the mover's or not, square in the mover's frame) each piece is: the learned judgement's inputs.
    fn cells_seen_by_the_mover(&self) -> Vec<u16> {
        let g = self.g;
        let me = self.b.turn;
        let squares = (g.rows * g.cols) as usize;
        let mut seen = Vec::with_capacity(32);
        for &sq in &g.on_board {
            let c = self.b.cells[sq];
            if c == EMPTY {
                continue;
            }
            let rel = (side_of(c) != me) as usize;
            let (r, col) = g.rc(sq);
            let row = if me == 0 { r } else { g.rows - 1 - r };
            seen.push(((kind_of(c) * 2 + rel) * squares + (row * g.cols + col) as usize) as u16);
        }
        seen
    }

    fn learned_judgement(&self) -> i32 {
        let Some(l) = &self.g.learned else { return 0 };
        let g = self.g;
        let me = self.b.turn;
        let squares = (g.rows * g.cols) as usize;
        let mut h = [0f32; HIDDEN_AT_MOST];
        h[..l.hidden].copy_from_slice(&l.first_bias);
        for &sq in &g.on_board {
            let c = self.b.cells[sq];
            if c == EMPTY {
                continue;
            }
            let rel = (side_of(c) != me) as usize;
            let (r, col) = g.rc(sq);
            let row = if me == 0 { r } else { g.rows - 1 - r };
            let at = ((kind_of(c) * 2 + rel) * squares + (row * g.cols + col) as usize) * l.hidden;
            let w = &l.first[at..at + l.hidden];
            for j in 0..l.hidden {
                h[j] += w[j];
            }
        }
        let mut out = l.out_bias;
        for j in 0..l.hidden {
            out += l.out[j] * h[j].max(0.0);
        }
        out.round() as i32
    }

    fn worth_of(&self, c: u8) -> i32 {
        if c == EMPTY || c == OFF {
            0
        } else {
            self.g.kinds[kind_of(c)].worth.max(1)
        }
    }

    fn order(&self, moves: &mut [Move], best: Option<u32>, ply: usize) {
        let mut keyed: Vec<(i64, Move)> = moves
            .iter()
            .map(|&m| {
                let key: i64 = if best.is_some() && packed(&m) == best {
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
            && (self.started.elapsed().as_secs_f64() > self.seconds
                || (self.max_nodes > 0 && self.nodes > self.max_nodes)
                || self.halt.is_some_and(|h| h.load(Ordering::Relaxed)))
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
        if self.g.arrival_wins && self.b.arrived(self.g, 1 - self.b.turn) {
            return -MATE + ply as i32;
        }
        let side = self.b.turn;
        let checked = self.b.in_check(self.g, side);
        let mut stand = -INF;
        if !checked {
            stand = self.evaluate();
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
        let unit = self.g.margin.load(Ordering::Relaxed);
        for m in moves {
            // A take that would leave her short of the bound even with what it takes and two margins cannot raise the
            // bound: not looked at.
            if !checked && unit > 0 && m.becomes == 0 && m.taken_at != NONE
                && stand + self.worth_of(self.b.cells[m.taken_at as usize]) + 2 * unit < alpha
            {
                continue;
            }
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
            if g.arrival_wins && self.b.arrived(g, 1 - self.b.turn) {
                return -MATE + ply as i32;
            }
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
        let entry = self.tt.get(idx);
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
        let unit = g.margin.load(Ordering::Relaxed);
        let stand = if !checked && !pv_node { self.evaluate() } else { -INF };
        // A position past the bound by more than a margin for each ply still to look: what those plies could find
        // against it is not worth looking for.
        if !checked && !pv_node && unit > 0 && depth <= 6 && beta.abs() < MATE - 1000
            && stand - unit * depth >= beta
        {
            return stand - unit * depth;
        }
        if !checked && null_ok && !pv_node && depth >= 3 && self.has_pieces_beyond_irreversible(side) {
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
        let mut quiets_tried: Vec<Move> = Vec::new();
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
            if legal > 1 && !pv_node && !checked && quiet && !gives_check && best_score > -MATE + 1000 {
                // Near the end of a look, a quiet move late in the order, or one that even with a margin cannot lift
                // her to the bound, is not looked at.
                let late = depth <= 4 && legal as i32 > 3 + depth * depth;
                let futile = unit > 0 && depth <= 2 && stand + 2 * unit * depth <= alpha;
                if late || futile {
                    self.b.unmake(u);
                    continue;
                }
            }
            let mut score;
            if legal == 1 {
                score = -self.search(depth - 1, -beta, -alpha, ply + 1, true);
            } else {
                let mut reduction = 0;
                if depth >= 3 && legal > 3 && quiet && !checked && !gives_check {
                    // The later a move comes in the order and the deeper the look, the less it is likely to be the
                    // one: reduced by both, in proportion to their logarithms.
                    let r = 0.75 + (depth as f64).ln() * (legal as f64).ln() / 2.25;
                    reduction = (r as i32 - pv_node as i32).clamp(1, depth - 2);
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
                        // The quiet moves tried before it and found wanting are tried later next time.
                        for q in &quiets_tried {
                            let h = &mut self.history[q.from as usize * g.size + q.to as usize];
                            *h = (*h - depth * depth).max(-(1 << 20));
                        }
                    }
                    break;
                }
            }
            if quiet {
                quiets_tried.push(m);
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
            self.tt.set(idx, Entry { key, depth: depth.min(127) as i8, flag, score: stored,
                                     best: best_move.and_then(|m| packed(&m)) });
        }
        best_score
    }

    fn line(&mut self, first: Move, most: usize) -> Vec<Move> {
        let mut line = vec![first];
        let mut undos = vec![self.b.make(self.g, first)];
        let mut seen = vec![self.b.hash];
        while line.len() < most {
            let e = self.tt.get(self.tt_index(self.b.hash));
            if e.key != self.b.hash {
                break;
            }
            let Some(k) = e.best else { break };
            let legal = self.b.legal(self.g);
            let Some(m) = found_among(&legal, k) else { break };
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
    fn fresh(g: &'a Game, b: Board, tt: &'a Table, seconds: f64, started: Instant, max_nodes: u64) -> Searcher<'a> {
        Searcher {
            g,
            b,
            tt,
            halt: None,
            killers: vec![[Move::null(); 2]; MAX_PLY],
            history: vec![0; g.size * g.size],
            nodes: 0,
            started,
            seconds,
            stop: false,
            max_nodes,
            excluded: vec![],
            unsteady: 0,
        }
    }
}

impl<'a> Searcher<'a> {
    /// Iterative deepening at the root over ``legal``: the best move, its score, and the depth completed.
    fn best(&mut self, deepest: i32, legal: &[Move]) -> (Option<Move>, i32, i32) {
        self.best_from(1, deepest, legal)
    }

    /// The same, beginning at ``first``: a searcher beside another begins a depth on from it, to look where it has
    /// not yet.
    fn best_from(&mut self, first: i32, deepest: i32, legal: &[Move]) -> (Option<Move>, i32, i32) {
        if legal.is_empty() {
            return (None, 0, 0);
        }
        let mut best = legal[0];
        let mut best_score = 0;
        let mut done = 0;
        let mut window = (-INF, INF);
        for depth in first.max(1)..=deepest.max(first.max(1)) {
            let mut score = self.search(depth, window.0, window.1, 0, false);
            if !self.stop && (score <= window.0 || score >= window.1) {
                score = self.search(depth, -INF, INF, 0, false);
            }
            if self.stop {
                break;
            }
            let e = self.tt.get(self.tt_index(self.b.hash));
            if e.key == self.b.hash {
                if let Some(m) = e.best.and_then(|k| found_among(legal, k)) {
                    best = m;
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
        let mut f = vec![0f32; 2 * k + 2 + k * squares + k + BEYOND];
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
        // Moves, pressure on a royal piece and what is left exposed, from the same walk of each side's reach the
        // search judges positions by, so what is fitted is what is searched with.
        let b = &self.b;
        REACH.with(|cell| {
            let mut reach = cell.borrow_mut();
            reach.walked(g, b);
            for side in 0..2 {
                let sign = if side == 0 { 1.0 } else { -1.0 };
                for kind in 0..k {
                    f[k + kind] += sign * reach.mobility[side][kind] as f32;
                }
                for r in b.royals(g, side) {
                    for dr in -1..=1 {
                        for dc in -1..=1 {
                            let t = (r as i32 + dr * g.stride + dc) as usize;
                            if b.cells[t] != OFF && reach.attacks[1 - side][t] > 0 {
                                f[2 * k] -= sign;
                            }
                        }
                    }
                }
            }
            // Per kind, the pieces of it each side leaves attacked and undefended: the second side's less the first's.
            let exposed_at = 2 * k + 2 + k * squares;
            for &sq in &g.on_board {
                let c = b.cells[sq];
                if c == EMPTY {
                    continue;
                }
                let s = side_of(c);
                if reach.attacks[1 - s][sq] > 0 && reach.attacks[s][sq] == 0 {
                    f[exposed_at + kind_of(c)] += if s == 0 { -1.0 } else { 1.0 };
                }
            }
            // What lies beyond the horizon: the first side's less the second's.
            let counted = beyond_the_horizon(g, b, &reach);
            for i in 0..BEYOND {
                f[exposed_at + k + i] = counted[0][i] - counted[1][i];
            }
        });
        f[2 * k + 1] = if self.b.turn == 0 { 1.0 } else { -1.0 };
        f
    }
}

/// What lies beyond a search's horizon at ``b``, per side, in BEYOND's order, from the rules and the walk of each side's
/// reach. Nothing here knows a game: a piece that only goes forward and becomes another kind is read from its steps.
fn beyond_the_horizon(g: &Game, b: &Board, reach: &Reach) -> [[f32; BEYOND]; 2] {
    // One pass over the board: every square's ground, every piece held, and where the runners and royal pieces stand;
    // then only those few. Walking the board once per quantity cost a sixth of her search's pace.
    const MAX_COLS: usize = 32;
    const MOST: usize = 64;
    let mut out = [[0f32; BEYOND]; 2];
    let last = (g.rows - 1).max(1) as f32;
    let cols = (g.cols as usize).min(MAX_COLS);
    let half = g.rows;
    // Per line, how far up the board the second side's runners reach and how far down the first side's do.
    let mut second_highest = [-1i32; MAX_COLS];
    let mut first_lowest = [g.rows; MAX_COLS];
    let mut runners = [(0usize, 0usize, 0i32, 0i32); MOST];
    let mut n_runners = 0;
    let mut royals = [(0usize, 0usize); 4];
    let mut n_royals = 0;
    for &sq in &g.on_board {
        let r = g.row(sq);
        let (a0, a1) = (reach.attacks[0][sq], reach.attacks[1][sq]);
        // Ground: a square in the other side's half that one side reaches and the other does not.
        if 2 * r >= half && a0 > 0 && a1 == 0 {
            out[0][3] += 1.0;
        } else if 2 * (g.rows - 1 - r) >= half && a1 > 0 && a0 == 0 {
            out[1][3] += 1.0;
        }
        let c = b.cells[sq];
        if c == EMPTY {
            continue;
        }
        let s = side_of(c);
        let k = &g.kinds[kind_of(c)];
        if k.royal {
            if n_royals < royals.len() {
                royals[n_royals] = (sq, s);
                n_royals += 1;
            }
        } else if reach.attacks[s][sq] > 0 {
            out[s][2] += 1.0;
        }
        if k.runner {
            let col = sq as i32 % g.stride - g.border;
            if (col as usize) < cols {
                if s == 1 {
                    second_highest[col as usize] = second_highest[col as usize].max(r);
                } else {
                    first_lowest[col as usize] = first_lowest[col as usize].min(r);
                }
            }
            if n_runners < MOST {
                runners[n_runners] = (sq, s, r, col);
                n_runners += 1;
            }
        }
    }
    for &(sq, s, r, col) in &runners[..n_runners] {
        let k = &g.kinds[kind_of(b.cells[sq])];
        let mut opposed = false;
        for cc in (col - 1)..=(col + 1) {
            if cc < 0 || cc as usize >= cols {
                continue;
            }
            let cc = cc as usize;
            if (s == 0 && second_highest[cc] > r) || (s == 1 && first_lowest[cc] < r) {
                opposed = true;
            }
        }
        if !opposed {
            let own_row = if s == 0 { r } else { g.rows - 1 - r };
            let share = own_row as f32 / last;
            out[s][0] += share * share;
        }
        let mut quiet_ways = 0;
        let mut open = false;
        for step in &k.steps {
            if step.may == 2 || step.may == 3 || step.only_unmoved {
                continue;
            }
            for d in &step.units[s] {
                quiet_ways += 1;
                if b.cells[(sq as i32 + d) as usize] == EMPTY {
                    open = true;
                }
            }
        }
        if quiet_ways > 0 && !open {
            out[s][1] += 1.0;
        }
    }
    for &(sq, s) in &royals[..n_royals] {
        for dr in -1..=1 {
            for dc in -1..=1 {
                if dr == 0 && dc == 0 {
                    continue;
                }
                let o = b.cells[(sq as i32 + dr * g.stride + dc) as usize];
                if o != EMPTY && o != OFF && side_of(o) == s {
                    out[s][4] += 1.0;
                }
            }
        }
    }
    out
}

/// The most one quiet move moves her judgement at ``b``, the turn's own worth left aside: set as the game's margin.
fn measure_the_margin(g: &Game, b: &Board) {
    let table = Table::new(1);
    let mut s = Searcher::fresh(g, b.clone(), &table, 1e9, Instant::now(), 0);
    let before = s.evaluate();
    let mut most = 0;
    for m in s.b.legal(g) {
        if m.taken_at != NONE || m.becomes != 0 {
            continue;
        }
        let u = s.b.make(g, m);
        let after = -s.evaluate();
        s.b.unmake(u);
        most = most.max((after - before + 2 * g.tempo).abs());
    }
    g.margin.store(most, Ordering::Relaxed);
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

/// What lies beyond the horizon at a position, per side, in BEYOND's order (core/agency/what_lies_beyond_the_horizon.py).
#[pyfunction]
fn beyond_counts(description: &str, position: &str) -> PyResult<Vec<Vec<f32>>> {
    let (g, b) = parse(description, position, "{}")?;
    let counted = REACH.with(|cell| {
        let mut reach = cell.borrow_mut();
        reach.walked(&g, &b);
        beyond_the_horizon(&g, &b, &reach)
    });
    Ok(counted.iter().map(|side| side.to_vec()).collect())
}

/// The value of the position for the side to move, by the weights, without search.
#[pyfunction]
fn evaluate(description: &str, position: &str, weights: &str) -> PyResult<i32> {
    let (g, b) = parse(description, position, weights)?;
    let table = Table::new(1);
    let mut s = Searcher {
        g: &g,
        b,
        tt: &table,
        halt: None,
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
#[pyo3(signature = (description, position, weights, seconds, deepest=64, hash_mb=64, nodes=0, excluded=vec![], threads=1))]
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
    threads: usize,
) -> PyResult<PyObject> {
    let (g, b) = parse(description, position, weights)?;
    measure_the_margin(&g, &b);
    let entries = ((hash_mb.max(1) * 1024 * 1024) / std::mem::size_of::<(AtomicU64, AtomicU64)>()).max(1024);
    if threads > 1 && nodes == 0 {
        // Searchers in threads of their own over the same position, sharing one table, so what each finds is there
        // for the others. The first one's answer is the answer; the rest stop with it.
        let result = py.allow_threads(|| {
            let table = kept_table(description, weights, entries);
            let table: &Table = &table;
            let halt = AtomicBool::new(false);
            let started = Instant::now();
            let mut main = Searcher::fresh(&g, b.clone(), table, seconds, started, 0);
            let all = main.b.legal(&g);
            let legal: Vec<Move> = all.iter().copied().filter(|m| !excluded.contains(&move_out(&g, m))).collect();
            if legal.is_empty() {
                return (None, 0, 0, 0u64, vec![], 0, 0);
            }
            main.excluded = all.into_iter().filter(|m| !legal.contains(m)).collect();
            let mut helpers: Vec<Searcher> = (1..threads)
                .map(|_| {
                    let mut h = Searcher::fresh(&g, b.clone(), table, seconds, started, 0);
                    h.halt = Some(&halt);
                    h.excluded = main.excluded.clone();
                    h
                })
                .collect();
            let (found, best_score, done) = std::thread::scope(|scope| {
                for (n, helper) in helpers.iter_mut().enumerate() {
                    let legal = &legal;
                    scope.spawn(move || {
                        helper.best_from(1 + (n as i32 % 2), deepest, legal);
                    });
                }
                let answer = main.best(deepest, &legal);
                halt.store(true, Ordering::Relaxed);
                answer
            });
            let best = found.unwrap_or(legal[0]);
            let line = main.line(best, 12);
            let n = main.nodes + helpers.iter().map(|h| h.nodes).sum::<u64>();
            let ms = (started.elapsed().as_secs_f64() * 1000.0) as u64;
            (Some(best), best_score, done, n, line, ms, main.unsteady)
        });
        return search_result(py, &g, result);
    }
    let result = py.allow_threads(|| {
        // Searched by time, a search starts where the last one of this game left its table; searched by a count of
        // nodes, it starts from nothing, so that a count means the same thing every time.
        let table = if nodes == 0 { kept_table(description, weights, entries) } else { Arc::new(Table::new(entries)) };
        let mut s = Searcher {
            g: &g,
            b,
            tt: &table,
            halt: None,
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
    search_result(py, &g, result)
}

#[allow(clippy::type_complexity)]
fn search_result(py: Python<'_>, g: &Game, result: (Option<Move>, i32, i32, u64, Vec<Move>, u64, i32))
                 -> PyResult<PyObject> {
    let d = PyDict::new_bound(py);
    let (best, score, depth, n, line, ms, unsteady) = result;
    d.set_item("margin", g.margin.load(Ordering::Relaxed))?;
    match best {
        Some(m) => d.set_item("move", move_out(g, &m))?,
        None => d.set_item("move", py.None())?,
    }
    d.set_item("score", score)?;
    d.set_item("depth", depth)?;
    d.set_item("nodes", n)?;
    d.set_item("ms", ms)?;
    d.set_item("mate", score.abs() > MATE - 1000)?;
    d.set_item("unsteady", unsteady)?;
    let line_out: Vec<(i32, i32, i32, i32, String)> = line.iter().map(|m| move_out(g, m)).collect();
    d.set_item("line", line_out)?;
    Ok(d.into())
}

/// Games played in the model between two sets of weights, colours swapped each game, the first ``random_plies``
/// of each chosen at random for variety: the points the first weights won, and samples of quiet positions (features,
/// and the game's result for the first side: 1, 0.5, 0) for fitting weights to what happened. With them, which game
/// each sample came from, and each game's count of every kind where it stopped (first side less second): a game that
/// stopped without a result can still be judged by how it stood.
#[pyfunction]
#[pyo3(signature = (description, start, weights_a, weights_b, games, nodes_per_move, random_plies=4, seed=1, max_plies=300, sample_every=2, deep_nodes=0, nodes_per_move_b=0))]
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
    deep_nodes: u64,
    nodes_per_move_b: u64,
) -> PyResult<PyObject> {
    let (ga, b0) = parse(description, start, weights_a)?;
    let (gb, _) = parse(description, start, weights_b)?;
    let (points, results, samples, lengths, sample_games, final_counts, sample_scores, sample_cells, sample_static,
         sample_quiet, sample_turns) = py.allow_threads(|| {
        let mut rng = Rng(seed.wrapping_mul(0x2545F4914F6CDD1D) | 1);
        let entries = (4 * 1024 * 1024) / std::mem::size_of::<Entry>();
        let mut points = 0.0f64;
        let mut results: Vec<f64> = Vec::new();
        let mut lengths: Vec<u32> = Vec::new();
        let mut samples: Vec<(Vec<f32>, f64)> = Vec::new();
        let mut sample_games: Vec<u32> = Vec::new();
        let mut final_counts: Vec<Vec<f32>> = Vec::new();
        // What her search found each sampled position worth, for the first side, in the order the game reached them:
        // what a position turned out to be worth, by the positions after it, is learned from these.
        let mut sample_scores: Vec<i32> = Vec::new();
        // For a judgement learned over where each kind stands: each sample's pieces as the mover sees them, her first
        // judgement of it for the mover, whether a look at the takes alone leaves that judgement as it is, and whose
        // turn it was.
        let mut sample_cells: Vec<Vec<u16>> = Vec::new();
        let mut sample_static: Vec<i32> = Vec::new();
        let mut sample_quiet: Vec<bool> = Vec::new();
        let mut sample_turns: Vec<u8> = Vec::new();
        for game in 0..games {
            let a_is_first = game % 2 == 0;
            let (ta, tb) = (Table::new(entries), Table::new(entries));
            let mut sa = Searcher { g: &ga, b: b0.clone(), tt: &ta, halt: None, killers: vec![[Move::null(); 2]; MAX_PLY], history: vec![0; ga.size * ga.size], nodes: 0, started: Instant::now(), seconds: 1e9, stop: false, max_nodes: nodes_per_move, excluded: vec![], unsteady: 0 };
            let mut sb = Searcher { g: &gb, b: b0.clone(), tt: &tb, halt: None, killers: vec![[Move::null(); 2]; MAX_PLY], history: vec![0; gb.size * gb.size], nodes: 0, started: Instant::now(), seconds: 1e9, stop: false, max_nodes: nodes_per_move, excluded: vec![], unsteady: 0 };
            // The second side looks at as many positions as its own pace affords in the same time, where it is given.
            if nodes_per_move_b > 0 {
                sb.max_nodes = nodes_per_move_b;
            }
            let mut board = b0.clone();
            let mut taken: Vec<Vec<f32>> = Vec::new();
            let mut taken_scores: Vec<i32> = Vec::new();
            let mut result_first = 0.5;
            let mut last_took = false;
            let mut plies = 0u32;
            for ply in 0..max_plies {
                plies = ply;
                if ga.arrival_wins && board.arrived(&ga, 1 - board.turn) {
                    result_first = if board.turn == 1 { 1.0 } else { 0.0 };
                    break;
                }
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
                let mut found_score = 0;
                let chosen = if ply < random_plies {
                    legal[(rng.next() % legal.len() as u64) as usize]
                } else {
                    measure_the_margin(if first_moves { &ga } else { &gb }, &board);
                    let s = if first_moves { &mut sa } else { &mut sb };
                    s.b = board.clone();
                    s.nodes = 0;
                    s.stop = false;
                    s.started = Instant::now();
                    let (best, score, _) = s.best(64, &legal);
                    found_score = if board.turn == 0 { score } else { -score };
                    best.unwrap_or(legal[0])
                };
                if ply >= random_plies && sample_every > 0 && ply % sample_every == 0 && !last_took
                    && !board.in_check(&ga, board.turn)
                {
                    sa.b = board.clone();
                    taken.push(sa.features());
                    sample_cells.push(sa.cells_seen_by_the_mover());
                    let judged = sa.evaluate();
                    // Her first judgement without the learned one: what a learned judgement is fitted on top of.
                    let first_look = judged - sa.learned_judgement();
                    sa.nodes = 0;
                    sa.stop = false;
                    sa.started = Instant::now();
                    let saved_nodes = sa.max_nodes;
                    sa.max_nodes = 0;
                    let takes_settled = sa.quiesce(-INF, INF, 0);
                    sa.max_nodes = saved_nodes;
                    sample_static.push(first_look);
                    sample_quiet.push(takes_settled == judged);
                    sample_turns.push(board.turn as u8);
                    // With ``deep_nodes``, what a much longer look of hers finds the position worth: her slow
                    // thinking, kept to teach her quick judgement what it would have found.
                    let score = if deep_nodes > 0 {
                        measure_the_margin(&ga, &board);
                        let saved = sa.max_nodes;
                        sa.max_nodes = deep_nodes;
                        sa.nodes = 0;
                        sa.stop = false;
                        sa.started = Instant::now();
                        let (_, deep, _) = sa.best(64, &legal);
                        sa.max_nodes = saved;
                        if board.turn == 0 { deep } else { -deep }
                    } else {
                        found_score
                    };
                    taken_scores.push(score);
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
            sample_scores.extend(taken_scores);
            sa.b = board.clone();
            let k = ga.kinds.len();
            final_counts.push(sa.features()[..k].to_vec());
        }
        (points, results, samples, lengths, sample_games, final_counts, sample_scores, sample_cells, sample_static,
         sample_quiet, sample_turns)
    });
    let d = PyDict::new_bound(py);
    d.set_item("points", points)?;
    d.set_item("results", results)?;
    d.set_item("plies", lengths)?;
    d.set_item("samples", samples)?;
    d.set_item("sample_games", sample_games)?;
    d.set_item("final_counts", final_counts)?;
    d.set_item("sample_scores", sample_scores)?;
    d.set_item("sample_cells", sample_cells)?;
    d.set_item("sample_static", sample_static)?;
    d.set_item("sample_quiet", sample_quiet)?;
    d.set_item("sample_turns", sample_turns)?;
    d.set_item("kind_order", ga.kinds.iter().map(|k| k.letter.clone()).collect::<Vec<_>>())?;
    Ok(d.into())
}

#[pymodule]
fn aura_board_search(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(perft, m)?)?;
    m.add_function(wrap_pyfunction!(legal_moves, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate, m)?)?;
    m.add_function(wrap_pyfunction!(beyond_counts, m)?)?;
    m.add_function(wrap_pyfunction!(search, m)?)?;
    m.add_function(wrap_pyfunction!(self_play, m)?)?;
    Ok(())
}
