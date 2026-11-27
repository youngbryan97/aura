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
use serde::{Deserialize, Serialize};
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
    /// Whether the game has shown this way often enough to be relied on. A side asked to be wary of its own unsure ways
    /// does not count on them past its first move (see `Board::wary`).
    #[serde(default = "sure_by_default")]
    sure: bool,
}

fn sure_by_default() -> bool {
    true
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
    /// How many pieces stood at the start: how far the board has emptied is counted against it.
    #[serde(default)]
    start_count: i32,
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
    /// The side that acts in the world this search is for, and so does not count on its own unsure ways.
    #[serde(default)]
    wary: Option<i32>,
    /// Moves the other side has been seen to make, as (row, column) from and to and the share of its moves each was:
    /// tried early where they could make them again.
    #[serde(default)]
    expected: Vec<(i32, i32, i32, i32, f64)>,
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
    /// How she searches: the numbers her look is shaped by, tuned by her own games where she has tuned them.
    #[serde(default)]
    habits: Habits,
    /// Per kind, rows by columns in the first side's frame as `squares` is: what each square adds as the board empties,
    /// weighed by the share of the pieces at the start that are gone.
    #[serde(default)]
    emptying: HashMap<String, Vec<Vec<f64>>>,
}

/// How she searches, in numbers: how far a late move's look is shortened, how many quiet moves are tried near the end
/// of a look, how many plies margins reach and how wide they are, how far a passed turn shortens a look, and how wide
/// a look's first window is. Each begins where it was set by hand and measured (22 November 2026), and her rehearsal
/// may move it by playing herself.
#[derive(Deserialize, Serialize, Clone, Copy)]
#[serde(default)]
struct Habits {
    shorten_from: f64,
    shorten_by: f64,
    quiet_tried: f64,
    quiet_tried_within: f64,
    margin_within: f64,
    futile_within: f64,
    futile_margins: f64,
    take_margins: f64,
    asked_takes_within: f64,
    asked_takes_margins: f64,
    passed_turn: f64,
    passed_turn_by: f64,
    window: f64,
    /// How much of what her quick judgement has been found off by, in positions whose commonest pieces stand as these
    /// do, is put back into it: none leaves her judgement as it was. A person's quick sense of a kind of position is
    /// corrected by how far it has been off in positions like it; this is that, kept while she searches.
    corrected: f64,
    /// How a search given a soft time decides when to stop (`Searcher::soft`): each change of its best move between
    /// depths adds `unsteady_by` of the soft time, those changes fading by `unsteady_kept` a depth; a score that fell
    /// by more than a quiet move's worth adds `falling_by`; and a best move unchanged three depths running stops it
    /// once `steady_share` of the soft time is gone. The general shape of an anytime search's time: think longer where
    /// more thinking is likely to change the decision, and stop where it is not.
    unsteady_by: f64,
    unsteady_kept: f64,
    falling_by: f64,
    steady_share: f64,
    /// Whether what one move's search learned of which moves tend to be best (its history and countermoves) is kept,
    /// halved, for the next move's search of the same game: 1 keeps it, nought starts each move afresh.
    keeps_ordering: f64,
    /// Whether a take is weighed by what it comes to once both sides have taken back on its square while it pays them
    /// (`Searcher::exchange`): 1 orders the takes that lose after the quiet moves and leaves them unlooked at past the
    /// horizon, nought weighs a take by what it takes.
    weighs_exchanges: f64,
    /// Within this many plies of a look's end, away from her best line and out of check, a move that loses more than
    /// it can win back once its square is fought over is not looked at: a take that loses more than
    /// `exchange_takes_margins` margins a ply, a quiet move that loses more than `exchange_quiet_margins` margins a
    /// ply squared. Nought looks at them all.
    exchange_within: f64,
    exchange_takes_margins: f64,
    exchange_quiet_margins: f64,
    /// Whether what a look past the horizon found is kept in the table, where a deeper look's finding is not
    /// displaced by it: 1 keeps it, nought does not.
    keeps_the_horizon: f64,
    /// Whether a look past the horizon ends a game as the search does: a side left no move there has lost or drawn by
    /// the rules, and a position repeated or counted out is drawn. Nought judges it as it stands, as before: a side
    /// left no move a queen down was scored a queen down, +569, where it was a draw (suggestions of 23 November 2026).
    ends_past_the_horizon: f64,
    /// From this many plies of look on, a position the table holds no best move for is looked at a ply shorter, for
    /// a shorter look finds one to begin with next time: nought never.
    unhinted_from: f64,
    /// Whether a position judged better for the side to move than the one two plies before (its side's last turn)
    /// is looked at more fully: where it is not improving, a late quiet move is shortened a ply more and fewer quiet
    /// moves are tried near the end of a look. Nought looks at both alike.
    improving: f64,
    /// What a drawn game is worth to the side she plays (the side the look is wary for), in her judgement's units:
    /// the judgement at which her chance against this opponent is even, as her games against it have turned out.
    /// Nought where the two sides are as likely to win, as against herself.
    draw_worth: f64,
    /// Where a look is about to end and her two judgements of the position disagree by more than this many margins
    /// (what her learned judgement adds to the weighted sum), the look goes a ply further, once on a line: what she
    /// is unsure of is looked into. Nought never.
    unsure_extends: f64,
    /// The chance that another depth changes the answer, as her own searches have shown it
    /// (core/cognition/when_more_thought_changes_the_answer.py): a logistic curve over how often the best move has
    /// changed lately (fading as `unsteady_kept`), how many depths it has held, how far the score just fell in
    /// margins, and the depth. Given it, a soft time is spent by it: the time meant, scaled by that chance against
    /// `flip_mean`, how often a depth has changed the answer. `flip_mean` nought spends it by the rule above.
    flip_mean: f64,
    flip_level: f64,
    flip_changes: f64,
    flip_held: f64,
    flip_fell: f64,
    flip_depth: f64,
    /// How much history shortens a late move's look less, a ply for each so much of it (nought: history does not
    /// change how far a move is shortened). A quiet move that has often been best looks further; one that has often
    /// failed, less far.
    history_lengthens_by: f64,
    /// The most plies a passed turn's look is shortened further for each margin her judgement stands above the
    /// bound: the further above, the less a short look risks. Nought shortens it by depth alone.
    passed_turn_margins: f64,
    /// From this many plies of look on, the move the table holds as best is looked at a ply further where every
    /// other move, looked at half as far, falls short of what the table says it is worth by `singular_margins`
    /// sixty-fourths of a margin a ply: a move that alone holds the position is looked into. Nought never.
    singular_from: f64,
    singular_margins: f64,
    /// Whether searchers beside the first spread over depths, each passing over some in its own phase, so that they
    /// look where the first is not yet looking and not where it already is: 1 spreads them, nought lets each go
    /// depth by depth.
    helpers_spread: f64,
    /// Whether the answer is the one of whichever searcher completed the deepest look (its score breaking ties), not
    /// the first searcher's alone: 1 takes the deepest, nought the first's.
    deepest_answers: f64,
    /// How far a move the other side has been seen to make is put forward among the quiet moves at their turns, as
    /// history counted for its share of their moves (`PosD::expected`). Nought leaves it as any other move.
    expected_weight: f64,
    /// How much of what the side to move stands to lose to a second threat is counted against it: where two or more
    /// of its pieces would each be lost to an exchange on their squares, it can save one, and the next largest loss
    /// is counted, by this share. Nought counts none: a look that stops with two threats standing sees neither.
    double_threats: f64,
}

impl Default for Habits {
    fn default() -> Habits {
        Habits {
            shorten_from: 0.75,
            shorten_by: 2.25,
            quiet_tried: 3.0,
            quiet_tried_within: 4.0,
            margin_within: 6.0,
            futile_within: 2.0,
            futile_margins: 2.0,
            take_margins: 2.0,
            asked_takes_within: 2.0,
            asked_takes_margins: 2.0,
            passed_turn: 2.0,
            passed_turn_by: 6.0,
            window: 50.0,
            // Measured 23 November 2026 against Stockfish held to depth 5, 160 games each way at 0.2 s a move: kept
            // with the table and carried from move to move, 0.406 on against 0.394 off; kept per move, 0.422 both.
            corrected: 1.0,
            unsteady_by: 1.0,
            unsteady_kept: 0.5,
            falling_by: 0.5,
            steady_share: 1.0,
            keeps_ordering: 0.0,
            // Measured 23 November 2026 in her model of chess, 300 games each: 0.702 against weighing a take by what
            // it takes at equal nodes, and 0.637 with her nodes cut by the 12 per cent the weighing costs.
            weighs_exchanges: 1.0,
            exchange_within: 0.0,
            exchange_takes_margins: 1.0,
            exchange_quiet_margins: 0.25,
            // Measured 23 November 2026 in her model of chess at equal nodes, 300 games against the habits as they
            // were: keeping the horizon 0.507 +- 0.026; shortening an unhinted look from four plies 0.558 and, from
            // other openings, 0.493; looking more fully where improving 0.560 and 0.535 (pooled 0.548 +- 0.018).
            keeps_the_horizon: 0.0,
            ends_past_the_horizon: 1.0,
            unhinted_from: 0.0,
            improving: 1.0,
            draw_worth: 0.0,
            unsure_extends: 0.0,
            flip_mean: 0.0,
            flip_level: 0.0,
            flip_changes: 0.0,
            flip_held: 0.0,
            flip_fell: 0.0,
            flip_depth: 0.0,
            history_lengthens_by: 0.0,
            passed_turn_margins: 0.0,
            singular_from: 0.0,
            singular_margins: 64.0,
            // Measured 23 November 2026 against Stockfish held to depth 6 on three threads, 0.2 s a move, 120 games
            // each: both on 0.350 +- 0.032, both off 0.338 +- 0.032; on eight threads for three seconds, a ply deeper
            // over four positions.
            helpers_spread: 1.0,
            deepest_answers: 1.0,
            expected_weight: 0.0,
            // Measured in her model of chess at equal nodes, 300 games: 0.465 +- 0.025 at a full share.
            double_threats: 0.0,
        }
    }
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
    /// Per number of pieces standing, a row of weights into the hidden layer: the same pieces in the same places
    /// stand differently on a board that has emptied. Empty where the judgement was learned without it.
    #[serde(default)]
    count: Vec<Vec<f32>>,
}

/// The learned judgement as the search reads it: weights by (kind, the mover's or not, square) into the hidden layer.
struct Learned {
    hidden: usize,
    first: Vec<f32>,
    first_bias: Vec<f32>,
    out: Vec<f32>,
    out_bias: f32,
    /// Rows by the number of pieces standing, one after another; none where there are none.
    count: Vec<f32>,
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
    sure: bool,
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
    /// Two of its ways can reach the same square, so a move found twice must be listed once.
    overlapping: bool,
    worth: i32,
    mobility: i32,
    squares: [Vec<i32>; 2],
    /// What each square adds as the board empties, per side, before it is weighed by how far it has.
    emptying: [Vec<i32>; 2],
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
    /// How many pieces stood at the start, and whether any kind's squares change as the board empties.
    start_count: i32,
    any_emptying: bool,
    /// The most one quiet move moved her judgement at the position a search began from, the turn's own worth left
    /// aside: the unit the search's margins are counted in. Zero where no quiet move moved it, and then there are no
    /// margins. Measured at each search's start, because how far a move moves her judgement is a property of the
    /// weights she judges by: 94 on her offline weights of 22 November 2026, 357 on her live ones, which count each
    /// square a piece reaches nearly half a pawn.
    margin: AtomicI32,
    learned: Option<Learned>,
    habits: Habits,
    zob_piece: Vec<u64>,
    zob_unmoved: Vec<u64>,
    zob_passable: Vec<u64>,
    zob_turn: u64,
    on_board: Vec<usize>,
    /// Per side, the lines its pieces take along, each walked once whatever kinds share it.
    rays: [Vec<Ray>; 2],
    /// The kinds that take over: asked about apart from the lines.
    taking_over: Vec<usize>,
    /// Per side, by how far one square lies from another (offset by the board's size), which of the side's lines
    /// from the first pass over the second: after a move, only those lines through a square it changed need asking
    /// about. Empty where the side has more lines than one mask holds.
    through: [Vec<u128>; 2],
    /// Endings worked out to the end, where a search is given them (`hold_endings`), and the most pieces any has.
    endings: Option<Arc<Vec<EndingHeld>>>,
    endings_most: usize,
}

/// One ending worked out to the end (core/agency/what_a_small_ending_comes_to.py): its pieces by side and kind in the
/// order its situations are named by, and for each situation what it comes to for the first side, above nought won in
/// that many plies less one, below nought lost so, nought neither.
struct EndingHeld {
    pieces: Vec<(usize, usize)>,
    squares: usize,
    values: Vec<i16>,
}

static ENDINGS: OnceLock<Mutex<HashMap<String, Arc<Vec<EndingHeld>>>>> = OnceLock::new();

/// One line a side's pieces take along, looked at from the square taken. Every kind that takes along the same line
/// shares one walk outward from the square, which stops at the first piece: whether that piece takes there is then a
/// lookup of how far its kind takes along the line. Asking whether a square is attacked walks each line once, not once
/// for each kind that has it.
struct Ray {
    d: i32,
    /// The farthest any kind takes along it, i32::MAX for no limit.
    farthest: i32,
    /// Per kind, the farthest it takes along the line (0 not at all): by its sure steps, by any step, by its sure
    /// steps a piece may take only unmoved, and by any such step.
    reach: Vec<[i32; 4]>,
}

impl Ray {
    fn lines(kinds: &[Kind], side: usize) -> Vec<Ray> {
        let mut rays: Vec<Ray> = Vec::new();
        for (ki, k) in kinds.iter().enumerate() {
            for s in &k.steps {
                if s.may != 0 && s.may != 2 {
                    continue;
                }
                let far = if s.rides == 0 { i32::MAX } else { s.rides };
                for &d in &s.deltas[side] {
                    let at = match rays.iter().position(|r| r.d == d) {
                        Some(i) => i,
                        None => {
                            rays.push(Ray { d, farthest: 0, reach: vec![[0; 4]; kinds.len()] });
                            rays.len() - 1
                        }
                    };
                    let ray = &mut rays[at];
                    ray.farthest = ray.farthest.max(far);
                    let base = if s.only_unmoved { 2 } else { 0 };
                    let r = &mut ray.reach[ki];
                    r[base + 1] = r[base + 1].max(far);
                    if s.sure {
                        r[base] = r[base].max(far);
                    }
                }
            }
        }
        rays
    }

    fn through(rays: &[Ray], size: usize, span: i32) -> Vec<u128> {
        if rays.len() > 128 {
            return Vec::new();
        }
        let mut table = vec![0u128; 2 * size + 1];
        for (i, ray) in rays.iter().enumerate() {
            for n in 1..=ray.farthest.min(span) {
                let apart = n as i64 * ray.d as i64;
                if apart.unsigned_abs() as usize > size {
                    break;
                }
                table[(apart + size as i64) as usize] |= 1u128 << i;
            }
        }
        table
    }
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
                    sure: s.sure,
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
            let mut emptying: [Vec<i32>; 2] = [vec![0; size], vec![0; size]];
            if let Some(table) = w.emptying.get(&k.letter) {
                for r in 0..f.rows {
                    for c in 0..f.columns {
                        let v = table.get(r as usize).and_then(|row| row.get(c as usize)).copied().unwrap_or(0.0)
                            .round() as i32;
                        emptying[0][((r + border) * stride + c + border) as usize] = v;
                        emptying[1][((f.rows - 1 - r + border) * stride + c + border) as usize] = v;
                    }
                }
            }
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
            let overlapping = {
                let span = f.rows.max(f.columns);
                let mut reached = std::collections::HashSet::new();
                let mut twice = false;
                for s in steps.iter().filter(|s| s.may != 3) {
                    for &d in &s.deltas[0] {
                        let most = if s.rides == 0 { span } else { s.rides.min(span) };
                        for n in 1..=most {
                            twice |= !reached.insert(n as i64 * d as i64);
                        }
                    }
                }
                twice
            };
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
                overlapping,
                worth,
                mobility,
                squares,
                emptying,
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
                if l.count.iter().any(|row| row.len() != l.hidden) {
                    return Err("learned: a count's row is not the hidden layer's size".into());
                }
                let count = l.count.iter().flatten().copied().collect();
                Some(Learned { hidden: l.hidden, first, first_bias: l.first_bias.clone(), out: l.out.clone(),
                               out_bias: l.out_bias, count })
            }
        };
        Ok(Game {
            rows: f.rows,
            cols: f.columns,
            border,
            stride,
            size,
            count_to_draw: f.count_to_draw,
            repeats_to_draw: f.repeats_to_draw,
            no_move_draws: f.no_move_unattacked == "draw",
            takes_first: f.takes_first,
            arrival_wins: f.kinds.iter().any(|k| k.wins_arriving),
            tempo: w.tempo.round() as i32,
            royal_pressed: w.royal_pressed.round() as i32,
            start_count: f.start_count,
            any_emptying: f.start_count > 0 && w.emptying.values().any(|t| t.iter().flatten().any(|v| *v != 0.0)),
            beyond: {
                let mut beyond = [0f32; BEYOND];
                for (i, v) in w.beyond.iter().take(BEYOND).enumerate() {
                    beyond[i] = *v as f32;
                }
                beyond
            },
            margin: AtomicI32::new(0),
            learned,
            habits: w.habits,
            zob_piece,
            zob_unmoved,
            zob_passable,
            zob_turn,
            on_board,
            through: {
                let span = f.rows.max(f.columns);
                [Ray::through(&Ray::lines(&kinds, 0), size, span), Ray::through(&Ray::lines(&kinds, 1), size, span)]
            },
            rays: [Ray::lines(&kinds, 0), Ray::lines(&kinds, 1)],
            taking_over: kinds.iter().enumerate().filter(|(_, k)| k.takes_over).map(|(i, _)| i).collect(),
            endings: None,
            endings_most: 0,
            kinds,
        })
    }
    fn is_royal(&self, c: u8) -> bool {
        c != EMPTY && c != OFF && self.kinds[kind_of(c)].royal
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
    /// The move moved, took or made a royal piece: the royal squares are found again when it is taken back.
    royal: bool,
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
    /// The side that does not count on its own unsure ways (2: neither). Its first move may use one, for that move is
    /// checked in the world before it is made; its later moves, and its threats, may not. The other side's unsure ways
    /// all count: what is plausible for them is planned against.
    wary: usize,
    /// Set while the first moves of a search are being listed.
    at_root: bool,
    /// Per side, the squares of its royal pieces: changed in place by a move that moved, took or made one, and back
    /// again when it is taken back.
    royal: [Vec<u16>; 2],
    /// Moves the other side has been seen to make, by square, with the share of its moves each was.
    expected: Vec<(u16, u16, f64)>,
    /// How many pieces stand.
    standing: usize,
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
        let wary = p.wary.map(|w| w as usize).filter(|w| *w < 2).unwrap_or(2);
        let mut b = Board { cells, unmoved, turn: p.turn as usize, passable, count: p.count, hash: 0, history: vec![],
                            wary, at_root: false, royal: [Vec::new(), Vec::new()],
                            expected: p.expected.iter()
                                .map(|&(a, b, c, d, w)| (g.sq(a, b) as u16, g.sq(c, d) as u16, w)).collect(),
                            standing: 0 };
        b.standing = g.on_board.iter().filter(|&&sq| b.cells[sq] != EMPTY).count();
        b.hash = b.full_hash(g);
        b.find_royals(g);
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
        for &ki in &g.taking_over {
            if self.taken_over(sq, by, &g.kinds[ki], code(ki, by)) {
                return true;
            }
        }
        let wary = by == self.wary;
        g.rays[by].iter().any(|ray| self.along(sq, by, ray, wary))
    }

    /// Whether the first piece out from ``sq`` along ``ray`` is one of ``by``'s that takes there along it.
    #[inline(always)]
    fn along(&self, sq: usize, by: usize, ray: &Ray, wary: bool) -> bool {
        let mut t = sq as i32 - ray.d;
        let mut n = 1;
        loop {
            let c = self.cells[t as usize];
            if c == OFF {
                return false;
            }
            if c != EMPTY {
                if side_of(c) != by {
                    return false;
                }
                let r = &ray.reach[kind_of(c)];
                let mut far = if wary { r[0] } else { r[1] };
                if self.unmoved[t as usize] {
                    far = far.max(if wary { r[2] } else { r[3] });
                }
                return n <= far;
            }
            if n >= ray.farthest {
                return false;
            }
            t -= ray.d;
            n += 1;
        }
    }

    /// Whether a royal piece of ``side`` stands attacked after ``m``, which moved no royal piece, where none did
    /// before it: only a line from the royal piece through a square the move changed can have opened.
    fn attacked_after(&self, g: &Game, side: usize, m: &Move, before: bool, moved_royal: bool) -> bool {
        let by = 1 - side;
        let through = &g.through[by];
        if before || moved_royal || through.is_empty() || !g.taking_over.is_empty() {
            return self.in_check(g, side);
        }
        let mut changed = [NONE; 5 + MORE];
        changed[..5].copy_from_slice(&[m.from, m.to, m.taken_at, m.partner_from, m.partner_to]);
        changed[5..5 + m.n_more as usize].copy_from_slice(&m.more[..m.n_more as usize]);
        let wary = by == self.wary;
        for &r in &self.royal[side] {
            let mut lines = 0u128;
            for &v in &changed {
                if v == NONE {
                    continue;
                }
                let apart = r as i64 - v as i64;
                if apart == 0 {
                    return self.in_check(g, side);
                }
                lines |= through[(apart + g.size as i64) as usize];
            }
            while lines != 0 {
                let i = lines.trailing_zeros() as usize;
                lines &= lines - 1;
                if self.along(r as usize, by, &g.rays[by][i], wary) {
                    return true;
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
            // Only a kind two of whose ways can reach one square needs telling which it has counted already.
            let twice = k.overlapping || k.takes_over;
            if twice {
                *stamp = stamp.wrapping_add(1);
                if *stamp == 0 {
                    seen.fill(0);
                    *stamp = 1;
                }
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
                if (s.only_unmoved && !self.unmoved[sq]) || s.may == 3 || s.may == 4 || (!s.sure && side == self.wary) {
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
                        if movable && (!twice || seen[tu] != *stamp) {
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

    fn royals<'b>(&'b self, _g: &'b Game, side: usize) -> impl Iterator<Item = usize> + 'b {
        self.royal[side].iter().map(|&sq| sq as usize)
    }

    fn royal_moved(&mut self, side: usize, from: u16, to: u16) {
        if let Some(at) = self.royal[side].iter_mut().find(|sq| **sq == from) {
            *at = to;
        }
    }

    fn royal_gone(&mut self, side: usize, at: u16) {
        if let Some(i) = self.royal[side].iter().position(|sq| *sq == at) {
            self.royal[side].remove(i);
        }
    }

    fn find_royals(&mut self, g: &Game) {
        self.royal[0].clear();
        self.royal[1].clear();
        for &sq in &g.on_board {
            let c = self.cells[sq];
            if g.is_royal(c) {
                self.royal[side_of(c)].push(sq as u16);
            }
        }
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
        self.royal[side].iter().any(|&sq| self.attacked(g, sq as usize, 1 - side))
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
                if (s.only_unmoved && !self.unmoved[sq]) || s.may == 3 || (!s.sure && side == self.wary && !self.at_root) {
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
                                        if !k.overlapping || !out[start..].contains(&m) {
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
                                    if !k.overlapping || !out[start..].contains(&m) {
                                        out.push(m);
                                    }
                                }
                            } else if !k.overlapping || !out[start..].contains(&base) {
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
            royal: false,
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
        undo.royal = g.is_royal(moved) || g.is_royal(placed) || g.is_royal(undo.captured) || g.is_royal(partner)
            || undo.captured_more[..m.n_more as usize].iter().any(|&c| g.is_royal(c));
        self.standing -= (undo.captured != EMPTY) as usize
            + undo.captured_more[..m.n_more as usize].iter().filter(|&&c| c != EMPTY).count();
        if undo.royal {
            if g.is_royal(undo.captured) {
                self.royal_gone(side_of(undo.captured), m.taken_at);
            }
            for i in 0..m.n_more as usize {
                if g.is_royal(undo.captured_more[i]) {
                    self.royal_gone(side_of(undo.captured_more[i]), m.more[i]);
                }
            }
            if g.is_royal(partner) {
                self.royal_moved(side_of(partner), m.partner_from, m.partner_to);
            }
            let side = side_of(moved);
            match (g.is_royal(moved), g.is_royal(placed)) {
                (true, true) => self.royal_moved(side, m.from, m.to),
                (true, false) => self.royal_gone(side, m.from),
                (false, true) => self.royal[side].push(m.to),
                _ => {}
            }
        }
        undo
    }

    fn unmake(&mut self, g: &Game, u: Undo) {
        let m = u.m;
        let from = m.from as usize;
        let to = m.to as usize;
        let partner = if m.partner_from != NONE { self.cells[m.partner_to as usize] } else { EMPTY };
        if u.royal {
            // The move's changes to the royal squares, undone in the reverse order.
            let placed = self.cells[to];
            let side = side_of(u.moved);
            match (g.is_royal(u.moved), g.is_royal(placed)) {
                (true, true) => self.royal_moved(side, m.to, m.from),
                (true, false) => self.royal[side].push(m.from),
                (false, true) => self.royal_gone(side, m.to),
                _ => {}
            }
            if g.is_royal(partner) {
                self.royal_moved(side_of(partner), m.partner_to, m.partner_from);
            }
            for i in (0..m.n_more as usize).rev() {
                if g.is_royal(u.captured_more[i]) {
                    self.royal[side_of(u.captured_more[i])].push(m.more[i]);
                }
            }
            if g.is_royal(u.captured) {
                self.royal[side_of(u.captured)].push(m.taken_at);
            }
        }
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
        self.standing += (u.captured != EMPTY) as usize
            + u.captured_more[..m.n_more as usize].iter().filter(|&&c| c != EMPTY).count();
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
        // The moves of the position asked about: a first move, checked in the world before it is made.
        let was = self.at_root;
        self.at_root = true;
        self.possible(g, &mut out, false);
        self.at_root = was;
        let side = self.turn;
        let checked = self.in_check(g, side);
        out.into_iter()
            .filter(|&m| {
                let u = self.make(g, m);
                let ok = !self.attacked_after(g, side, &m, checked, u.royal);
                self.unmake(g, u);
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
    /// How far her quick judgement has been off from what looking further found, kept by the side to move and by how
    /// the commonest kind's pieces stand (in sixteenths of a point of judgement): shared by her searchers, and kept with
    /// the table from one move to the next.
    corrections: Vec<AtomicI32>,
    /// What the last search of this game learned of which moves tend to be best: its history and countermoves.
    ordering: Mutex<Option<(Vec<i32>, Vec<Move>)>>,
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
        Table {
            slots: (0..entries.max(1)).map(|_| (AtomicU64::new(0), AtomicU64::new(0))).collect(),
            // A table of one entry is for a single judgement, where nothing is searched to correct it by.
            corrections: if entries > 1 { (0..2 * CORRECTIONS).map(|_| AtomicI32::new(0)).collect() } else { vec![] },
            ordering: Mutex::new(None),
        }
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

/// What a search did, for knowing what it cost and where its time went: hits in the table that settled a position, the
/// positions looked at past the horizon, the furthest ply reached, and each depth completed with its score, whether its
/// best move changed from the depth before, and the time it had taken by then.
#[derive(Default, Clone)]
struct Accounting {
    tt_hits: u64,
    qnodes: u64,
    seldepth: usize,
    iterations: Vec<(i32, i32, bool, u64)>,
}

/// Positions whose judgement one search keeps, at most: a power of two, about a megabyte.
const JUDGEMENTS_KEPT: usize = 1 << 16;

struct Searcher<'a> {
    /// The judgements this search has made, by the position's key: a position met again, by other moves or by a
    /// deeper pass, is not judged again. Everything a judgement reads is in the key (the pieces, which are unmoved,
    /// what may be taken in passing, whose turn), and the search's own setting (who is wary) does not change within it.
    judgements: Vec<(u64, i32)>,
    /// Kept for ordering moves, so a node's ordering allocates nothing.
    keyed: Vec<(std::cmp::Reverse<i64>, u32)>,
    listed: Vec<Move>,
    /// A board's cells, copied to play an exchange out on.
    scratch: Vec<u8>,
    /// Her quick judgement of the position at each ply of the line being looked at, for the side to move there;
    /// -INF where it was not judged (in check).
    stands: Vec<i32>,
    /// Whether the look was taken a ply further at each ply of the line, for her judgements disagreeing there.
    extended: Vec<bool>,
    /// At each ply, the move left out while the others are asked whether any comes near it; NONE-filled for none.
    left_out: Vec<Option<u32>>,
    /// Which searcher beside the first this is (0 for the first): the phase it spreads over depths in.
    helper: usize,
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
    /// The move made at each ply of the line being searched, and per move the answer that last refuted it.
    played: Vec<Move>,
    counter: Vec<Move>,
    /// Which kind has most pieces on the board, whose arrangement her corrections are kept by.
    commonest: usize,
    /// The time a search means to take, where its hard limit (`seconds`) is longer: it stops between depths by how
    /// likely another depth is to change its answer (`Habits`). Nought: it searches until the hard limit.
    soft: f64,
    acc: Accounting,
}

/// How many arrangements of the commonest kind a searcher keeps corrections for, a power of two.
const CORRECTIONS: usize = 1 << 14;

impl<'a> Searcher<'a> {
    /// Where the corrections for this position are kept: by the side to move, and by how the pieces of the kind with
    /// most pieces on the board stand (the kind that changes slowest in most games of pieces, and whose arrangement
    /// decides much of what a quick judgement misses).
    fn correction_at(&mut self) -> usize {
        let g = self.g;
        if self.commonest == usize::MAX {
            let mut counts = vec![0usize; g.kinds.len()];
            for &sq in &g.on_board {
                let c = self.b.cells[sq];
                if c != EMPTY {
                    counts[kind_of(c)] += 1;
                }
            }
            self.commonest = (0..counts.len()).max_by_key(|&k| counts[k]).unwrap_or(0);
        }
        let mut key = 0u64;
        for &sq in &g.on_board {
            let c = self.b.cells[sq];
            if c != EMPTY && kind_of(c) == self.commonest {
                key ^= g.zob_piece[c as usize * g.size + sq];
            }
        }
        self.b.turn * CORRECTIONS + (key as usize & (CORRECTIONS - 1))
    }

    /// Her quick judgement, with what it has been found off by in positions whose commonest pieces stand as these do.
    fn judged(&mut self) -> (i32, usize) {
        let raw = self.evaluate();
        if self.g.habits.corrected <= 0.0 || self.tt.corrections.is_empty() {
            return (raw, usize::MAX);
        }
        let at = self.correction_at();
        let kept = self.tt.corrections[at].load(Ordering::Relaxed);
        (raw + ((kept as f64 / 16.0) * self.g.habits.corrected) as i32, at)
    }

    /// What looking further found against what the quick judgement said, kept: weighed by how far it looked.
    fn found_off_by(&mut self, at: usize, judged: i32, found: i32, depth: i32) {
        if at == usize::MAX || found.abs() >= MATE - 1000 {
            return;
        }
        let unit = self.g.margin.load(Ordering::Relaxed).max(1);
        // At most a few quiet moves' worth: a correction is of a judgement, not a replacement for one.
        let most = 16 * 4 * unit;
        let off = ((found - judged) * 16).clamp(-most, most);
        let weight = (depth + 1).clamp(1, 16);
        let e = self.tt.corrections[at].load(Ordering::Relaxed);
        let now = ((e as i64 * (256 - weight) as i64 + off as i64 * weight as i64) / 256) as i32;
        self.tt.corrections[at].store(now, Ordering::Relaxed);
    }

    /// A drawn game's worth to the side to move: what it is worth to her side where she is to move, the less of it
    /// where they are; nought where the look is wary for neither side.
    fn drawn(&self) -> i32 {
        let worth = self.g.habits.draw_worth;
        if worth == 0.0 || self.b.wary > 1 {
            0
        } else if self.b.turn == self.b.wary {
            worth.round() as i32
        } else {
            -(worth.round() as i32)
        }
    }

    fn evaluate(&mut self) -> i32 {
        let key = self.b.hash;
        let at = key as usize & (JUDGEMENTS_KEPT - 1);
        let (kept_key, kept) = self.judgements[at];
        if kept_key == key && key != 0 {
            return kept;
        }
        let judged = self.evaluate_afresh();
        self.judgements[at] = (key, judged);
        judged
    }

    fn evaluate_afresh(&mut self) -> i32 {
        let g = self.g;
        let mut score = [0i32; 2];
        let mut late = [0i32; 2];
        let mut standing = 0;
        for &sq in &g.on_board {
            let c = self.b.cells[sq];
            if c == EMPTY {
                continue;
            }
            let k = &g.kinds[kind_of(c)];
            let s = side_of(c);
            score[s] += k.worth + k.squares[s][sq];
            if g.any_emptying {
                late[s] += k.emptying[s][sq];
                standing += 1;
            }
        }
        if g.any_emptying {
            let gone = emptied(g, standing);
            for side in 0..2 {
                score[side] += (late[side] as f32 * gone).round() as i32;
            }
        }
        let any_mobility = g.kinds.iter().any(|k| k.mobility != 0);
        let any_exposed = g.kinds.iter().any(|k| k.exposed != 0);
        let any_beyond = g.beyond.iter().any(|w| *w != 0.0);
        let threats = g.habits.double_threats > 0.0 && g.taking_over.is_empty();
        let me = self.b.turn;
        let mut threatened = [NONE; 16];
        let mut n_threatened = 0;
        if any_mobility || any_exposed || any_beyond || g.royal_pressed != 0 || threats {
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
                if threats {
                    // The mover's pieces the other side could take: each is played out below.
                    for &sq in &g.on_board {
                        let c = b.cells[sq];
                        if c != EMPTY && side_of(c) == me && !g.kinds[kind_of(c)].royal && reach.attacks[1 - me][sq] > 0
                            && n_threatened < threatened.len()
                        {
                            threatened[n_threatened] = sq as u16;
                            n_threatened += 1;
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
        let mut second_loss = 0;
        if n_threatened >= 2 {
            let mut scratch = std::mem::take(&mut self.scratch);
            let (mut most, mut next) = (0, 0);
            for &sq in &threatened[..n_threatened] {
                let Some(at) = self.least_taker(&self.b.cells, sq as usize, 1 - me) else { continue };
                let take = Move { from: at as u16, to: sq, taken_at: sq, ..Move::null() };
                let lost = self.exchange(&take, &mut scratch);
                if lost > most {
                    next = most;
                    most = lost;
                } else if lost > next {
                    next = lost;
                }
            }
            self.scratch = scratch;
            second_loss = (next as f64 * g.habits.double_threats) as i32;
        }
        score[me] - score[1 - me] + g.tempo + self.learned_judgement() - second_loss
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
        let mut standing = 0usize;
        for &sq in &g.on_board {
            let c = self.b.cells[sq];
            if c == EMPTY {
                continue;
            }
            standing += 1;
            let rel = (side_of(c) != me) as usize;
            let (r, col) = g.rc(sq);
            let row = if me == 0 { r } else { g.rows - 1 - r };
            let at = ((kind_of(c) * 2 + rel) * squares + (row * g.cols + col) as usize) * l.hidden;
            let w = &l.first[at..at + l.hidden];
            for j in 0..l.hidden {
                h[j] += w[j];
            }
        }
        // How many pieces stand, as one more input.
        let at = standing * l.hidden;
        if at + l.hidden <= l.count.len() {
            for j in 0..l.hidden {
                h[j] += l.count[at + j];
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

    /// What an ending worked out to the end says this position comes to for the side to move: a win or a loss in so
    /// many plies, scored as one found by looking, or nought where neither side can force one. None where no ending
    /// held has these pieces.
    fn known_ending(&self, ply: usize) -> Option<i32> {
        let held = self.g.endings.as_ref()?;
        let g = self.g;
        if self.b.standing > g.endings_most || self.b.standing > 8 {
            return None;
        }
        let mut here = [(0usize, 0usize, 0usize); 8];
        let mut n = 0;
        for &sq in &g.on_board {
            let c = self.b.cells[sq];
            if c != EMPTY {
                let (r, col) = g.rc(sq);
                here[n] = (side_of(c), kind_of(c), (r * g.cols + col) as usize);
                n += 1;
            }
        }
        'endings: for e in held.iter() {
            if e.pieces.len() != n {
                continue;
            }
            let mut index = 0usize;
            let mut i = 0;
            while i < n {
                let piece = e.pieces[i];
                let alike = e.pieces[i..].iter().take_while(|p| **p == piece).count();
                let mut at: Vec<usize> = here[..n].iter().filter(|h| (h.0, h.1) == piece).map(|h| h.2).collect();
                if at.len() != alike {
                    continue 'endings;
                }
                at.sort_unstable();
                for sq in at {
                    index = index * e.squares + sq;
                }
                i += alike;
            }
            let v = *e.values.get(index * 2 + self.b.turn)?;
            if v == 0 {
                return Some(0);
            }
            let plies = (v.unsigned_abs() as usize - 1 + ply) as i32;
            let winner = if v > 0 { 0 } else { 1 };
            return Some(if winner == self.b.turn { MATE - plies } else { -(MATE - plies) });
        }
        None
    }

    /// What taking with ``m`` comes to once both sides have taken back on its square while it pays them, each with
    /// the least worth piece that can, in the worth of kinds. A piece behind another on a line it takes along joins in
    /// when the one in front has gone, and a royal piece takes only where nothing can take it back. A move that is not
    /// one plain take or plain move, or a game whose pieces take over, is worth what it takes. A quiet move comes to
    /// what the piece moved is lost for, where it is.
    fn exchange(&self, m: &Move, cells: &mut Vec<u8>) -> i32 {
        let g = self.g;
        let to = m.to as usize;
        let victim = if m.taken_at == NONE { EMPTY } else { self.b.cells[m.taken_at as usize] };
        if (m.taken_at != NONE && m.taken_at as usize != to) || m.becomes != 0 || m.n_more > 0
            || m.partner_from != NONE || !g.taking_over.is_empty()
        {
            return self.worth_of(victim);
        }
        cells.clear();
        cells.extend_from_slice(&self.b.cells);
        let mut gain = [0i32; 48];
        gain[0] = self.worth_of(victim);
        let mut on_it = cells[m.from as usize];
        cells[m.from as usize] = EMPTY;
        let mut side = 1 - side_of(on_it);
        let mut d = 0;
        while d + 1 < gain.len() {
            let Some(at) = self.least_taker(cells, to, side) else { break };
            let taker = cells[at];
            if g.kinds[kind_of(taker)].royal {
                cells[at] = EMPTY;
                let answered = self.least_taker(cells, to, 1 - side).is_some();
                cells[at] = taker;
                if answered {
                    break;
                }
            }
            d += 1;
            gain[d] = self.worth_of(on_it) - gain[d - 1];
            if (-gain[d - 1]).max(gain[d]) < 0 {
                break;
            }
            cells[at] = EMPTY;
            on_it = taker;
            side = 1 - side;
        }
        while d > 0 {
            gain[d - 1] = -((-gain[d - 1]).max(gain[d]));
            d -= 1;
        }
        gain[0]
    }

    /// The square of the least worth piece of ``by`` that could take on ``sq`` in ``cells``.
    fn least_taker(&self, cells: &[u8], sq: usize, by: usize) -> Option<usize> {
        let g = self.g;
        let wary = by == self.b.wary;
        let mut best: Option<(i32, usize)> = None;
        for ray in &g.rays[by] {
            let mut t = sq as i32 - ray.d;
            let mut n = 1;
            loop {
                let c = cells[t as usize];
                if c == OFF {
                    break;
                }
                if c != EMPTY {
                    if side_of(c) == by {
                        let r = &ray.reach[kind_of(c)];
                        let mut far = if wary { r[0] } else { r[1] };
                        if self.b.unmoved[t as usize] {
                            far = far.max(if wary { r[2] } else { r[3] });
                        }
                        if n <= far {
                            let worth = self.worth_of(c);
                            if best.map_or(true, |(w, _)| worth < w) {
                                best = Some((worth, t as usize));
                            }
                        }
                    }
                    break;
                }
                if n >= ray.farthest {
                    break;
                }
                t -= ray.d;
                n += 1;
            }
        }
        best.map(|(_, at)| at)
    }

    fn order(&mut self, moves: &mut [Move], best: Option<u32>, ply: usize) {
        let mut keyed = std::mem::take(&mut self.keyed);
        let mut scratch = std::mem::take(&mut self.scratch);
        let exchanges = self.g.habits.weighs_exchanges > 0.0;
        keyed.clear();
        keyed.extend(moves.iter().enumerate().map(|(i, &m)| {
                let key: i64 = if best.is_some() && packed(&m) == best {
                    1 << 40
                } else if m.taken_at != NONE {
                    let victim = self.worth_of(self.b.cells[m.taken_at as usize]) as i64;
                    let attacker = self.worth_of(self.b.cells[m.from as usize]) as i64;
                    if exchanges && m.becomes == 0 && self.exchange(&m, &mut scratch) < 0 {
                        // A take that loses once the square is fought over comes after the quiet moves.
                        -(1 << 30) + victim * 64 - attacker / 16
                    } else {
                        (1 << 30) + victim * 64 - attacker / 16
                    }
                } else if m.becomes != 0 {
                    (1 << 30) + self.g.kinds[(m.becomes - 1) as usize].worth as i64 * 64
                } else if ply < MAX_PLY && self.killers[ply][0] == m {
                    1 << 29
                } else if ply < MAX_PLY && self.killers[ply][1] == m {
                    (1 << 29) - 1
                } else if ply > 0 && ply < MAX_PLY && {
                    // The answer that last refuted the move just made, tried after the killers.
                    let p = self.played[ply - 1];
                    p != Move::null() && self.counter[p.from as usize * self.g.size + p.to as usize] == m
                } {
                    (1 << 29) - 2
                } else {
                    self.history[m.from as usize * self.g.size + m.to as usize] as i64
                };
                (std::cmp::Reverse(key), i as u32)
            }));
        // Highest key first, and among equal keys the order they were listed in: a stable sort's order, sorted as
        // small pairs in a buffer the searcher keeps, not as whole moves in a new one.
        keyed.sort_unstable();
        let mut listed = std::mem::take(&mut self.listed);
        listed.clear();
        listed.extend_from_slice(moves);
        for (i, &(_, at)) in keyed.iter().enumerate() {
            moves[i] = listed[at as usize];
        }
        self.listed = listed;
        self.keyed = keyed;
        self.scratch = scratch;
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

    /// A position's worth past the horizon of a look: her judgement of it where she may stand on it, or the best of
    /// the takes, and at the horizon itself (``at_the_horizon``) also of the moves that leave the other side's royal
    /// piece attacked. A move like that leaves them only its answers, so where it leads is decided by them and not by
    /// how the position looks: a look that stops before them misses what it forces.
    /// Measured 23 November 2026 against Stockfish held to depth 5 at 0.2 s a move, 40 games on each of four seeds:
    /// 0.438, 0.525, 0.425 and 0.450 with these followed, 0.388, 0.463, 0.412 and 0.412 without.
    fn quiesce_from(&mut self, alpha: i32, beta: i32, ply: usize, at_the_horizon: bool) -> i32 {
        self.quiesce_inner(alpha, beta, ply, at_the_horizon)
    }

    fn quiesce(&mut self, alpha: i32, beta: i32, ply: usize) -> i32 {
        self.quiesce_inner(alpha, beta, ply, false)
    }

    fn quiesce_inner(&mut self, alpha: i32, beta: i32, ply: usize, with_checks: bool) -> i32 {
        let found = self.quiesce_looked(alpha, beta, ply, with_checks);
        if self.g.habits.keeps_the_horizon > 0.0 && !self.stop && found.abs() < INF {
            let idx = self.tt_index(self.b.hash);
            let entry = self.tt.get(idx);
            if entry.depth <= 0 {
                let flag = if found >= beta { 1 } else if found <= alpha { 2 } else { 0 };
                let mut stored = found;
                if stored > MATE - 1000 {
                    stored += ply as i32;
                } else if stored < -MATE + 1000 {
                    stored -= ply as i32;
                }
                self.tt.set(idx, Entry { key: self.b.hash, depth: 0, flag, score: stored, best: None });
            }
        }
        found
    }

    fn quiesce_looked(&mut self, mut alpha: i32, beta: i32, ply: usize, with_checks: bool) -> i32 {
        self.nodes += 1;
        self.acc.qnodes += 1;
        self.acc.seldepth = self.acc.seldepth.max(ply);
        if self.time_up() {
            return 0;
        }
        {
            // What a search of this position already found, where it settles the bound.
            let entry = self.tt.get(self.tt_index(self.b.hash));
            if entry.key == self.b.hash {
                let mut s = entry.score;
                if s > MATE - 1000 {
                    s -= ply as i32;
                } else if s < -MATE + 1000 {
                    s += ply as i32;
                }
                let settles = match entry.flag {
                    0 => true,
                    1 => s >= beta,
                    2 => s <= alpha,
                    _ => false,
                };
                if settles {
                    self.acc.tt_hits += 1;
                    return s;
                }
            }
        }
        if self.g.arrival_wins && self.b.arrived(self.g, 1 - self.b.turn) {
            return -MATE + ply as i32;
        }
        let ends = self.g.habits.ends_past_the_horizon > 0.0;
        if ply > 0 {
            // As in the search: a position repeated, or counted out, is drawn wherever the look meets it.
            if ends && (self.b.repeated(1) || (self.g.count_to_draw > 0 && self.b.count >= self.g.count_to_draw)) {
                return self.drawn();
            }
            if let Some(known) = self.known_ending(ply) {
                return known;
            }
        }
        let side = self.b.turn;
        let checked = self.b.in_check(self.g, side);
        let mut stand = -INF;
        if !checked {
            stand = self.judged().0;
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
        // At the horizon, every move is generated so that the quiet ones that attack their royal piece can be found.
        let checks_too = with_checks && !checked && self.g.kinds.iter().any(|k| k.royal);
        // Whether their royal piece already stands attacked: only where the search began, for a position reached by a
        // move she let stand never has it (a turn is passed only out of check). Each move then asks only about the
        // lines it changed.
        let theirs_checked = checks_too && ply == 0 && self.b.in_check(self.g, 1 - side);
        self.b.possible(self.g, &mut moves, !checked && !checks_too);
        self.order(&mut moves, None, ply);
        let mut any = false;
        let mut legal_seen = false;
        let mut best = if checked { -MATE + ply as i32 } else { alpha };
        let unit = self.g.margin.load(Ordering::Relaxed);
        let exchanges = self.g.habits.weighs_exchanges > 0.0;
        for m in moves {
            let quiet = m.taken_at == NONE && m.becomes == 0;
            // A take that would leave her short of the bound even with what it takes and two margins cannot raise the
            // bound: not looked at.
            if !checked && unit > 0 && m.becomes == 0 && m.taken_at != NONE
                && stand + self.worth_of(self.b.cells[m.taken_at as usize]) + ((self.g.habits.take_margins * unit as f64) as i32) < alpha
            {
                continue;
            }
            // A take that loses once both sides have taken back on its square is not looked at past the horizon.
            if !checked && exchanges && m.taken_at != NONE && m.becomes == 0 && {
                let mut scratch = std::mem::take(&mut self.scratch);
                let lost = self.exchange(&m, &mut scratch) < 0;
                self.scratch = scratch;
                lost
            } {
                continue;
            }
            let u = self.b.make(self.g, m);
            if self.b.attacked_after(self.g, side, &m, checked, u.royal) {
                self.b.unmake(self.g, u);
                continue;
            }
            legal_seen = true;
            if checks_too && quiet && !self.b.attacked_after(self.g, 1 - side, &m, theirs_checked, u.royal) {
                // A quiet move that forces nothing is past the horizon as before.
                self.b.unmake(self.g, u);
                continue;
            }
            any = true;
            let score = -self.quiesce(-beta, -alpha, ply + 1);
            self.b.unmake(self.g, u);
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
        // A side left no move is not judged as it stands: by the rules it has lost, or drawn. At the horizon, where
        // the look begins past it, a side with no move among those it made is asked whether it has any at all, the
        // ones passed over unmade and the ones not listed among takes included.
        if ends && with_checks && !checked && !legal_seen && !self.has_a_legal_move() {
            return if self.g.no_move_draws { 0 } else { -MATE + ply as i32 };
        }
        best.max(if checked { best } else { alpha }).min(INF)
    }

    /// Whether the side to move has any move its rules allow.
    fn has_a_legal_move(&mut self) -> bool {
        let side = self.b.turn;
        let checked = self.b.in_check(self.g, side);
        let mut moves = Vec::with_capacity(48);
        self.b.possible(self.g, &mut moves, false);
        for m in moves {
            let u = self.b.make(self.g, m);
            let illegal = self.b.attacked_after(self.g, side, &m, checked, u.royal);
            self.b.unmake(self.g, u);
            if !illegal {
                return true;
            }
        }
        false
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
        self.acc.seldepth = self.acc.seldepth.max(ply);
        if self.time_up() {
            return 0;
        }
        let g = self.g;
        if ply > 0 {
            if g.arrival_wins && self.b.arrived(g, 1 - self.b.turn) {
                return -MATE + ply as i32;
            }
            if self.b.repeated(1) || (g.count_to_draw > 0 && self.b.count >= g.count_to_draw) {
                return self.drawn();
            }
            if let Some(known) = self.known_ending(ply) {
                return known;
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
        self.extended[ply] = false;
        if depth == 1 && !checked && ply > 0 && g.habits.unsure_extends > 0.0 && !self.extended[ply - 1]
            && (ply < 2 || !self.extended[ply - 2])
        {
            let unit = g.margin.load(Ordering::Relaxed);
            if unit > 0 && self.learned_judgement().abs() as f64 > g.habits.unsure_extends * unit as f64 {
                depth += 1;
                self.extended[ply] = true;
            }
        }
        if depth <= 0 {
            return self.quiesce_from(alpha, beta, ply, true);
        }
        let key = self.b.hash;
        let idx = self.tt_index(key);
        let entry = self.tt.get(idx);
        let mut tt_move = None;
        let leaving_out = self.left_out[ply];
        if entry.key == key {
            tt_move = entry.best;
            if ply > 0 && entry.depth as i32 >= depth && leaving_out.is_none() {
                let mut s = entry.score;
                if s > MATE - 1000 {
                    s -= ply as i32;
                } else if s < -MATE + 1000 {
                    s += ply as i32;
                }
                let settles = match entry.flag {
                    0 => true,
                    1 => s >= beta,
                    2 => s <= alpha,
                    _ => false,
                };
                if settles {
                    self.acc.tt_hits += 1;
                    return s;
                }
            }
        }
        let pv_node = beta - alpha > 1;
        let unit = g.margin.load(Ordering::Relaxed);
        let habits = g.habits;
        if tt_move.is_none() && habits.unhinted_from > 0.0 && depth as f64 >= habits.unhinted_from && !checked {
            depth -= 1;
        }
        let (stand, corrected_at) = if !checked && (!pv_node || habits.improving > 0.0) {
            self.judged()
        } else {
            (-INF, usize::MAX)
        };
        let corrected_at = if pv_node { usize::MAX } else { corrected_at };
        self.stands[ply] = stand;
        let improving = habits.improving <= 0.0
            || (stand > -INF && (ply < 2 || self.stands[ply - 2] == -INF || stand > self.stands[ply - 2]));
        let stand = if pv_node { -INF } else { stand };
        // A position past the bound by more than a margin for each ply still to look: what those plies could find
        // against it is not worth looking for.
        if !checked && !pv_node && unit > 0 && (depth as f64) <= habits.margin_within && beta.abs() < MATE - 1000
            && stand - unit * depth >= beta
        {
            return stand - unit * depth;
        }
        // Near the end of a look, a position short of the bound by more than two margins a ply is asked only what its
        // takes, and the moves that force, can win back.
        if !checked && !pv_node && unit > 0 && (depth as f64) <= habits.asked_takes_within && alpha.abs() < MATE - 1000
            && stand + ((habits.asked_takes_margins * (unit * depth) as f64) as i32) < alpha
        {
            let q = self.quiesce_from(alpha, alpha + 1, ply, true);
            if self.stop {
                return 0;
            }
            if q <= alpha {
                return q;
            }
        }
        if !checked && null_ok && !pv_node && depth >= 3 && leaving_out.is_none() && self.has_pieces_beyond_irreversible(side) {
            if stand >= beta {
                let above = if habits.passed_turn_margins > 0.0 && unit > 0 {
                    ((stand - beta) as f64 / unit as f64).clamp(0.0, habits.passed_turn_margins)
                } else {
                    0.0
                };
                let r = (habits.passed_turn + depth as f64 / habits.passed_turn_by.max(1.0) + above) as i32;
                if ply < MAX_PLY {
                    self.played[ply] = Move::null();
                }
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
        self.b.at_root = ply == 0;
        self.b.possible(g, &mut moves, false);
        self.b.at_root = false;
        self.order(&mut moves, tt_move, ply);
        let original_alpha = alpha;
        let mut best_score = -INF;
        let mut best_move = None;
        let mut legal = 0;
        let mut quiets_tried: Vec<Move> = Vec::new();
        let theirs_checked = ply == 0 && self.b.in_check(g, 1 - side);
        // Whether the table's best move alone holds the position: every other move, looked at half as far, falls short
        // of what the table says it is worth.
        let mut singular = None;
        if habits.singular_from > 0.0 && ply > 0 && depth as f64 >= habits.singular_from && leaving_out.is_none()
            && entry.key == key && tt_move.is_some() && (entry.flag == 0 || entry.flag == 1)
            && entry.depth as i32 >= depth - 3 && entry.score.abs() < MATE - 1000 && unit > 0
        {
            let short_of = entry.score - (habits.singular_margins * (unit * depth) as f64 / 64.0) as i32;
            self.left_out[ply] = tt_move;
            let others = self.search((depth - 1) / 2, short_of - 1, short_of, ply, false);
            self.left_out[ply] = None;
            if self.stop {
                return 0;
            }
            if others < short_of {
                singular = tt_move;
            }
        }
        for m in moves {
            if ply == 0 && self.excluded.contains(&m) {
                continue;
            }
            if leaving_out.is_some() && packed(&m) == leaving_out {
                continue;
            }
            let further = (singular.is_some() && packed(&m) == singular) as i32;
            let quiet = m.taken_at == NONE && m.becomes == 0;
            let mut u = self.b.make(g, m);
            if self.b.attacked_after(g, side, &m, checked, u.royal) {
                self.b.unmake(self.g, u);
                continue;
            }
            legal += 1;
            if ply < MAX_PLY {
                self.played[ply] = m;
            }
            let gives_check = self.b.attacked_after(g, 1 - side, &m, theirs_checked, u.royal);
            if legal > 1 && !pv_node && !checked && quiet && !gives_check && best_score > -MATE + 1000 {
                // Near the end of a look, a quiet move late in the order, or one that even with a margin cannot lift
                // her to the bound, is not looked at.
                let tried = habits.quiet_tried + (depth * depth) as f64;
                let late = (depth as f64) <= habits.quiet_tried_within
                    && legal as f64 > if improving { tried } else { tried / 2.0 };
                let futile = unit > 0 && (depth as f64) <= habits.futile_within
                    && stand + ((habits.futile_margins * (unit * depth) as f64) as i32) <= alpha;
                if late || futile {
                    self.b.unmake(self.g, u);
                    continue;
                }
            }
            if legal > 1 && !pv_node && !checked && !gives_check && unit > 0 && (depth as f64) <= habits.exchange_within
                && best_score > -MATE + 1000 && m.becomes == 0
            {
                let most = if quiet {
                    habits.exchange_quiet_margins * (unit * depth * depth) as f64
                } else {
                    habits.exchange_takes_margins * (unit * depth) as f64
                };
                self.b.unmake(self.g, u);
                let mut scratch = std::mem::take(&mut self.scratch);
                let comes_to = self.exchange(&m, &mut scratch);
                self.scratch = scratch;
                if (comes_to as f64) < -most {
                    continue;
                }
                u = self.b.make(g, m);
            }
            let mut score;
            if legal == 1 {
                score = -self.search(depth - 1 + further, -beta, -alpha, ply + 1, true);
            } else {
                let mut reduction = 0;
                if depth >= 3 && legal > 3 && quiet && !checked && !gives_check {
                    // The later a move comes in the order and the deeper the look, the less it is likely to be the
                    // one: reduced by both, in proportion to their logarithms.
                    let mut r = habits.shorten_from + (depth as f64).ln() * (legal as f64).ln() / habits.shorten_by.max(0.1);
                    if habits.history_lengthens_by > 0.0 {
                        r -= self.history[m.from as usize * g.size + m.to as usize] as f64 / habits.history_lengthens_by;
                    }
                    reduction = (r as i32 - pv_node as i32 + !improving as i32).clamp(1, depth - 2);
                }
                score = -self.search(depth - 1 + further - reduction, -alpha - 1, -alpha, ply + 1, true);
                if score > alpha && reduction > 0 {
                    score = -self.search(depth - 1 + further, -alpha - 1, -alpha, ply + 1, true);
                }
                if score > alpha && score < beta {
                    score = -self.search(depth - 1 + further, -beta, -alpha, ply + 1, true);
                }
            }
            self.b.unmake(self.g, u);
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
                        // The answer that refuted the move before it is tried early after that move next time.
                        if ply > 0 {
                            let p = self.played[ply - 1];
                            if p != Move::null() {
                                self.counter[p.from as usize * g.size + p.to as usize] = m;
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
            // With its best move left out and no other, nothing comes near it.
            if leaving_out.is_some() {
                return alpha;
            }
            return if checked || !g.no_move_draws { -MATE + ply as i32 } else { self.drawn() };
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
        let replace = (entry.key != key || depth >= entry.depth as i32 || flag == 0) && leaving_out.is_none();
        if replace {
            self.tt.set(idx, Entry { key, depth: depth.min(127) as i8, flag, score: stored,
                                     best: best_move.and_then(|m| packed(&m)) });
        }
        // What looking further found, where it says how far off the quick judgement was: a quiet best move (a take's
        // worth is the take's, not the judgement's), and a bound only where it bounds on the side it was off.
        let best_quiet = best_move.map_or(true, |m| m.taken_at == NONE && m.becomes == 0);
        if corrected_at != usize::MAX && best_quiet && leaving_out.is_none()
            && !(flag == 1 && best_score <= stand) && !(flag == 2 && best_score >= stand)
        {
            self.found_off_by(corrected_at, stand, best_score, depth);
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
            self.b.unmake(self.g, u);
        }
        line
    }
}

impl<'a> Searcher<'a> {
    fn fresh(g: &'a Game, b: Board, tt: &'a Table, seconds: f64, started: Instant, max_nodes: u64) -> Searcher<'a> {
        let mut history = vec![0; g.size * g.size];
        if g.habits.expected_weight > 0.0 {
            for &(from, to, share) in &b.expected {
                history[from as usize * g.size + to as usize] += (share * g.habits.expected_weight) as i32;
            }
        }
        Searcher {
            g,
            b,
            tt,
            halt: None,
            killers: vec![[Move::null(); 2]; MAX_PLY],
            history,
            nodes: 0,
            started,
            seconds,
            stop: false,
            max_nodes,
            excluded: vec![],
            unsteady: 0,
            played: vec![Move::null(); MAX_PLY],
            counter: vec![Move::null(); g.size * g.size],
            commonest: usize::MAX,
            soft: 0.0,
            acc: Accounting::default(),
            judgements: vec![(0, 0); JUDGEMENTS_KEPT],
            keyed: Vec::new(),
            listed: Vec::new(),
            scratch: Vec::new(),
            stands: vec![-INF; MAX_PLY + 1],
            extended: vec![false; MAX_PLY + 1],
            left_out: vec![None; MAX_PLY + 1],
            helper: 0,
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
        let mut last_best: Option<Move> = None;
        let mut window = (-INF, INF);
        // How often the best move has changed between depths, fading, and how many depths it has held.
        let mut changes = 0.0f64;
        let mut held = 0;
        for depth in first.max(1)..=deepest.max(first.max(1)) {
            if self.helper > 0 && self.g.habits.helpers_spread > 0.0 && done > 0 {
                // Each searcher beside the first passes over depths in its own phase: groups of one, two, three and
                // four depths, every other group passed over, each searcher starting at a different point.
                const SIZE: [i32; 20] = [1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4];
                const PHASE: [i32; 20] = [0, 1, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7];
                let i = (self.helper - 1) % 20;
                if ((depth + PHASE[i]) / SIZE[i]) % 2 == 1 {
                    continue;
                }
            }
            let mut score = self.search(depth, window.0, window.1, 0, false);
            // Outside the window, it is widened on that side, twice as far each time, until the score is inside.
            let mut widen = self.g.habits.window.max(1.0) as i32;
            while !self.stop && (score <= window.0 || score >= window.1) {
                widen *= 2;
                if score <= window.0 {
                    window.0 = if widen > 1600 { -INF } else { (score - widen).max(-INF) };
                } else {
                    window.1 = if widen > 1600 { INF } else { (score + widen).min(INF) };
                }
                score = self.search(depth, window.0, window.1, 0, false);
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
            let changed = last_best.is_some_and(|b| b != best);
            self.acc.iterations.push((depth, score, changed, (self.started.elapsed().as_secs_f64() * 1000.0) as u64));
            if self.soft > 0.0 && done > 0 {
                // Between depths, given a soft time: go on while another depth is likely to change the answer.
                let h = self.g.habits;
                changes *= h.unsteady_kept;
                if Some(best) != last_best {
                    changes += 1.0;
                    held = 0;
                } else {
                    held += 1;
                }
                let unit = self.g.margin.load(Ordering::Relaxed).max(1);
                let fell = score < best_score - unit;
                let learned = h.flip_mean > 0.0;
                let wanted = if learned {
                    let fell_by = ((best_score - score) as f64 / unit as f64).clamp(-4.0, 4.0);
                    let z = h.flip_level + h.flip_changes * changes + h.flip_held * held.min(6) as f64
                        + h.flip_fell * fell_by + h.flip_depth * depth as f64;
                    let chance = 1.0 / (1.0 + (-z.clamp(-30.0, 30.0)).exp());
                    self.soft * chance / h.flip_mean
                } else {
                    self.soft * (1.0 + h.unsteady_by * changes) * if fell { 1.0 + h.falling_by } else { 1.0 }
                };
                let elapsed = self.started.elapsed().as_secs_f64();
                if elapsed >= wanted.min(self.seconds)
                    || (!learned && held >= 3 && elapsed >= self.soft * h.steady_share)
                {
                    best_score = score;
                    done = depth;
                    break;
                }
            }
            last_best = Some(best);
            best_score = score;
            done = depth;
            let half = self.g.habits.window.max(1.0) as i32;
            window = if score.abs() < MATE - 1000 { (score - half, score + half) } else { (-INF, INF) };
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
        let mut f = vec![0f32; 2 * k + 2 + k * squares + k + BEYOND + k * squares];
        let standing = g.on_board.iter().filter(|&&sq| self.b.cells[sq] != EMPTY).count() as i32;
        let gone = emptied(g, standing);
        let emptying_at = 2 * k + 2 + k * squares + k + BEYOND;
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
            f[emptying_at + kind * squares + (own_row * g.cols + col) as usize] += sign * gone;
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

/// The share of the pieces at the start that are gone, nought to one: how far the board has emptied.
fn emptied(g: &Game, standing: i32) -> f32 {
    if g.start_count <= 0 {
        return 0.0;
    }
    (1.0 - standing as f32 / g.start_count as f32).clamp(0.0, 1.0)
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
        s.b.unmake(g, u);
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
        b.unmake(g, u);
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
    let mut s = Searcher::fresh(&g, b, &table, 1e9, Instant::now(), 0);
    Ok(s.evaluate())
}

/// One judgement, the game and the weights parsed once: what each position is worth to the side to move, without
/// search, as ``evaluate`` says it. Parsing the weights cost nine tenths of a millisecond a call against a tenth for
/// judging, so a reader of many positions by one judgement holds it here.
#[pyclass(unsendable)]
struct Judgement {
    g: Game,
    table: Table,
}

#[pymethods]
impl Judgement {
    #[new]
    fn new(description: &str, weights: &str) -> PyResult<Self> {
        let f: FamilyD = serde_json::from_str(description).map_err(|e| PyValueError::new_err(format!("description: {e}")))?;
        let w: WeightsD = if weights.trim().is_empty() {
            WeightsD::default()
        } else {
            serde_json::from_str(weights).map_err(|e| PyValueError::new_err(format!("weights: {e}")))?
        };
        let g = Game::new(&f, &w).map_err(PyValueError::new_err)?;
        Ok(Judgement { g, table: Table::new(1) })
    }

    /// What each position is worth to its side to move, in order.
    fn of_each(&self, positions: Vec<String>) -> PyResult<Vec<i32>> {
        let mut judged = Vec::with_capacity(positions.len());
        let mut searcher: Option<Searcher> = None;
        for position in &positions {
            let p: PosD = serde_json::from_str(position).map_err(|e| PyValueError::new_err(format!("position: {e}")))?;
            let b = Board::from(&self.g, &p).map_err(PyValueError::new_err)?;
            let s = match searcher.as_mut() {
                Some(s) => {
                    s.b = b;
                    s
                }
                None => searcher.insert(Searcher::fresh(&self.g, b, &self.table, 1e9, Instant::now(), 0)),
            };
            judged.push(s.evaluate_afresh());
        }
        Ok(judged)
    }

    /// What each position is made of, as self-play's samples are: from the first side's view, for fitting weights to
    /// how games ended. For positions she met in games played in the world, not in her model.
    fn features_of_each(&self, positions: Vec<String>) -> PyResult<Vec<Vec<f32>>> {
        let mut rows = Vec::with_capacity(positions.len());
        let mut searcher: Option<Searcher> = None;
        for position in &positions {
            let p: PosD = serde_json::from_str(position).map_err(|e| PyValueError::new_err(format!("position: {e}")))?;
            let b = Board::from(&self.g, &p).map_err(PyValueError::new_err)?;
            let s = match searcher.as_mut() {
                Some(s) => {
                    s.b = b;
                    s
                }
                None => searcher.insert(Searcher::fresh(&self.g, b, &self.table, 1e9, Instant::now(), 0)),
            };
            rows.push(s.features());
        }
        Ok(rows)
    }
}

/// Positions as her practice reads the ones it keeps for learning how the pieces stand together: for each, which kind
/// stands where as the side to move sees it, her first judgement of it for that side without the learned one, whether
/// it is quiet (a look at the takes alone changes her judgement by no more than one quiet move could), and whose turn it
/// is; and the kinds in the order the first of those counts them. For positions met in the world, not in her model.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn seen_as_practice_sees(
    description: &str,
    positions: Vec<String>,
    weights: &str,
) -> PyResult<(Vec<String>, Vec<(Vec<u16>, i32, bool, u8)>)> {
    let f: FamilyD = serde_json::from_str(description).map_err(|e| PyValueError::new_err(format!("description: {e}")))?;
    let w: WeightsD = if weights.trim().is_empty() {
        WeightsD::default()
    } else {
        serde_json::from_str(weights).map_err(|e| PyValueError::new_err(format!("weights: {e}")))?
    };
    let g = Game::new(&f, &w).map_err(PyValueError::new_err)?;
    let table = Table::new(1 << 12);
    let mut rows = Vec::with_capacity(positions.len());
    for position in &positions {
        let p: PosD = serde_json::from_str(position).map_err(|e| PyValueError::new_err(format!("position: {e}")))?;
        let b = Board::from(&g, &p).map_err(PyValueError::new_err)?;
        measure_the_margin(&g, &b);
        let mut s = Searcher::fresh(&g, b.clone(), &table, 1e9, Instant::now(), 0);
        let judged = s.evaluate();
        let first_look = judged - s.learned_judgement();
        let (stands, _) = s.judged();
        let takes_settled = s.quiesce(-INF, INF, 0);
        let quiet = (takes_settled - stands).abs() <= g.margin.load(Ordering::Relaxed);
        rows.push((s.cells_seen_by_the_mover(), first_look, quiet, b.turn as u8));
    }
    Ok((g.kinds.iter().map(|k| k.letter.clone()).collect(), rows))
}

/// Her search's habits as they were set, before anything of hers moved them: each by name, as weights carry them.
#[pyfunction]
fn habits_as_set() -> PyResult<String> {
    serde_json::to_string(&Habits::default()).map_err(|e| PyValueError::new_err(format!("habits: {e}")))
}

/// Hold endings worked out to the end under ``key``, for searches given that key: each as its pieces (side, kind
/// letter) in the order its situations are named by, its board's rows and columns, and its values as little-endian
/// 16-bit numbers. Held once and kept, so a game's searches do not pass them again.
#[pyfunction]
fn hold_endings(description: &str, key: &str, endings: Vec<(Vec<(usize, String)>, i32, i32, Vec<u8>)>) -> PyResult<usize> {
    let f: FamilyD = serde_json::from_str(description).map_err(|e| PyValueError::new_err(format!("description: {e}")))?;
    let mut held = Vec::new();
    for (pieces, rows, cols, bytes) in endings {
        let mut kinds = Vec::new();
        for (side, letter) in pieces {
            let k = f.kinds.iter().position(|k| k.letter == letter)
                .ok_or_else(|| PyValueError::new_err(format!("unknown kind {letter}")))?;
            kinds.push((side, k));
        }
        let values = bytes.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]])).collect();
        held.push(EndingHeld { pieces: kinds, squares: (rows * cols) as usize, values });
    }
    let n = held.len();
    ENDINGS.get_or_init(|| Mutex::new(HashMap::new())).lock().unwrap().insert(key.to_string(), Arc::new(held));
    Ok(n)
}

/// What the move from (r, c) to (r2, c2) comes to once both sides have taken back on its square while it pays them, in
/// the worth of kinds: its first listed form, where a move has several.
#[pyfunction]
fn exchange(description: &str, position: &str, weights: &str, start: (i32, i32), end: (i32, i32)) -> PyResult<i32> {
    let (g, mut b) = parse(description, position, weights)?;
    let (from, to) = (g.sq(start.0, start.1) as u16, g.sq(end.0, end.1) as u16);
    let m = b.legal(&g).into_iter().find(|m| m.from == from && m.to == to)
        .ok_or_else(|| PyValueError::new_err("no such move"))?;
    let table = Table::new(1);
    let s = Searcher::fresh(&g, b, &table, 1e9, Instant::now(), 0);
    let mut cells = Vec::new();
    Ok(s.exchange(&m, &mut cells))
}

/// The best move found in ``seconds`` (or ``nodes``, when given): a dict with the move, its score for the side to move,
/// the depth completed, the nodes looked at, and the line expected after it.
#[pyfunction]
#[pyo3(signature = (description, position, weights, seconds, deepest=64, hash_mb=64, nodes=0, excluded=vec![], threads=1, soft=0.0, endings=""))]
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
    soft: f64,
    endings: &str,
) -> PyResult<PyObject> {
    let (mut g, b) = parse(description, position, weights)?;
    if !endings.is_empty() {
        g.endings = ENDINGS.get_or_init(|| Mutex::new(HashMap::new())).lock().unwrap().get(endings).cloned();
        g.endings_most = g.endings.as_ref().map_or(0, |e| e.iter().map(|x| x.pieces.len()).max().unwrap_or(0));
    }
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
            main.soft = soft;
            let all = main.b.legal(&g);
            let legal: Vec<Move> = all.iter().copied().filter(|m| !excluded.contains(&move_out(&g, m))).collect();
            if legal.is_empty() {
                return (None, 0, 0, 0u64, vec![], 0, 0, Accounting::default());
            }
            main.excluded = all.into_iter().filter(|m| !legal.contains(m)).collect();
            let kept_ordering = if g.habits.keeps_ordering > 0.0 { table.ordering.lock().unwrap().clone() } else { None };
            if let Some((h, c)) = &kept_ordering {
                if h.len() == main.history.len() && c.len() == main.counter.len() {
                    main.history = h.iter().map(|v| v / 2).collect();
                    main.counter = c.clone();
                }
            }
            let mut helpers: Vec<Searcher> = (1..threads)
                .map(|n| {
                    let mut h = Searcher::fresh(&g, b.clone(), table, seconds, started, 0);
                    h.helper = n;
                    h.halt = Some(&halt);
                    h.excluded = main.excluded.clone();
                    h.history = main.history.clone();
                    h.counter = main.counter.clone();
                    h
                })
                .collect();
            let spread = g.habits.helpers_spread > 0.0;
            let (found, best_score, done) = std::thread::scope(|scope| {
                let mut answers = Vec::new();
                for (n, helper) in helpers.iter_mut().enumerate() {
                    let legal = &legal;
                    answers.push(scope.spawn(move || {
                        helper.best_from(if spread { 1 } else { 1 + (n as i32 % 2) }, deepest, legal)
                    }));
                }
                let mut answer = main.best(deepest, &legal);
                halt.store(true, Ordering::Relaxed);
                for handle in answers {
                    if let Ok(theirs) = handle.join() {
                        // The deepest completed look answers; between two as deep, the better score.
                        if g.habits.deepest_answers > 0.0 && theirs.0.is_some()
                            && (theirs.2 > answer.2 || (theirs.2 == answer.2 && theirs.1 > answer.1))
                        {
                            answer = theirs;
                        }
                    }
                }
                answer
            });
            let best = found.unwrap_or(legal[0]);
            let line = main.line(best, 12);
            let n = main.nodes + helpers.iter().map(|h| h.nodes).sum::<u64>();
            let ms = (started.elapsed().as_secs_f64() * 1000.0) as u64;
            if g.habits.keeps_ordering > 0.0 {
                *table.ordering.lock().unwrap() = Some((main.history.clone(), main.counter.clone()));
            }
            let mut acc = main.acc.clone();
            for h in &helpers {
                acc.tt_hits += h.acc.tt_hits;
                acc.qnodes += h.acc.qnodes;
                acc.seldepth = acc.seldepth.max(h.acc.seldepth);
            }
            (Some(best), best_score, done, n, line, ms, main.unsteady, acc)
        });
        return search_result(py, &g, result);
    }
    let result = py.allow_threads(|| {
        // Searched by time, a search starts where the last one of this game left its table; searched by a count of
        // nodes, it starts from nothing, so that a count means the same thing every time.
        let table = if nodes == 0 { kept_table(description, weights, entries) } else { Arc::new(Table::new(entries)) };
        let mut s = Searcher::fresh(&g, b, &table, seconds, Instant::now(), nodes);
        s.soft = soft;
        let all = s.b.legal(&g);
        let legal: Vec<Move> = all.into_iter().filter(|m| !excluded.contains(&move_out(&g, m))).collect();
        s.excluded = s.b.legal(&g).into_iter().filter(|m| !legal.contains(m)).collect();
        if legal.is_empty() {
            return (None, 0, 0, 0u64, vec![], 0, 0, Accounting::default());
        }
        if g.habits.keeps_ordering > 0.0 && nodes == 0 {
            if let Some((h, c)) = table.ordering.lock().unwrap().clone() {
                if h.len() == s.history.len() && c.len() == s.counter.len() {
                    s.history = h.iter().map(|v| v / 2).collect();
                    s.counter = c;
                }
            }
        }
        let (found, best_score, done) = s.best(deepest, &legal);
        let best = found.unwrap_or(legal[0]);
        let line = s.line(best, 12);
        let n = s.nodes;
        let ms = (s.started.elapsed().as_secs_f64() * 1000.0) as u64;
        if g.habits.keeps_ordering > 0.0 && nodes == 0 {
            *table.ordering.lock().unwrap() = Some((s.history.clone(), s.counter.clone()));
        }
        (Some(best), best_score, done, n, line, ms, s.unsteady, s.acc.clone())
    });
    search_result(py, &g, result)
}

#[allow(clippy::type_complexity)]
fn search_result(py: Python<'_>, g: &Game, result: (Option<Move>, i32, i32, u64, Vec<Move>, u64, i32, Accounting))
                 -> PyResult<PyObject> {
    let d = PyDict::new_bound(py);
    let (best, score, depth, n, line, ms, unsteady, acc) = result;
    d.set_item("tt_hits", acc.tt_hits)?;
    d.set_item("qnodes", acc.qnodes)?;
    d.set_item("seldepth", acc.seldepth)?;
    d.set_item("iterations", acc.iterations)?;
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
#[pyo3(signature = (description, start, weights_a, weights_b, games, nodes_per_move, random_plies=4, seed=1, max_plies=300, sample_every=2, deep_nodes=0, nodes_per_move_b=0, deep_quiet_only=false))]
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
    deep_quiet_only: bool,
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
            let mut sa = Searcher::fresh(&ga, b0.clone(), &ta, 1e9, Instant::now(), nodes_per_move);
            let mut sb = Searcher::fresh(&gb, b0.clone(), &tb, 1e9, Instant::now(), nodes_per_move);
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
                    // What a look at the takes alone begins from: her judgement as corrected by what her looks have
                    // found of it. A position is quiet where no take moves that. LIVE-like 26 November 2026, compared
                    // with her judgement uncorrected instead, two positions a game in her model of chess counted as
                    // quiet once corrections had gathered, against thirty-seven in a seven-square world.
                    let (stands, _) = sa.judged();
                    sa.nodes = 0;
                    sa.stop = false;
                    sa.started = Instant::now();
                    let saved_nodes = sa.max_nodes;
                    sa.max_nodes = 0;
                    let takes_settled = sa.quiesce(-INF, INF, 0);
                    sa.max_nodes = saved_nodes;
                    sample_static.push(first_look);
                    // Quiet where the takes change her judgement by no more than one quiet move could: what is left to
                    // learn there is how the pieces stand. Exactly equal is too strict for a judgement that is not a
                    // plain sum of its parts: LIVE-like 26 November 2026, her learned layer made nine in ten positions
                    // of her model of chess unquiet, where her first guess left nine in ten quiet.
                    measure_the_margin(&ga, &board);
                    let quiet = (takes_settled - stands).abs() <= ga.margin.load(Ordering::Relaxed);
                    sample_quiet.push(quiet);
                    sample_turns.push(board.turn as u8);
                    // With ``deep_nodes``, what a much longer look of hers finds the position worth: her slow
                    // thinking, kept to teach her quick judgement what it would have found.
                    // With ``deep_quiet_only``, only a quiet position is given the longer look: what is gathered to
                    // teach how the pieces stand together keeps only quiet ones, and the rest keep the look the game
                    // gave them.
                    let score = if deep_nodes > 0 && (!deep_quiet_only || quiet) {
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
    m.add_class::<Judgement>()?;
    m.add_function(wrap_pyfunction!(exchange, m)?)?;
    m.add_function(wrap_pyfunction!(hold_endings, m)?)?;
    m.add_function(wrap_pyfunction!(beyond_counts, m)?)?;
    m.add_function(wrap_pyfunction!(search, m)?)?;
    m.add_function(wrap_pyfunction!(self_play, m)?)?;
    m.add_function(wrap_pyfunction!(habits_as_set, m)?)?;
    m.add_function(wrap_pyfunction!(seen_as_practice_sees, m)?)?;
    Ok(())
}
