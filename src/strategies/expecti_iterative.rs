//! An implementation of iterative deepening evaluation.
//!
//! Search and evaluate at depth 1, then start over at depth 2, then depth 3,
//! etc. Can keep going until a maximum depth or maximum time or either. Uses
//! a transposition table to reuse information from previous iterations.

use crate::IterativeOptions;

//#[cfg(not(target_arch = "wasm32"))]
use crate::strategies::iterative::SearchStopSignal;
use crate::strategies::iterative::Stats;
use crate::strategies::iterative::TranspositionTable;

use super::super::interface::*;
use super::super::util::*;
use super::common::*;
#[cfg(not(target_arch = "wasm32"))]
use super::sync_util::timeout_signal;
use super::table::*;

use instant::Instant;
use rand::prelude::SliceRandom;
use std::cmp::max;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

pub(super) struct ExpectiMinimaxer<E: TurnBasedGameEvaluator, T>
where
    E::G: TurnBasedGame+StochasticGame,
{
    #[cfg(not(target_arch = "wasm32"))]
    timeout: Arc<AtomicBool>,
    #[cfg(target_arch = "wasm32")]
    deadline: Instant,
    #[cfg(target_arch = "wasm32")]
    timeout_counter: u32,
    pub(super) table: T,
    pub(super) countermoves: CounterMoves<E::G>,
    move_pool: MovePool<<E::G as Game>::M>,
    eval: E,

    opts: IterativeOptions,
    pub(crate) stats: Stats,
}

impl<E: TurnBasedGameEvaluator, T: Table<<E::G as Game>::M>> ExpectiMinimaxer<E, T>
where
    E::G: TurnBasedGame+StochasticGame,
    <E::G as Game>::M: Copy + Eq,
{
    pub(super) fn new(table: T, eval: E, opts: IterativeOptions) -> Self {
        Self {
            #[cfg(not(target_arch = "wasm32"))]
            timeout: Arc::new(AtomicBool::new(false)),
            #[cfg(target_arch = "wasm32")]
            deadline: Instant::now(),
            #[cfg(target_arch = "wasm32")]
            timeout_counter: 1000,
            table,
            countermoves: CounterMoves::new(opts.countermove_table, opts.countermove_history_table),
            eval,
            move_pool: MovePool::default(),
            opts,
            stats: Stats::default(),
        }
    }
    /// Returns a handle to the signal used to stop the search.
    /// This should be obtained before starting a search.
    //#[cfg(not(target_arch = "wasm32"))]
    pub(super) fn next_search_stop_signal(&self) -> SearchStopSignal {
        #[cfg(not(target_arch = "wasm32"))]
        {SearchStopSignal(self.timeout.clone())}
        #[cfg(target_arch = "wasm32")]
        {SearchStopSignal::new()}
    }

    //#[cfg(not(target_arch = "wasm32"))]
    //pub(super) fn set_timeout(&mut self, timeout: Arc<AtomicBool>) {
    //    self.timeout = timeout;
    //}

    #[cfg(target_arch = "wasm32")]
    fn reset_timeout(&mut self, duration: Duration) {
        self.timeout_counter = if duration == Duration::new(0, 0) {
            // Too high counter that never hits the maximum.
            1000
        } else {
            0
        };
        self.deadline = Instant::now() + duration;
    }
    #[cfg(not(target_arch = "wasm32"))]
    fn reset_timeout(&mut self, duration: Duration) {
        if duration == Duration::new(0, 0) {
            self.timeout.store(false, Ordering::Relaxed);
        } else {
            timeout_signal(duration, &self.timeout);
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn timeout_check(&mut self) -> bool {
        self.timeout_counter += 1;
        if self.timeout_counter != 100 {
            return false;
        }
        self.timeout_counter = 0;
        Instant::now() >= self.deadline
    }
    #[cfg(not(target_arch = "wasm32"))]
    fn timeout_check(&mut self) -> bool {
        self.timeout.load(Ordering::Relaxed)
    }

    fn null_move_check(
        &mut self, s: &mut <E::G as Game>::S, depth: u8, player_to_move: i8, beta: Evaluation,
    ) -> Option<Evaluation> {
        if let (Some(depth_reduction), Some(null_move)) =
            (self.opts.null_move_depth, E::G::null_move(s))
        {
            // Default to a minimum of depth=1 after null moving.
            if depth > depth_reduction &&
            // If the position already seems pretty awesome.
            self.eval.evaluate(s) >= beta
            {
                // If we just pass and let the opponent play this position (at reduced depth),
                let mut nulled = AppliedMove::<E::G>::new(s, null_move);
                let value =
                    self.expectiminimax(&mut nulled, None, depth - depth_reduction, player_to_move, beta-1, beta)?;
                // is the result still so good that we shouldn't bother with a full search?
                if value >= beta {
                    return Some(value);
                }
            }
        }
        // If we didn't check, return a low value that won't trigger beta cutoff.
        Some(WORST_EVAL)
    }

    // Negamax only among noisy moves.
    fn noisy_negamax(
        &mut self, s: &mut <E::G as Game>::S, depth: u8, player_to_move: i8, mut alpha: Evaluation, beta: Evaluation,
    ) -> Option<Evaluation> {
        if self.timeout_check() {
            return None;
        }
        /*if let Some(winner) = E::G::get_winner(s) {
            return if E::G::current_player(s) == player_to_move {
                    match winner {
                        crate::Winner::PlayerJustMoved => Some(BEST_EVAL),
                        crate::Winner::Draw => Some(0),
                        crate::Winner::PlayerToMove => Some(WORST_EVAL),
                    }
                } else {
                    match winner {
                        crate::Winner::PlayerJustMoved => Some(WORST_EVAL),
                        crate::Winner::Draw => Some(0),
                        crate::Winner::PlayerToMove => Some(BEST_EVAL),
                    }
                };
        }*/
        if let Some(winner) = E::G::get_explicit_winner(s) {
                return match winner {
                    crate::TurnBasedWinner::Player(p) if p == player_to_move => Some(BEST_EVAL),
                    crate::TurnBasedWinner::Player(_) => Some(WORST_EVAL),
                    crate::TurnBasedWinner::Draw => Some(0),
                };
            }
        if depth == 0 {
            return Some(self.eval.evaluate(s));
        }

        let mut moves = self.move_pool.alloc();
        self.eval.generate_noisy_moves(s, &mut moves);
        if moves.is_empty() {
            self.move_pool.free(moves);
            return Some(self.eval.evaluate(s));
        }
        //TODO
        let mut best = WORST_EVAL;
        for m in moves.iter() {
            let mut new = AppliedMove::<E::G>::new(s, *m);
            let value = self.noisy_negamax(&mut new, depth - 1, player_to_move, alpha, beta)?;
            best = max(best, value);
            alpha = max(alpha, value);
            if alpha >= beta {
                break;
            }
        }
        self.move_pool.free(moves);
        Some(best)
    }

    // Recursively compute negamax on the game state. Returns None if it hits the timeout.
    pub(super) fn expectiminimax(
        &mut self, s: &mut <E::G as Game>::S, prev_move: Option<<E::G as Game>::M>, mut depth: u8,
        player_to_move: i8,
        mut alpha: Evaluation, mut beta: Evaluation,
    ) -> Option<Evaluation> {
        if self.timeout_check() {
            return None;
        }

        self.stats.explore_node();

        if depth == 0 {
            // Evaluate quiescence search on leaf nodes.
            // Will just return the node's evaluation if quiescence search is disabled.
            return self.noisy_negamax(s, self.opts.max_quiescence_depth, player_to_move, alpha, beta);
            //return Some(self.eval.evaluate(s));
        }

        let alpha_orig = alpha;
        let hash = E::G::zobrist_hash(s);
        let mut good_move = None;
        if let Some(value) = self.table.check(hash, depth, &mut good_move, &mut alpha, &mut beta) {
            //println!("In cache: {:?} => {}", hash, value);
            return Some(value);
        }

        let mut moves = self.move_pool.alloc();
        if let Some(_winner) = E::G::generate_moves(s, &mut moves) {
            //TODO: this is not ok...
            /*return if E::G::current_player(s) == player_to_move {
                    match winner {
                        crate::Winner::PlayerJustMoved => Some(BEST_EVAL),
                        crate::Winner::Draw => Some(0),
                        crate::Winner::PlayerToMove => Some(WORST_EVAL),
                    }
                } else {
                    match winner {
                        crate::Winner::PlayerJustMoved => Some(WORST_EVAL),
                        crate::Winner::Draw => Some(0),
                        crate::Winner::PlayerToMove => Some(BEST_EVAL),
                    }
                };*/
        }
        self.stats.generate_moves(moves.len());
        if moves.is_empty() {
            self.move_pool.free(moves);
            if player_to_move == E::G::current_player(s) {
                return Some(WORST_EVAL);
            } else {
                return Some(BEST_EVAL);
            }
        }

        if self.null_move_check(s, depth, player_to_move, beta)? >= beta {
            println!("null move");
            return Some(beta);
        }

        // TODO: Also do a pre-search to look for moves much better than others.
        if self.opts.singular_extension && moves.len() == 1 {
            depth += 1;
        }

        // Reorder moves.
        if depth >= self.opts.min_reorder_moves_depth {
            // TODO reorder moves
        }
        self.countermoves.reorder(prev_move, &mut moves);
        if let Some(good) = good_move {
            move_to_front(good, &mut moves);
        }

        let mut best;
        let mut best_move = moves[0];
        let mut null_window = false;
        let mut need_store = true;
        if E::G::is_random_move(s) {
            need_store = false;
            let mut value = 0.0;

            for m in moves.iter() {
                let p = <E::G as StochasticGame>::get_probability(s, *m);
                let mut new = AppliedMove::<E::G>::new(s, *m);
                let score = self.expectiminimax(&mut new, prev_move, depth - 1, player_to_move, alpha, beta)?;
                value += p * score as f32;
            }
            
            best = value.round() as Evaluation;
        } else if E::G::current_player(s) == player_to_move {
            best = WORST_EVAL;
            for &m in moves.iter() {
                let mut new = AppliedMove::<E::G>::new(s, m);
                let value = if null_window {
                    let probe = self.expectiminimax(&mut new, Some(m), depth - 1, player_to_move, alpha, alpha+1)?;
                    if probe > alpha && probe < beta {
                        // Full search fallback.
                        self.expectiminimax(&mut new, Some(m), depth - 1, player_to_move, probe, beta)?
                    } else {
                        probe
                    }
                } else {
                    self.expectiminimax(&mut new, Some(m), depth - 1, player_to_move, alpha, beta)?
                };
                if value > best {
                    best = value;
                    best_move = m;
                }
                if value >= alpha {
                    alpha = value;
                    // Now that we've found a good move, assume following moves
                    // are worse, and seek to cull them without full evaluation.
                    null_window = self.opts.null_window_search;
                }
                if best >= beta {
                    self.countermoves.update(prev_move, m);
                    break;
                }
            }
        } else {
            best = BEST_EVAL;
            for &m in moves.iter() {
                let mut new = AppliedMove::<E::G>::new(s, m);
                let value = if null_window {
                    let probe = self.expectiminimax(&mut new, Some(m), depth - 1, player_to_move, alpha, alpha+1)?;
                    if probe > alpha && probe < beta {
                        // Full search fallback.
                        self.expectiminimax(&mut new, Some(m), depth - 1, player_to_move, probe, beta)?
                    } else {
                        probe
                    }
                } else {
                    self.expectiminimax(&mut new, Some(m), depth - 1, player_to_move, alpha, beta)?
                };
                if value < best {
                    best = value;
                    best_move = m;
                }
                if best < beta {
                    beta = value;
                    // Now that we've found a good move, assume following moves
                    // are worse, and seek to cull them without full evaluation.
                    null_window = self.opts.null_window_search;
                }
                if best <= alpha {
                    self.countermoves.update(prev_move, m);
                    break;
                }
            }
        }
        if need_store { //TODO: check if we need to store a random
            self.table.update(hash, alpha_orig, beta, depth, best, best_move);
        }
        self.move_pool.free(moves);
        Some(clamp_value(best))
    }

    // Try to find the value within a window around the estimated value.
    // Results, whether exact, overshoot, or undershoot, are stored in the table.
    pub(super) fn aspiration_search(
        &mut self, s: &mut <E::G as Game>::S, depth: u8, player_to_move: i8, target: Evaluation, window: Evaluation,
    ) -> Option<()> {
        if depth < 2 {
            // Do a full search on shallow nodes to establish the target.
            return Some(());
        }
        let alpha = max(target.saturating_sub(window), WORST_EVAL);
        let beta = target.saturating_add(window);
        self.expectiminimax(s, None, depth, player_to_move, alpha, beta)?;
        Some(())
    }

    pub(super) fn search_and_reorder(
        &mut self, s: &mut <E::G as Game>::S, moves: &mut [ValueMove<<E::G as Game>::M>], depth: u8, player_to_move: i8, 
    ) -> Option<Evaluation> {
        let mut alpha = WORST_EVAL;
        let beta = BEST_EVAL;
        for value_move in moves.iter_mut() {
            let mut new = AppliedMove::<E::G>::new(s, value_move.m);
            let value = self.expectiminimax(&mut new, Some(value_move.m), depth - 1, player_to_move, alpha, beta)?;

            alpha = max(alpha, value);
            value_move.value = value;
        }
        moves.sort_by_key(|vm| -vm.value);
        self.table.update(E::G::zobrist_hash(s), alpha, beta, depth, moves[0].value, moves[0].m);
        Some(moves[0].value)
    }
}

pub struct ExpectiIterativeSearch<E: TurnBasedGameEvaluator>
where
    E::G: TurnBasedGame+StochasticGame,
{
    max_depth: u8,
    max_time: Duration,
    minimaxer: ExpectiMinimaxer<E, TranspositionTable<<E::G as Game>::M>>,
    prev_value: Evaluation,
    opts: IterativeOptions,

    // Runtime stats for the last move generated.

    // Maximum depth used to produce the move.
    actual_depth: u8,
    // Nodes explored at each depth.
    nodes_explored: Vec<u64>,
    pv: Vec<<E::G as Game>::M>,
    wall_time: Duration,
}

impl<E: TurnBasedGameEvaluator> ExpectiIterativeSearch<E>
where
    E::G: TurnBasedGame+StochasticGame,
    <E::G as Game>::M: Copy + Eq,
    <E::G as Game>::S: Clone,
{
    pub fn new(eval: E, opts: IterativeOptions) -> ExpectiIterativeSearch<E> {
        let table = TranspositionTable::new(opts.table_byte_size, opts.strategy);
        let negamaxer = ExpectiMinimaxer::new(table, eval, opts);
        ExpectiIterativeSearch {
            max_depth: 99,
            max_time: Duration::from_secs(5),
            prev_value: 0,
            minimaxer: negamaxer,
            opts,
            actual_depth: 0,
            nodes_explored: Vec::new(),
            pv: Vec::new(),
            wall_time: Duration::default(),
        }
    }

    /// Return a human-readable summary of the last move generation.
    pub fn stats(&self, s: &mut <E::G as Game>::S) -> String {
        let total_nodes_explored: u64 = self.nodes_explored.iter().sum();
        let mean_branching_factor = self.minimaxer.stats.total_generated_moves as f64
            / self.minimaxer.stats.total_generate_move_calls as f64;
        let effective_branching_factor = (*self.nodes_explored.last().unwrap_or(&0) as f64)
            .powf((self.actual_depth as f64 + 1.0).recip());
        let throughput = (total_nodes_explored + self.minimaxer.stats.nodes_explored) as f64
            / self.wall_time.as_secs_f64();
        format!(
            "Principal variation: {}\nExplored {} nodes to depth {}. MBF={:.1} EBF={:.1}\nPartial exploration of next depth hit {} nodes.\n{} nodes/sec",
            pv_string::<E::G>(&self.pv[..], s),
            total_nodes_explored,
            self.actual_depth,
            mean_branching_factor,
            effective_branching_factor,
            self.minimaxer.stats.nodes_explored,
            throughput as usize
        )
    }

    /// Return the options used in this search.
    pub fn options(&self) -> &IterativeOptions {
        &self.opts
    }
    /// Return the search options used in this search.
    pub fn get_max_depth(&self) -> u8 {
        self.max_depth
    }
    /// Return the search options used in this search.
    pub fn get_max_time(&self) -> &Duration {
        &self.max_time
    }
    
    /// Returns a handle to the signal used to stop the search.
    /// This should be obtained before starting a search.
    //#[cfg(not(target_arch = "wasm32"))]
    pub fn next_search_stop_signal(&self) -> SearchStopSignal {
        self.minimaxer.next_search_stop_signal()
    }

    #[doc(hidden)]
    pub fn root_value(&self) -> Evaluation {
        unclamp_value(self.prev_value)
    }

}

impl<E: TurnBasedGameEvaluator> Strategy<E::G> for ExpectiIterativeSearch<E>
where
    E::G: TurnBasedGame+StochasticGame,
    <E::G as Game>::S: Clone,
    <E::G as Game>::M: Copy + Eq,
{
    fn choose_move(&mut self, s: &<E::G as Game>::S) -> Option<<E::G as Game>::M> {
        self.minimaxer.table.advance_generation();
        self.minimaxer.countermoves.advance_generation(E::G::null_move(s));
        // Reset stats.
        self.nodes_explored.clear();
        self.minimaxer.stats.reset();
        self.actual_depth = 0;
        let start_time = Instant::now();
        // Start timer if configured.
        self.minimaxer.reset_timeout(self.max_time);

        let root_hash = E::G::zobrist_hash(s);
        let mut s_clone = s.clone();
        let mut best_move = None;
        let mut interval_start;
        let player_to_move = E::G::current_player(s);
        self.minimaxer.eval.set_evaluated_player(player_to_move);
        // Store the moves so they can be reordered every iteration.
        let mut moves = Vec::new();
        if E::G::generate_moves(&s_clone, &mut moves).is_some() {
            return None;
        }
        // Start in a random order.
        moves.shuffle(&mut rand::rng());
        let mut moves = moves.into_iter().map(|m| ValueMove::new(0, m)).collect::<Vec<_>>();

        // Start at 1 or 2 to hit the max depth.
        let mut depth = self.max_depth % self.opts.step_increment;
        if depth == 0 {
            depth = self.opts.step_increment;
        }
        while depth <= self.max_depth {
            interval_start = Instant::now();
            let search = {
                if let Some(window) = self.opts.aspiration_window {
                    // Results of the search are stored in the table.
                    if self
                        .minimaxer
                        .aspiration_search(&mut s_clone, depth, player_to_move, self.prev_value, window)
                        .is_none()
                    {
                        // Timeout.
                        break;
                    }
                    if self.opts.verbose {
                        if let Some(entry) = self.minimaxer.table.lookup(root_hash) {
                            let end = Instant::now();
                            let interval = end - interval_start;
                            eprintln!(
                                "Iterative aspiration depth{:>2} took{:>5}ms; bounds{:>5}; bestmove={}",
                                depth,
                                interval.as_millis(),
                                entry.bounds(),
                                move_id::<E::G>(&s_clone, entry.best_move)
                            );
                            interval_start = end;
                        }
                    }
                }

                self.minimaxer.search_and_reorder(&mut s_clone, &mut moves[..], depth, player_to_move)
            };
            if search.is_none() {
                // Timeout. Return the best move from the previous depth.
                break;
            }
            let entry = self.minimaxer.table.lookup(root_hash).unwrap();
            best_move = entry.best_move;

            if self.opts.verbose {
                let interval = Instant::now() - interval_start;
                eprintln!(
                    "Iterative fullsearch depth{:>2} took{:>5}ms; value{:>6}; bestmove={}",
                    depth,
                    interval.as_millis(),
                    entry.value_string(),
                    move_id::<E::G>(&s_clone, best_move)
                );
            }

            self.actual_depth = max(self.actual_depth, depth);
            self.nodes_explored.push(self.minimaxer.stats.nodes_explored);
            self.minimaxer.stats.nodes_explored = 0;
            self.prev_value = entry.value;
            depth += self.opts.step_increment;
            self.minimaxer.table.populate_pv::<E::G>(&mut self.pv, &s_clone);
            if unclamp_value(entry.value).abs() == BEST_EVAL {
                break;
            }
        }
        self.wall_time = start_time.elapsed();
        if self.opts.verbose {
            let mut s_clone = s.clone();
            eprintln!("{}", self.stats(&mut s_clone));
        }
        best_move
    }

    fn set_timeout(&mut self, max_time: Duration) {
        self.max_time = max_time;
        self.max_depth = 99;
    }

    fn set_max_depth(&mut self, depth: u8) {
        self.max_depth = depth;
        self.max_time = Duration::new(0, 0);
    }

    fn set_depth_or_timeout(&mut self, depth: u8, max_time: Duration) {
        self.max_time = max_time;
        self.max_depth = depth;
    }

    fn principal_variation(&self) -> Vec<<E::G as Game>::M> {
        self.pv.clone()
    }
}
