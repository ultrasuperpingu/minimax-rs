//! Strategy implementations.

pub mod iterative;
#[cfg(not(target_arch = "wasm32"))]
pub mod mcts;
pub mod negamax;
pub mod random;
#[cfg(not(target_arch = "wasm32"))]
pub mod ybw;
pub mod expectiminimax;
pub mod expecti_iterative;

mod common;
#[cfg(not(target_arch = "wasm32"))]
mod sync_util;
pub mod table;
