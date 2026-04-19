#[macro_use]
extern crate bencher;
extern crate minimax;
#[path = "../examples/dice_race.rs"]
mod dice_race;

use bencher::Bencher;
use minimax::{
    strategies::{
        expecti_iterative::ExpectiIterativeSearch, expecti_ybw::ParallelSearch,
        expectiminimax::ExpectiMinimax,
    },
    *,
};

fn bench_expectiminimax(b: &mut Bencher) {
    let board = dice_race::Board::default();
    b.iter(|| {
        let mut s = ExpectiMinimax::new(dice_race::DiceRaceEvaluator::default(), 8);
        let m = s.choose_move(&board);
        assert!(m.is_some());
    });
}

fn bench_expecti_iterative(b: &mut Bencher) {
    let board = dice_race::Board::default();
    b.iter(|| {
        let mut s = ExpectiIterativeSearch::new(
            dice_race::DiceRaceEvaluator::default(),
            IterativeOptions::new().with_table_byte_size(32_000),
        );
        s.set_max_depth(8);
        let m = s.choose_move(&board);
        assert!(m.is_some());
    });
}

#[cfg(not(target_arch = "wasm32"))]
fn bench_expecti_parallel(b: &mut Bencher) {
    let board = dice_race::Board::default();
    b.iter(|| {
        let mut s = ParallelSearch::new(
            dice_race::DiceRaceEvaluator::default(),
            IterativeOptions::new().with_table_byte_size(32_000),
            ParallelOptions::new(),
        );
        s.set_max_depth(8);
        let m = s.choose_move(&board);
        assert!(m.is_some());
    });
}
#[cfg(not(target_arch = "wasm32"))]
benchmark_group!(benches, bench_expectiminimax, bench_expecti_iterative, bench_expecti_parallel);
#[cfg(target_arch = "wasm32")]
benchmark_group!(benches, bench_expectiminimax, bench_expecti_iterative);
benchmark_main!(benches);
