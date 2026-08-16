use rlgym::{GameState, Terminal};

/// A terminal condition that returns true if any of the given terminal conditions return true.
///
/// # Example
///
/// ```
/// # use rlgymppo::any_terminal;
/// # use rlgymppo::utils::shared_info::SharedInfoReport;
/// # use rlgymppo::utils::terminal::{AnyTerminal, GameEndedCondition, OnGoalCondition};
/// # use rlgymppo::utils::Report;
/// # #[derive(Default)]
/// # struct MySharedInfo { report: Report }
/// # impl SharedInfoReport for MySharedInfo {
/// #     fn report(&mut self) -> &mut Report { &mut self.report }
/// # }
/// let terminal: AnyTerminal<MySharedInfo> = any_terminal![
///     OnGoalCondition,
///     GameEndedCondition<600>,
/// ];
/// ```
#[derive(Default)]
pub struct AnyTerminal<SI> {
    terminals: Vec<Box<dyn Terminal<SI> + Send>>,
}

impl<SI> AnyTerminal<SI> {
    /// Prefer the [`any_terminal!`] macro over calling this directly.
    pub fn new(terminals: Vec<Box<dyn Terminal<SI> + Send>>) -> Self {
        Self { terminals }
    }
}

impl<SI> Terminal<SI> for AnyTerminal<SI> {
    fn reset(&mut self, initial_state: &GameState, shared_info: &mut SI) {
        for terminal in &mut self.terminals {
            terminal.reset(initial_state, shared_info);
        }
    }

    fn is_terminal(&mut self, state: &GameState, shared_info: &mut SI) -> bool {
        self.terminals
            .iter_mut()
            .any(|terminal| terminal.is_terminal(state, shared_info))
    }
}

#[macro_export]
macro_rules! any_terminal {
    ($($terminal:ty),+ $(,)?) => {
        $crate::terminal::AnyTerminal::new(vec![
            $(Box::<$terminal>::default(),)+
        ])
    };
}
