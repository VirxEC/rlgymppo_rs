use std::collections::HashMap;
use std::io::{self, Write};
use std::sync::mpsc::{self, Sender};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crossterm::event::{self, DisableMouseCapture, EnableMouseCapture};
use crossterm::execute;
use crossterm::terminal::{
    self, EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Layout};

use crate::render::{
    MetricHistory, SPARKLINE_HISTORY_LEN, render_header, render_metrics_grid, render_status_bar,
};

/// A terminal-based metrics dashboard using ratatui.
///
/// Create one at the start of training, call [`update`](Self::update) each
/// iteration, call [`notify`](Self::notify) for status messages (save,
/// toggle, etc.), and call [`close`](Self::close) when done.
pub struct TuiDisplay {
    terminal: Terminal<CrosstermBackend<io::Stdout>>,
    notification: String,
    /// Previous iteration's values, used to compute deltas (e.g. Rating).
    prev_vals: HashMap<String, f64>,
    metric_history: MetricHistory,
    scroll_offset: u16,
    max_scroll: u16,
    mouse_capture_enabled: bool,
    closed: bool,
}

/// Thread-owned TUI controller.
pub struct TuiHandle {
    tx: Sender<TuiCommand>,
    thread: Option<JoinHandle<io::Result<()>>>,
}

/// Cloneable sender for immediate TUI notifications from background threads.
#[derive(Clone)]
pub struct TuiNotifier {
    tx: Sender<TuiCommand>,
}

enum TuiCommand {
    Update {
        metrics: HashMap<String, f64>,
        fresh_rating: bool,
    },
    Notify(String),
    Scroll(ScrollCommand),
    DisableMouseCapture,
    Close,
}

#[derive(Clone, Copy)]
pub enum ScrollCommand {
    Up,
    Down,
    MouseUp,
    MouseDown,
    PageUp,
    PageDown,
    Home,
    End,
}

impl TuiDisplay {
    /// Create a new TUI display.
    ///
    /// Enters the alternate screen and enables raw mode so ratatui can
    /// render directly to the terminal.
    pub fn new() -> io::Result<Self> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
        let backend = CrosstermBackend::new(stdout);
        let terminal = Terminal::new(backend)?;
        Ok(Self {
            terminal,
            notification: String::new(),
            prev_vals: HashMap::new(),
            metric_history: HashMap::new(),
            scroll_offset: 0,
            max_scroll: 0,
            mouse_capture_enabled: true,
            closed: false,
        })
    }

    /// Render the latest metrics.
    ///
    /// Call this once per training iteration.  `metrics` should be a flat
    /// `{ "Group/key": value }` map (the same shape used for wandb).
    pub fn update(&mut self, metrics: &HashMap<String, f64>) -> io::Result<()> {
        self.update_with_fresh_rating(metrics, false)
    }

    /// Render the latest metrics and mark `Rating/*` values as freshly updated.
    pub fn update_with_fresh_rating(
        &mut self,
        metrics: &HashMap<String, f64>,
        fresh_rating: bool,
    ) -> io::Result<()> {
        update_metric_history(&mut self.metric_history, metrics, fresh_rating);
        self.render(metrics)?;
        self.prev_vals = metrics.keys().map(|k| (k.clone(), metrics[k])).collect();
        Ok(())
    }

    fn render(&mut self, metrics: &HashMap<String, f64>) -> io::Result<()> {
        let notification = std::mem::take(&mut self.notification);

        // Snapshot for delta computation.
        let prev_vals = self.prev_vals.clone();

        let mut max_scroll = 0;
        let scroll_offset = self.scroll_offset;

        self.terminal.draw(|frame| {
            let area = frame.area();

            let chunks = Layout::vertical([
                Constraint::Length(3), // header
                Constraint::Min(0),    // metrics
                Constraint::Length(1), // status bar
            ])
            .split(area);

            render_header(frame, chunks[0], metrics);
            max_scroll = render_metrics_grid(
                frame,
                chunks[1],
                metrics,
                &prev_vals,
                &self.metric_history,
                scroll_offset,
            );
            render_status_bar(
                frame,
                chunks[2],
                &notification,
                scroll_offset.min(max_scroll),
                max_scroll,
            );
        })?;

        self.max_scroll = max_scroll;
        self.scroll_offset = self.scroll_offset.min(self.max_scroll);

        Ok(())
    }

    /// Set a notification message shown in the status bar for one
    /// iteration cycle (until the next [`update`](Self::update)).
    pub fn notify(&mut self, msg: impl Into<String>) {
        self.notification = msg.into();
    }

    fn scroll_up(&mut self, lines: u16) {
        self.scroll_offset = self.scroll_offset.saturating_sub(lines);
    }

    fn scroll_down(&mut self, lines: u16) {
        self.scroll_offset = self
            .scroll_offset
            .saturating_add(lines)
            .min(self.max_scroll);
    }

    fn scroll_home(&mut self) {
        self.scroll_offset = 0;
    }

    fn scroll_end(&mut self) {
        self.scroll_offset = self.max_scroll;
    }

    pub fn disable_mouse_capture(&mut self) -> io::Result<()> {
        if !self.mouse_capture_enabled {
            return Ok(());
        }

        self.mouse_capture_enabled = false;
        execute!(self.terminal.backend_mut(), DisableMouseCapture)?;
        write!(
            self.terminal.backend_mut(),
            "\x1b[?1000l\x1b[?1002l\x1b[?1003l\x1b[?1006l\x1b[?1015l"
        )?;
        self.terminal.backend_mut().flush()?;
        Ok(())
    }

    /// Close the display and restore the terminal to its normal state.
    pub fn close(&mut self) -> io::Result<()> {
        if self.closed {
            return Ok(());
        }

        self.closed = true;
        self.disable_mouse_capture()?;
        drain_pending_events();
        execute!(self.terminal.backend_mut(), LeaveAlternateScreen)?;
        disable_raw_mode()?;
        self.terminal.show_cursor()?;
        Ok(())
    }
}

impl Drop for TuiDisplay {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

impl TuiHandle {
    /// Spawn a background TUI thread.
    ///
    /// Metric updates can be sent from the training loop with [`update`](Self::update),
    /// while notifications can be sent from any thread via [`TuiNotifier`]. Every
    /// update or notification triggers an immediate redraw using the latest metrics.
    pub fn new() -> io::Result<Self> {
        let (tx, rx) = mpsc::channel();
        let (init_tx, init_rx) = mpsc::sync_channel(1);

        let thread = thread::Builder::new()
            .name("tui".into())
            .spawn(move || {
                let mut display = match TuiDisplay::new() {
                    Ok(display) => {
                        let _ = init_tx.send(Ok(()));
                        display
                    }
                    Err(e) => {
                        let message = e.to_string();
                        let _ = init_tx.send(Err(io::Error::new(e.kind(), message.clone())));
                        return Err(io::Error::new(e.kind(), message));
                    }
                };

                let mut latest_metrics = HashMap::new();
                let mut latest_fresh_rating = false;
                let mut last_size = terminal::size()?;

                loop {
                    let mut should_redraw = false;
                    let mut metrics_updated = false;

                    match rx.recv_timeout(Duration::from_millis(100)) {
                        Ok(TuiCommand::Update {
                            metrics,
                            fresh_rating,
                        }) => {
                            latest_metrics = metrics;
                            latest_fresh_rating = fresh_rating;
                            metrics_updated = true;
                            should_redraw = true;
                        }
                        Ok(TuiCommand::Notify(message)) => {
                            display.notify(message);
                            should_redraw = true;
                        }
                        Ok(TuiCommand::Scroll(command)) => {
                            apply_scroll_command(
                                &mut display,
                                command,
                                last_size.1.saturating_sub(4).max(1),
                            );
                            should_redraw = true;
                        }
                        Ok(TuiCommand::DisableMouseCapture) => {
                            display.disable_mouse_capture()?;
                        }
                        Ok(TuiCommand::Close) => break,
                        Err(mpsc::RecvTimeoutError::Timeout) => {}
                        Err(mpsc::RecvTimeoutError::Disconnected) => break,
                    }

                    while let Ok(command) = rx.try_recv() {
                        match command {
                            TuiCommand::Update {
                                metrics,
                                fresh_rating,
                            } => {
                                latest_metrics = metrics;
                                latest_fresh_rating = fresh_rating;
                                metrics_updated = true;
                                should_redraw = true;
                            }
                            TuiCommand::Notify(message) => {
                                display.notify(message);
                                should_redraw = true;
                            }
                            TuiCommand::Scroll(command) => {
                                apply_scroll_command(
                                    &mut display,
                                    command,
                                    last_size.1.saturating_sub(4).max(1),
                                );
                                should_redraw = true;
                            }
                            TuiCommand::DisableMouseCapture => {
                                display.disable_mouse_capture()?;
                            }
                            TuiCommand::Close => {
                                display.close()?;
                                return Ok(());
                            }
                        }
                    }

                    let size = terminal::size()?;
                    if size != last_size {
                        last_size = size;
                        should_redraw = true;
                    }

                    if should_redraw {
                        if metrics_updated {
                            update_metric_history(
                                &mut display.metric_history,
                                &latest_metrics,
                                latest_fresh_rating,
                            );
                            latest_fresh_rating = false;
                        }
                        display.render(&latest_metrics)?;
                        if metrics_updated {
                            display.prev_vals = latest_metrics
                                .keys()
                                .map(|k| (k.clone(), latest_metrics[k]))
                                .collect();
                        }
                    }
                }

                display.close()
            })
            .map_err(|e| io::Error::other(format!("failed to spawn TUI thread: {e}")))?;

        match init_rx
            .recv()
            .map_err(|e| io::Error::other(format!("TUI thread failed to initialize: {e}")))?
        {
            Ok(()) => Ok(Self {
                tx,
                thread: Some(thread),
            }),
            Err(e) => {
                let _ = thread.join();
                Err(e)
            }
        }
    }

    /// Send a metrics update to the TUI thread. This returns once the update is queued.
    pub fn update(&self, metrics: HashMap<String, f64>) -> io::Result<()> {
        self.update_with_fresh_rating(metrics, false)
    }

    /// Send a metrics update whose `Rating/*` values came from a completed skill eval.
    pub fn update_with_fresh_rating(
        &self,
        metrics: HashMap<String, f64>,
        fresh_rating: bool,
    ) -> io::Result<()> {
        self.tx
            .send(TuiCommand::Update {
                metrics,
                fresh_rating,
            })
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "TUI thread has exited"))
    }

    /// Send a notification to the TUI thread and redraw immediately.
    pub fn notify(&self, msg: impl Into<String>) -> io::Result<()> {
        self.tx
            .send(TuiCommand::Notify(msg.into()))
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "TUI thread has exited"))
    }

    /// Create a cloneable notification sender for use by other threads.
    pub fn notifier(&self) -> TuiNotifier {
        TuiNotifier {
            tx: self.tx.clone(),
        }
    }

    /// Stop the TUI thread and restore the terminal.
    pub fn close(mut self) -> io::Result<()> {
        let _ = self.tx.send(TuiCommand::Close);

        if let Some(thread) = self.thread.take() {
            thread
                .join()
                .map_err(|_| io::Error::other("TUI thread panicked"))??;
        }

        Ok(())
    }
}

impl Drop for TuiHandle {
    fn drop(&mut self) {
        let _ = self.tx.send(TuiCommand::Close);

        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl TuiNotifier {
    /// Send a notification to the TUI thread and redraw immediately.
    pub fn notify(&self, msg: impl Into<String>) -> io::Result<()> {
        self.tx
            .send(TuiCommand::Notify(msg.into()))
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "TUI thread has exited"))
    }

    pub fn scroll(&self, command: ScrollCommand) -> io::Result<()> {
        self.tx
            .send(TuiCommand::Scroll(command))
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "TUI thread has exited"))
    }

    pub fn disable_mouse_capture(&self) -> io::Result<()> {
        self.tx
            .send(TuiCommand::DisableMouseCapture)
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "TUI thread has exited"))
    }
}

fn apply_scroll_command(display: &mut TuiDisplay, command: ScrollCommand, page_height: u16) {
    match command {
        ScrollCommand::Up => display.scroll_up(1),
        ScrollCommand::Down => display.scroll_down(1),
        ScrollCommand::MouseUp => display.scroll_up(3),
        ScrollCommand::MouseDown => display.scroll_down(3),
        ScrollCommand::PageUp => display.scroll_up(page_height),
        ScrollCommand::PageDown => display.scroll_down(page_height),
        ScrollCommand::Home => display.scroll_home(),
        ScrollCommand::End => display.scroll_end(),
    }
}

fn update_metric_history(
    history: &mut MetricHistory,
    metrics: &HashMap<String, f64>,
    fresh_rating: bool,
) {
    for (key, value) in metrics {
        if !should_track_metric_history(key, fresh_rating) || !value.is_finite() {
            continue;
        }

        let values = history.entry(key.clone()).or_default();
        values.push_back(*value);
        while values.len() > SPARKLINE_HISTORY_LEN {
            values.pop_front();
        }
    }
}

fn should_track_metric_history(key: &str, fresh_rating: bool) -> bool {
    let Some((group, _)) = key.split_once('/') else {
        return false;
    };

    (fresh_rating && group == "Rating")
        || matches!(group, "Loss" | "GAE" | "Update")
        || matches!(key, "Collect/avg step reward" | "Collect/episode length")
        || is_custom_sparkline_group(group)
}

fn is_custom_sparkline_group(group: &str) -> bool {
    !matches!(
        group,
        "Collect" | "Cumulative" | "GAE" | "Loss" | "Rating" | "Throughput" | "Timing" | "Update"
    )
}

fn drain_pending_events() {
    while event::poll(Duration::from_millis(0)).unwrap_or(false) {
        let _ = event::read();
    }
}
