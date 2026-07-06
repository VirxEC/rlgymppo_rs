//! Terminal-based metrics dashboard for rlgymppo — "local-only wandb".
//!
//! This crate provides a [`TuiDisplay`] that renders training metrics in a
//! ratatui alternate-screen TUI, updating after each iteration.  It is
//! designed to run alongside the cloud-based wandb logger, or standalone.

use std::collections::HashMap;
use std::io;
use std::sync::mpsc::{self, Sender};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crossterm::execute;
use crossterm::terminal::{
    self, EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, BorderType, Borders, Padding, Paragraph, Wrap};

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
    Update(HashMap<String, f64>),
    Notify(String),
    Close,
}

/// Category grouping information for the display layout.
struct MetricGroup {
    name: &'static str,
    key_prefix: &'static str,
    color: Color,
}

/// Pre-defined groups in display order.  Metrics whose keys start with a
/// group's `key_prefix + "/"` are shown under that group.
const GROUPS: &[MetricGroup] = &[
    MetricGroup {
        name: "Collect",
        key_prefix: "Collect",
        color: Color::Cyan,
    },
    MetricGroup {
        name: "GAE",
        key_prefix: "GAE",
        color: Color::Green,
    },
    MetricGroup {
        name: "Loss",
        key_prefix: "Loss",
        color: Color::Red,
    },
    MetricGroup {
        name: "Update",
        key_prefix: "Update",
        color: Color::Yellow,
    },
    MetricGroup {
        name: "Timing",
        key_prefix: "Timing",
        color: Color::Magenta,
    },
    MetricGroup {
        name: "Throughput",
        key_prefix: "Throughput",
        color: Color::Blue,
    },
    MetricGroup {
        name: "Cumulative",
        key_prefix: "Cumulative",
        color: Color::White,
    },
    MetricGroup {
        name: "Rating",
        key_prefix: "Rating",
        color: Color::LightYellow,
    },
];

impl TuiDisplay {
    /// Create a new TUI display.
    ///
    /// Enters the alternate screen and enables raw mode so ratatui can
    /// render directly to the terminal.
    pub fn new() -> io::Result<Self> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(stdout);
        let terminal = Terminal::new(backend)?;
        Ok(Self {
            terminal,
            notification: String::new(),
            prev_vals: HashMap::new(),
            closed: false,
        })
    }

    /// Render the latest metrics.
    ///
    /// Call this once per training iteration.  `metrics` should be a flat
    /// `{ "Group/key": value }` map (the same shape used for wandb).
    pub fn update(&mut self, metrics: &HashMap<String, f64>) -> io::Result<()> {
        let notification = std::mem::take(&mut self.notification);

        // Snapshot for delta computation.
        let prev_vals = self.prev_vals.clone();

        self.terminal.draw(|frame| {
            let area = frame.area();

            // ── Layout ──────────────────────────────────────────────
            //  1. header bar (title + cumulative stats)
            //  2. main metrics grid (fills remaining space)
            //  3. status bar (notifications / key-bindings hint)
            // ─────────────────────────────────────────────────────────
            let chunks = Layout::vertical([
                Constraint::Length(3), // header
                Constraint::Min(0),    // metrics
                Constraint::Length(1), // status bar
            ])
            .split(area);

            // ── Header ─────────────────────────────────────────────
            render_header(frame, chunks[0], metrics);

            // ── Metrics grid ───────────────────────────────────────
            render_metrics_grid(frame, chunks[1], metrics, &prev_vals);

            // ── Status bar ─────────────────────────────────────────
            render_status_bar(frame, chunks[2], &notification);
        })?;

        // Stash current values for delta computation next iteration.
        self.prev_vals = metrics.keys().map(|k| (k.clone(), metrics[k])).collect();

        Ok(())
    }

    /// Set a notification message shown in the status bar for one
    /// iteration cycle (until the next [`update`](Self::update)).
    pub fn notify(&mut self, msg: impl Into<String>) {
        self.notification = msg.into();
    }

    /// Close the display and restore the terminal to its normal state.
    pub fn close(&mut self) -> io::Result<()> {
        if self.closed {
            return Ok(());
        }

        self.closed = true;
        disable_raw_mode()?;
        execute!(self.terminal.backend_mut(), LeaveAlternateScreen)?;
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
                let mut last_size = terminal::size()?;

                loop {
                    let mut should_redraw = false;

                    match rx.recv_timeout(Duration::from_millis(100)) {
                        Ok(TuiCommand::Update(metrics)) => {
                            latest_metrics = metrics;
                            should_redraw = true;
                        }
                        Ok(TuiCommand::Notify(message)) => {
                            display.notify(message);
                            should_redraw = true;
                        }
                        Ok(TuiCommand::Close) => break,
                        Err(mpsc::RecvTimeoutError::Timeout) => {}
                        Err(mpsc::RecvTimeoutError::Disconnected) => break,
                    }

                    while let Ok(command) = rx.try_recv() {
                        match command {
                            TuiCommand::Update(metrics) => {
                                latest_metrics = metrics;
                                should_redraw = true;
                            }
                            TuiCommand::Notify(message) => {
                                display.notify(message);
                                should_redraw = true;
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
                        display.update(&latest_metrics)?;
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
        self.tx
            .send(TuiCommand::Update(metrics))
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
}

// ── Helpers ──────────────────────────────────────────────────────────

/// Build a sorted list of `(display_name, value)` pairs for a group.
fn group_entries<'a>(metrics: &'a HashMap<String, f64>, prefix: &str) -> Vec<(&'a str, f64)> {
    let mut entries: Vec<(&str, f64)> = metrics
        .iter()
        .filter_map(|(k, v)| k.strip_prefix(&format!("{prefix}/")).map(|name| (name, *v)))
        .collect();
    entries.sort_unstable_by(|a, b| a.0.cmp(b.0));
    entries
}

/// Render the header bar (title + key cumulative numbers).
fn render_header(frame: &mut ratatui::Frame, area: Rect, metrics: &HashMap<String, f64>) {
    let steps = metrics
        .get("Cumulative/steps")
        .map(|v| format_num(*v))
        .unwrap_or_default();
    let updates = metrics
        .get("Cumulative/updates")
        .map(|v| format_num(*v))
        .unwrap_or_default();

    let header_text = Text::from(Line::from(vec![
        Span::styled(
            " rlgymppo ",
            Style::default()
                .fg(Color::White)
                .bg(Color::Blue)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("  "),
        Span::styled(format!("Steps: {steps}"), Style::default().fg(Color::Cyan)),
        Span::raw(" │ "),
        Span::styled(
            format!("Iteration: {updates}"),
            Style::default().fg(Color::Green),
        ),
    ]));

    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded);
    frame.render_widget(
        Paragraph::new(header_text)
            .block(block)
            .wrap(Wrap { trim: false }),
        area,
    );
}

/// Render all metric groups in a responsive grid.
///
/// The grid chooses the largest content-fitting column count, then greedily packs
/// groups by their desired height so dense groups do not all land in one column.
fn render_metrics_grid(
    frame: &mut ratatui::Frame,
    area: Rect,
    metrics: &HashMap<String, f64>,
    prev_vals: &HashMap<String, f64>,
) {
    let groups: Vec<MetricGroupView<'_>> = GROUPS
        .iter()
        .filter_map(|group| {
            let entries = group_entries(metrics, group.key_prefix);
            if entries.is_empty() {
                None
            } else {
                Some(MetricGroupView { group, entries })
            }
        })
        .collect();

    if groups.is_empty() || area.width == 0 || area.height == 0 {
        return;
    }

    let num_cols = content_fit_column_count(area.width, &groups, prev_vals);
    let columns = pack_metric_groups(&groups, num_cols);

    let col_constraints = vec![Constraint::Ratio(1, num_cols as u32); num_cols];
    let cols = Layout::horizontal(col_constraints).split(area);

    for (column, col_area) in columns.iter().zip(cols.iter()) {
        render_column(frame, *col_area, prev_vals, column);
    }
}

/// Render a column of metric groups using content-aware heights.
fn render_column(
    frame: &mut ratatui::Frame,
    area: Rect,
    prev_vals: &HashMap<String, f64>,
    groups: &[MetricGroupView<'_>],
) {
    if groups.is_empty() {
        return;
    }

    let desired_heights: Vec<u16> = groups
        .iter()
        .map(|view| desired_group_height(view.entries.len()))
        .collect();
    let desired_total: u16 = desired_heights.iter().sum();

    let constraints: Vec<Constraint> = if desired_total <= area.height {
        desired_heights
            .into_iter()
            .map(Constraint::Length)
            .chain(std::iter::once(Constraint::Min(0)))
            .collect()
    } else {
        groups
            .iter()
            .map(|view| {
                Constraint::Ratio(
                    desired_group_height(view.entries.len()) as u32,
                    desired_total as u32,
                )
            })
            .collect()
    };

    let rows = Layout::vertical(constraints).split(area);

    for (i, view) in groups.iter().enumerate() {
        render_group(frame, rows[i], view.group, &view.entries, prev_vals);
    }
}

#[derive(Clone)]
struct MetricGroupView<'a> {
    group: &'a MetricGroup,
    entries: Vec<(&'a str, f64)>,
}

fn content_fit_column_count(
    width: u16,
    groups: &[MetricGroupView<'_>],
    prev_vals: &HashMap<String, f64>,
) -> usize {
    for num_cols in (1..=groups.len()).rev() {
        let constraints = vec![Constraint::Ratio(1, num_cols as u32); num_cols];
        let layout_area = Rect::new(0, 0, width, 1);
        let col_areas = Layout::horizontal(constraints).split(layout_area);
        let columns = pack_metric_groups(groups, num_cols);

        let all_columns_fit = columns.iter().zip(col_areas.iter()).all(|(column, area)| {
            column
                .iter()
                .all(|view| required_group_width(view, prev_vals) <= area.width)
        });

        if all_columns_fit {
            return num_cols;
        }
    }

    1
}

fn pack_metric_groups<'a>(
    groups: &[MetricGroupView<'a>],
    num_cols: usize,
) -> Vec<Vec<MetricGroupView<'a>>> {
    let mut columns: Vec<Vec<MetricGroupView<'_>>> = (0..num_cols).map(|_| Vec::new()).collect();
    let mut col_heights = vec![0_u16; num_cols];

    // Pack tallest groups first for balanced columns, then restore each
    // column's original display order for predictable scanning.
    let mut group_indices: Vec<usize> = (0..groups.len()).collect();
    group_indices
        .sort_by_key(|&idx| std::cmp::Reverse(desired_group_height(groups[idx].entries.len())));

    for idx in group_indices {
        let col_idx = col_heights
            .iter()
            .enumerate()
            .min_by_key(|(_, height)| **height)
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        col_heights[col_idx] =
            col_heights[col_idx].saturating_add(desired_group_height(groups[idx].entries.len()));
        columns[col_idx].push(groups[idx].clone());
    }

    for column in &mut columns {
        column.sort_by_key(|view| group_order(view.group.key_prefix));
    }

    columns
}

fn required_group_width(view: &MetricGroupView<'_>, prev_vals: &HashMap<String, f64>) -> u16 {
    let title_width = view.group.name.chars().count() + 4;
    let row_width = view
        .entries
        .iter()
        .map(|(name, value)| {
            let value = metric_value_display(view.group, name, *value, prev_vals);
            name.chars().count() + value.chars().count() + 7
        })
        .max()
        .unwrap_or(" (no data)".chars().count() + 4);

    title_width.max(row_width).min(u16::MAX as usize) as u16
}

fn desired_group_height(entry_count: usize) -> u16 {
    // Top/bottom border + title/padding, then one row per metric.
    (entry_count as u16).saturating_add(2).max(3)
}

fn group_order(prefix: &str) -> usize {
    GROUPS
        .iter()
        .position(|group| group.key_prefix == prefix)
        .unwrap_or(usize::MAX)
}

/// Render a single metric group as a bordered block.
fn render_group(
    frame: &mut ratatui::Frame,
    area: Rect,
    group: &MetricGroup,
    entries: &[(&str, f64)],
    prev_vals: &HashMap<String, f64>,
) {
    let mut lines = Vec::new();

    for &(name, value) in entries {
        let mut spans = vec![
            Span::styled(format!(" {name}"), Style::default().fg(Color::Gray)),
            Span::raw("  "),
        ];

        spans.push(Span::styled(
            metric_value_display(group, name, value, prev_vals),
            Style::default()
                .fg(group.color)
                .add_modifier(Modifier::BOLD),
        ));

        lines.push(Line::from(spans));
    }

    // If there are no entries, show a placeholder.
    if lines.is_empty() {
        lines.push(Line::from(Span::styled(
            " (no data)",
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::ITALIC),
        )));
    }

    let block = Block::default()
        .title(Line::from(Span::styled(
            format!(" {} ", group.name),
            Style::default()
                .fg(group.color)
                .add_modifier(Modifier::BOLD),
        )))
        .borders(Borders::ALL)
        .padding(Padding::new(1, 1, 0, 0));

    frame.render_widget(
        Paragraph::new(Text::from(lines))
            .block(block)
            .wrap(Wrap { trim: false }),
        area,
    );
}

/// Render the status bar at the bottom (notifications + key bindings).
fn render_status_bar(frame: &mut ratatui::Frame, area: Rect, notification: &str) {
    if notification.is_empty() {
        let text = Line::from(Span::styled(
            " Q:quit S:save R:toggle-render D:toggle-deterministic ",
            Style::default().fg(Color::DarkGray),
        ));
        frame.render_widget(Paragraph::new(text), area);
    } else {
        let text = Line::from(Span::styled(
            format!(" {notification}"),
            Style::default().fg(Color::Yellow),
        ));
        frame.render_widget(Paragraph::new(text), area);
    }
}

fn metric_value_display(
    group: &MetricGroup,
    name: &str,
    value: f64,
    prev_vals: &HashMap<String, f64>,
) -> String {
    if group.key_prefix == "Rating" {
        // Show delta alongside the current rating (1 decimal).
        let full_key = format!("Rating/{name}");
        let prev = prev_vals.get(&full_key).copied().unwrap_or(value);
        let delta = value - prev;
        let val_str = format!("{value:.1}");
        if delta.abs() >= 0.05 {
            let sign = if delta >= 0.0 { '+' } else { '-' };
            let delta_str = format!("{:.1}", delta.abs());
            format!("{val_str} ({sign}{delta_str})")
        } else {
            val_str
        }
    } else {
        format_num(value)
    }
}

/// Format a number for display (commas for integers, 4-decimal for floats).
fn format_num(val: f64) -> String {
    if val == (val as i64) as f64 {
        let int_val = val as i64;
        // Add thousands separators.
        let s = int_val.to_string();
        let mut result = String::with_capacity(s.len() + s.len() / 3);
        let negative = int_val < 0;
        let abs_s = if negative { &s[1..] } else { &s };
        for (i, ch) in abs_s.chars().enumerate() {
            if i > 0 && (abs_s.len() - i) % 3 == 0 {
                result.push(',');
            }
            result.push(ch);
        }
        if negative {
            result.insert(0, '-');
        }
        result
    } else if val.abs() < 1e-3 && val != 0.0 {
        format!("{val:.4e}")
    } else {
        format!("{val:.4}")
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use super::*;

    #[test]
    fn test_format_num_integer() {
        assert_eq!(format_num(1_234_567.0), "1,234,567");
        assert_eq!(format_num(0.0), "0");
        assert_eq!(format_num(-1000.0), "-1,000");
    }

    #[test]
    fn test_format_num_float() {
        assert_eq!(format_num(PI), "3.1416");
        assert_eq!(format_num(0.00123), "0.0012");
    }

    #[test]
    fn test_group_entries() {
        let mut m = HashMap::new();
        m.insert("Collect/avg step reward".into(), 1.23);
        m.insert("Collect/timesteps".into(), 1000.0);
        m.insert("GAE/time".into(), 0.5);

        let entries = group_entries(&m, "Collect");
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].0, "avg step reward");
        assert_eq!(entries[0].1, 1.23);
    }
}
