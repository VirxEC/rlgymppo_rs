//! Terminal-based metrics dashboard for rlgymppo — "local-only wandb".
//!
//! This crate provides a [`TuiDisplay`] that renders training metrics in a
//! ratatui alternate-screen TUI, updating after each iteration.  It is
//! designed to run alongside the cloud-based wandb logger, or standalone.

use std::{collections::HashMap, io};

use crossterm::{
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, BorderType, Borders, Padding, Paragraph, Wrap},
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

/// Render all metric groups in a two-column grid.  Groups that have no
/// entries in `metrics` are hidden to keep the display clean.
fn render_metrics_grid(
    frame: &mut ratatui::Frame,
    area: Rect,
    metrics: &HashMap<String, f64>,
    prev_vals: &HashMap<String, f64>,
) {
    // Only keep groups that currently have data.
    let populated: Vec<&MetricGroup> = GROUPS
        .iter()
        .filter(|g| !group_entries(metrics, g.key_prefix).is_empty())
        .collect();

    if populated.is_empty() {
        return;
    }

    // Split into two columns, balancing entries.
    let mid = populated.len().div_ceil(2);
    let left_groups = &populated[..mid];
    let right_groups = &populated[mid..];

    let cols =
        Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)]).split(area);

    render_column(frame, cols[0], metrics, prev_vals, left_groups);
    render_column(frame, cols[1], metrics, prev_vals, right_groups);
}

/// Render a column of metric groups.
fn render_column(
    frame: &mut ratatui::Frame,
    area: Rect,
    metrics: &HashMap<String, f64>,
    prev_vals: &HashMap<String, f64>,
    groups: &[&MetricGroup],
) {
    // Each group gets an equal vertical slice.
    let constraints: Vec<Constraint> = groups
        .iter()
        .map(|_| Constraint::Ratio(1, groups.len() as u32))
        .collect();
    let rows = Layout::vertical(constraints).split(area);

    for (i, group) in groups.iter().enumerate() {
        let entries = group_entries(metrics, group.key_prefix);
        render_group(frame, rows[i], group, &entries, prev_vals);
    }
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

        if group.key_prefix == "Rating" {
            // Show delta alongside the current rating (1 decimal).
            let full_key = format!("Rating/{name}");
            let prev = prev_vals.get(&full_key).copied().unwrap_or(value);
            let delta = value - prev;
            let val_str = format!("{value:.1}");
            if delta.abs() >= 0.05 {
                let sign = if delta >= 0.0 { '+' } else { '-' };
                let delta_str = format!("{:.1}", delta.abs());
                spans.push(Span::styled(
                    format!("{val_str} ({sign}{delta_str})"),
                    Style::default()
                        .fg(group.color)
                        .add_modifier(Modifier::BOLD),
                ));
            } else {
                spans.push(Span::styled(
                    val_str,
                    Style::default()
                        .fg(group.color)
                        .add_modifier(Modifier::BOLD),
                ));
            }
        } else {
            spans.push(Span::styled(
                format_num(value),
                Style::default()
                    .fg(group.color)
                    .add_modifier(Modifier::BOLD),
            ));
        }

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
