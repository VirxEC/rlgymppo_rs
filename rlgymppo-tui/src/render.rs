use std::collections::HashMap;

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, BorderType, Borders, Padding, Paragraph, Wrap};

use crate::format::{display_width, format_num, metric_value_display};

/// Category grouping information for the display layout.
pub(crate) struct MetricGroup {
    pub(crate) name: &'static str,
    pub(crate) key_prefix: &'static str,
    color: Color,
}

/// Pre-defined groups in display order. Metrics whose keys start with a
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
pub(crate) fn render_header(
    frame: &mut ratatui::Frame,
    area: Rect,
    metrics: &HashMap<String, f64>,
) {
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
pub(crate) fn render_metrics_grid(
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
    let title_width = display_width(view.group.name) + 4;
    let value_col_width = view
        .entries
        .iter()
        .map(|(name, value)| {
            display_width(&metric_value_display(view.group, name, *value, prev_vals))
        })
        .max()
        .unwrap_or(0);
    let row_width = view
        .entries
        .iter()
        .map(|(name, _)| display_width(name) + value_col_width + 7)
        .max()
        .unwrap_or(display_width(" (no data)") + 4);

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
    let value_col_width = entries
        .iter()
        .map(|(name, value)| display_width(&metric_value_display(group, name, *value, prev_vals)))
        .max()
        .unwrap_or(0);
    let inner_width = area.width.saturating_sub(4) as usize;

    for &(name, value) in entries {
        let value = metric_value_display(group, name, value, prev_vals);
        let name_width = display_width(name) + 1;
        let value_width = display_width(&value);
        let min_gap = 2;
        let gap_width = inner_width
            .saturating_sub(name_width + value_width)
            .max(min_gap)
            .max(value_col_width.saturating_sub(value_width) + min_gap);

        let spans = vec![
            Span::styled(format!(" {name}"), Style::default().fg(Color::Gray)),
            Span::raw(" ".repeat(gap_width)),
            Span::styled(
                value,
                Style::default()
                    .fg(group.color)
                    .add_modifier(Modifier::BOLD),
            ),
        ];

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
pub(crate) fn render_status_bar(frame: &mut ratatui::Frame, area: Rect, notification: &str) {
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

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

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
