use std::collections::HashMap;

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, BorderType, Borders, Paragraph, Wrap};

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
    scroll_offset: u16,
) -> u16 {
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
        return 0;
    }

    let num_cols = content_fit_column_count(area.width, &groups, prev_vals);
    let columns = pack_metric_groups(&groups, num_cols);

    let content_height = columns
        .iter()
        .map(|column| desired_column_height(column))
        .max()
        .unwrap_or(0);
    let max_scroll = content_height.saturating_sub(area.height);
    let scroll_offset = scroll_offset.min(max_scroll);

    let col_constraints = vec![Constraint::Ratio(1, num_cols as u32); num_cols];
    let cols = Layout::horizontal(col_constraints).split(area);

    for (column, col_area) in columns.iter().zip(cols.iter()) {
        render_column(frame, *col_area, prev_vals, column, scroll_offset);
    }

    max_scroll
}

/// Render a column of metric groups using content-aware heights.
fn render_column(
    frame: &mut ratatui::Frame,
    area: Rect,
    prev_vals: &HashMap<String, f64>,
    groups: &[MetricGroupView<'_>],
    scroll_offset: u16,
) {
    if groups.is_empty() {
        return;
    }

    let mut lines = Vec::new();

    for view in groups {
        lines.extend(render_group_lines(
            area.width,
            view.group,
            &view.entries,
            prev_vals,
        ));
    }

    frame.render_widget(
        Paragraph::new(Text::from(lines)).scroll((scroll_offset, 0)),
        area,
    );
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

fn desired_column_height(groups: &[MetricGroupView<'_>]) -> u16 {
    groups
        .iter()
        .map(|view| desired_group_height(view.entries.len()))
        .sum()
}

fn group_order(prefix: &str) -> usize {
    GROUPS
        .iter()
        .position(|group| group.key_prefix == prefix)
        .unwrap_or(usize::MAX)
}

/// Render a single metric group as bordered lines inside a scrollable column.
fn render_group_lines<'a>(
    width: u16,
    group: &MetricGroup,
    entries: &[(&str, f64)],
    prev_vals: &HashMap<String, f64>,
) -> Vec<Line<'a>> {
    if width == 0 {
        return Vec::new();
    }

    let width = width as usize;
    if width < 3 {
        return vec![Line::from(" "); desired_group_height(entries.len()) as usize];
    }

    let inner_width = width - 2;
    let title = format!(" {} ", group.name);
    let title_width = display_width(&title);
    let title_fill_width = inner_width.saturating_sub(title_width);
    let mut lines = vec![Line::from(vec![
        Span::raw("╭"),
        Span::styled(
            title,
            Style::default()
                .fg(group.color)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("─".repeat(title_fill_width)),
        Span::raw("╮"),
    ])];

    let value_col_width = entries
        .iter()
        .map(|(name, value)| display_width(&metric_value_display(group, name, *value, prev_vals)))
        .max()
        .unwrap_or(0);

    for &(name, value) in entries {
        let value = metric_value_display(group, name, value, prev_vals);
        let name_text = format!(" {name}");
        let name_width = display_width(&name_text);
        let value_width = display_width(&value);
        let min_gap = 2;
        let gap_width = inner_width
            .saturating_sub(name_width + value_width)
            .max(min_gap)
            .max(value_col_width.saturating_sub(value_width) + min_gap);

        lines.push(Line::from(vec![
            Span::raw("│"),
            Span::styled(name_text, Style::default().fg(Color::Gray)),
            Span::raw(" ".repeat(gap_width)),
            Span::styled(
                value,
                Style::default()
                    .fg(group.color)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("│"),
        ]));
    }

    if entries.is_empty() {
        let placeholder = " (no data)";
        lines.push(Line::from(vec![
            Span::raw("│"),
            Span::styled(
                placeholder,
                Style::default()
                    .fg(Color::DarkGray)
                    .add_modifier(Modifier::ITALIC),
            ),
            Span::raw(" ".repeat(inner_width.saturating_sub(display_width(placeholder)))),
            Span::raw("│"),
        ]));
    }

    lines.push(Line::from(vec![
        Span::raw("╰"),
        Span::raw("─".repeat(inner_width)),
        Span::raw("╯"),
    ]));

    lines
}

/// Render the status bar at the bottom (notifications + key bindings).
pub(crate) fn render_status_bar(
    frame: &mut ratatui::Frame,
    area: Rect,
    notification: &str,
    scroll_offset: u16,
    max_scroll: u16,
) {
    if notification.is_empty() {
        let scroll_hint = if max_scroll == 0 {
            String::new()
        } else {
            format!(" │ wheel/↑/↓ PgUp/PgDn scroll {scroll_offset}/{max_scroll}")
        };
        let text = Line::from(Span::styled(
            format!(" Q:quit S:save R:toggle-render D:toggle-deterministic{scroll_hint} "),
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
